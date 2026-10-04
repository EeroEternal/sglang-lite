# Copyright 2026 sglang-lite contributors
# Licensed under the Apache License, Version 2.0.
"""GPU mathematical operators, without SGLang/vLLM runtime imports.

Gated-residual/PLE formulas follow SGLang v0.5.20, commit
94602c9c2b7cbdb8efd5c52802dac6a1c180089e (Apache-2.0).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def rms(x, weight, eps=1e-6, group=None, gemma=True):
    shape = x.shape
    y = x.float()
    if group:
        y = y.reshape(*shape[:-1], -1, group)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    scale = weight.float()
    if gemma:
        scale = scale + 1.0
    y = y.reshape(shape) * scale
    return y.to(x.dtype)


@torch.compile(fullgraph=True)
def hc_mix(x, norm, down, up, hidden, branches):
    normalized = rms(x, norm, group=hidden)
    gates = torch.sigmoid(F.linear(F.silu(F.linear(normalized, down) / branches), up))
    mixed = (gates.reshape(1, branches, hidden) * normalized.reshape(1, branches, hidden)).mean(-2)
    return mixed, normalized


@torch.compile(fullgraph=True)
def hc_combine(output, residual, normalized, inject, hidden, branches):
    gate = 2 * torch.sigmoid(F.linear(normalized, inject) / branches)
    return (
        residual.reshape(1, branches, hidden) + output.unsqueeze(-2) * gate.unsqueeze(-1)
    ).flatten(-2)


@torch.compile(fullgraph=True)
def local_greedy_candidate(logits, vocab_start):
    score, index = logits.flatten().max(0)
    return torch.stack([score, (index + vocab_start).float()])


def greedy_from_candidates(candidates):
    # Gather order is rank order, so argmax ties keep the lowest global ID.
    return candidates[:, 1].index_select(0, candidates[:, 0].argmax().reshape(1)).long()


@torch.compile(fullgraph=True)
def gdn_update(q, k, v, z, b, a, bias, log_a, state, norm):
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(3, dim=0)
    k = k.repeat_interleave(3, dim=0)
    decay = torch.exp(-torch.exp(log_a.float()) * F.softplus(a.float() + bias.float()))
    beta = torch.sigmoid(b.float())
    updated = state * decay[:, None, None]
    delta = (v.float() - (updated * k[:, None, :]).sum(-1)) * beta[:, None]
    updated = updated + delta[:, :, None] * k[:, None, :]
    output = (updated * q[:, None, :]).sum(-1) * (128**-0.5)
    state.copy_(updated)
    output = output.to(v.dtype).float()
    output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-6)
    return (output * norm.float() * torch.sigmoid(z.float())).to(v.dtype).reshape(1, -1)


@triton.jit
def _gdn_conv_sum(projected, history, weight, output, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < CHANNELS
    h0 = tl.load(history + rows * 3, mask=valid, other=0).to(tl.float32)
    h1 = tl.load(history + rows * 3 + 1, mask=valid, other=0).to(tl.float32)
    h2 = tl.load(history + rows * 3 + 2, mask=valid, other=0).to(tl.float32)
    value = tl.load(projected + rows, mask=valid, other=0).to(tl.float32)
    w0 = tl.load(weight + rows * 4, mask=valid, other=0)
    w1 = tl.load(weight + rows * 4 + 1, mask=valid, other=0)
    w2 = tl.load(weight + rows * 4 + 2, mask=valid, other=0)
    w3 = tl.load(weight + rows * 4 + 3, mask=valid, other=0)
    # Match ATen's four-tap shuffle reduction, with separately rounded products.
    result = (h0 * w0 + h2 * w2) + (h1 * w1 + value * w3)
    tl.store(output + rows, result, mask=valid)
    tl.store(history + rows * 3, h1, mask=valid)
    tl.store(history + rows * 3 + 1, h2, mask=valid)
    tl.store(history + rows * 3 + 2, value, mask=valid)


def gdn_conv_sum(projected, history, weight):
    channels = history.shape[0]
    if (
        history.shape != (channels, 3)
        or weight.shape != (channels, 4)
        or projected.numel() != channels
        or weight.dtype != torch.float32
        or projected.dtype != history.dtype
        or not all(t.is_contiguous() for t in (projected, history, weight))
    ):
        raise ValueError(
            "GDN conv requires contiguous matching projections/history and FP32 weights"
        )
    output = torch.empty(channels, device=projected.device, dtype=torch.float32)
    _gdn_conv_sum[(triton.cdiv(channels, 256),)](
        projected, history, weight, output, channels, 256, enable_fp_fusion=False
    )
    return output


@triton.jit
def _normalize_topk(probabilities, ids, normalized, int_ids, COUNT: tl.constexpr):
    slots = tl.arange(0, 16)
    values = tl.load(probabilities + slots, mask=slots < COUNT, other=0)
    denominator = tl.sum(values, 0)
    result = tl.div_rn(values, denominator)
    indices = tl.load(ids + slots, mask=slots < COUNT, other=0).to(tl.int32)
    tl.store(normalized + slots, result, mask=slots < COUNT)
    tl.store(int_ids + slots, indices, mask=slots < COUNT)


def normalize_topk(probabilities, ids):
    if (
        probabilities.shape != (1, 10)
        or ids.shape != probabilities.shape
        or probabilities.dtype != torch.float32
        or ids.dtype != torch.int64
        or not probabilities.is_contiguous()
        or not ids.is_contiguous()
    ):
        raise ValueError("routing normalization requires contiguous FP32/Int64 single-token top-10")
    normalized = torch.empty_like(probabilities)
    int_ids = torch.empty_like(ids, dtype=torch.int32)
    _normalize_topk[(1,)](
        probabilities, ids, normalized, int_ids, 10, num_warps=1, enable_fp_fusion=False
    )
    return normalized, int_ids


@triton.jit
def _store_qsa_compressed(compressed, position, value, WIDTH: tl.constexpr):
    columns = tl.arange(0, WIDTH)
    token = tl.load(position)
    tl.store(
        compressed + (token // 4) * WIDTH + columns,
        tl.load(value + columns),
        mask=token % 4 == 3,
    )


def store_qsa_compressed(compressed, position, value):
    if (
        compressed.ndim != 2
        or compressed.shape[1] != 128
        or value.shape != (1, 128)
        or compressed.dtype != value.dtype
        or position.shape != (1,)
        or not all(t.is_cuda and t.is_contiguous() for t in (compressed, position, value))
    ):
        raise ValueError("QSA compressed update requires contiguous CUDA blocks and one position")
    _store_qsa_compressed[(1,)](compressed, position, value, 128, num_warps=4)


@triton.jit
def _qsa_block_mean(index_raw, position, output, CAPACITY: tl.constexpr):
    columns = tl.arange(0, 128)
    block = (tl.load(position) // 4) * 4
    a = tl.load(index_raw + tl.minimum(block, CAPACITY - 1) * 128 + columns).to(tl.float32)
    b = tl.load(index_raw + tl.minimum(block + 1, CAPACITY - 1) * 128 + columns).to(tl.float32)
    c = tl.load(index_raw + tl.minimum(block + 2, CAPACITY - 1) * 128 + columns).to(tl.float32)
    d = tl.load(index_raw + tl.minimum(block + 3, CAPACITY - 1) * 128 + columns).to(tl.float32)
    tl.store(output + columns, ((a + b) + (c + d)) * 0.25)


def qsa_block_mean(index_raw, position):
    if (
        index_raw.ndim != 2
        or index_raw.shape[1] != 128
        or index_raw.shape[0] % 4
        or index_raw.dtype != torch.bfloat16
        or position.shape != (1,)
        or not all(t.is_cuda and t.is_contiguous() for t in (index_raw, position))
    ):
        raise ValueError("QSA block mean requires aligned contiguous CUDA BF16 index state")
    output = torch.empty((1, 128), device=index_raw.device, dtype=index_raw.dtype)
    _qsa_block_mean[(1,)](
        index_raw, position, output, index_raw.shape[0], num_warps=4, enable_fp_fusion=False
    )
    return output


@triton.jit
def _table_lookup(
    table,
    ids,
    output,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    row = tl.load(ids + head) - START
    valid = (row >= 0) & (row < ROWS) & (columns < WIDTH)
    values = tl.load(table + row * WIDTH + columns, mask=valid, other=0.0).to(tl.bfloat16)
    tl.store(output + head * WIDTH + columns, values, mask=columns < WIDTH)


def table_lookup(table, ids, start):
    output = torch.empty((ids.numel(), table.shape[1]), device=table.device, dtype=torch.bfloat16)
    _table_lookup[(ids.numel(),)](
        table,
        ids,
        output,
        start,
        table.shape[0],
        table.shape[1],
        triton.next_power_of_2(table.shape[1]),
    )
    return output


def rope(x, cos, sin):
    half = cos.shape[-1]
    first, second = x[..., :half].float(), x[..., half : 2 * half].float()
    rotated = torch.cat([first * cos - second * sin, second * cos + first * sin], -1)
    return torch.cat([rotated.to(x.dtype), x[..., 2 * half :]], -1)


@triton.jit
def _attention(
    q,
    keys,
    values,
    positions,
    selected,
    output,
    CAPACITY: tl.constexpr,
    SELECTED: tl.constexpr,
    BUDGET: tl.constexpr,
    DENSE_LIMIT: tl.constexpr,
    DIM: tl.constexpr = 256,
    BLOCK: tl.constexpr = 32,
):
    head = tl.program_id(0)
    position = tl.load(positions).to(tl.int32)
    if DENSE_LIMIT:
        tl.device_assert(position >= 0, "negative dense attention position")
        tl.device_assert(position < DENSE_LIMIT, "dense attention execution limit exceeded")
    columns = tl.arange(0, DIM)
    query = tl.load(q + head * DIM + columns).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((DIM,), 0.0, tl.float32)
    length = position + 1 if DENSE_LIMIT else (position + 1 if position < BUDGET else SELECTED)
    for start in range(tl.cdiv(length, BLOCK)):
        rows = start * BLOCK + tl.arange(0, BLOCK)
        if DENSE_LIMIT or position < BUDGET:
            slots = rows
        else:
            slots = tl.load(selected + rows, mask=rows < SELECTED, other=-1)
        valid = (rows < length) & (slots >= 0) & (slots <= position) & (slots < CAPACITY)
        k = tl.load(
            keys + slots[:, None] * DIM + columns[None, :], mask=valid[:, None], other=0
        ).to(tl.float32)
        v = tl.load(
            values + slots[:, None] * DIM + columns[None, :], mask=valid[:, None], other=0
        ).to(tl.float32)
        scores = tl.sum(k * query[None, :], 1) * (DIM**-0.5)
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - new_maximum)
        probability = tl.exp(scores - new_maximum)
        numerator = numerator * alpha + tl.sum(probability[:, None] * v, 0)
        denominator = denominator * alpha + tl.sum(probability, 0)
        maximum = new_maximum
    tl.store(output + head * DIM + columns, numerator / denominator)


def attention(q, keys, values, position, selected, budget=2048, *, dense_limit=0):
    if not 0 <= dense_limit <= min(budget, keys.shape[0]):
        raise ValueError("dense attention limit must fit its budget and KV capacity")
    output = torch.empty_like(q)
    _attention[(q.shape[0],)](
        q,
        keys,
        values,
        position,
        selected,
        output,
        keys.shape[0],
        selected.numel(),
        budget,
        dense_limit,
        num_warps=8,
        debug=bool(dense_limit),
    )
    return output
