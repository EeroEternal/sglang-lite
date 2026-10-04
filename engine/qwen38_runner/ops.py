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


# Adapted from SGLang v0.5.20 fused_recurrent.py (Apache-2.0; derived from vLLM),
# commit 94602c9c2b7cbdb8efd5c52802dac6a1c180089e.
@triton.jit
def _gdn_recurrent_reference(q, k, v, beta, a, bias, log_a, state, output):
    i_v, i_hv = tl.program_id(0), tl.program_id(1)
    o_k = tl.arange(0, 128)
    o_v = i_v * 32 + tl.arange(0, 32)
    head = i_hv // 3
    b_h = tl.load(state + i_hv * 128 * 128 + o_v[:, None] * 128 + o_k[None, :]).to(tl.float32)
    b_q = tl.load(q + head * 128 + o_k).to(tl.float32)
    b_k = tl.load(k + head * 128 + o_k).to(tl.float32)
    b_v = tl.load(v + i_hv * 128 + o_v).to(tl.float32)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * (128**-0.5)

    a_val = tl.load(a + i_hv).to(tl.float32)
    b_val = tl.load(beta + i_hv).to(tl.float32)
    x = a_val + tl.load(bias + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(log_a + i_hv).to(tl.float32)) * softplus_x
    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(output + i_hv * 128 + o_v, b_o)
    tl.store(state + i_hv * 128 * 128 + o_v[:, None] * 128 + o_k[None, :], b_h)


@torch.compile(fullgraph=True)
def _gdn_output_norm(output, z, norm):
    output = output.float()
    output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-6)
    return (output * norm.float() * torch.sigmoid(z.float())).to(z.dtype).reshape(1, -1)


def gdn_update(q, k, v, z, beta, a, bias, log_a, state, norm):
    output = torch.empty_like(v)
    _gdn_recurrent_reference[(4, 6)](
        q, k, v, beta, a, bias, log_a, state, output, num_warps=1, num_stages=3
    )
    return _gdn_output_norm(output, z, norm)


@triton.jit
def _gdn_conv_sum(
    projected,
    history,
    weight,
    output,
    b,
    beta,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_BETA: tl.constexpr,
):
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
    if WITH_BETA and tl.program_id(0) == 0:
        heads = tl.arange(0, 8)
        gate = tl.load(b + heads, mask=heads < 6, other=0).to(tl.float32)
        tl.store(beta + heads, tl.sigmoid(gate), mask=heads < 6)


def gdn_conv_sum(projected, history, weight, b=None):
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
    if b is not None and (
        b.shape != (6,) or b.dtype != torch.bfloat16 or not b.is_cuda or not b.is_contiguous()
    ):
        raise ValueError("GDN gate requires six contiguous CUDA BF16 heads")
    output = torch.empty(channels, device=projected.device, dtype=torch.float32)
    beta = torch.empty_like(b) if b is not None else output
    _gdn_conv_sum[(triton.cdiv(channels, 256),)](
        projected,
        history,
        weight,
        output,
        b if b is not None else projected,
        beta,
        channels,
        256,
        b is not None,
        enable_fp_fusion=False,
    )
    return (output, beta) if b is not None else output


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


@triton.jit
def _rope(x, cos, sin, output, WIDTH: tl.constexpr, HALF: tl.constexpr, TAIL: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, HALF)
    first = tl.load(x + row * WIDTH + columns).to(tl.float32)
    second = tl.load(x + row * WIDTH + HALF + columns).to(tl.float32)
    angle_cos = tl.load(cos + columns)
    angle_sin = tl.load(sin + columns)
    tl.store(output + row * WIDTH + columns, first * angle_cos - second * angle_sin)
    tl.store(
        output + row * WIDTH + HALF + columns,
        second * angle_cos + first * angle_sin,
    )
    tail = tl.arange(0, TAIL)
    rest = tl.load(x + row * WIDTH + 2 * HALF + tail, mask=tail < WIDTH - 2 * HALF, other=0)
    tl.store(
        output + row * WIDTH + 2 * HALF + tail,
        rest,
        mask=tail < WIDTH - 2 * HALF,
    )


def rope(x, cos, sin):
    half = cos.shape[-1]
    if (
        x.ndim != 2
        or x.shape not in ((1, 128), (4, 128), (1, 256), (3, 256), (6, 256), (12, 256))
        or cos.shape != (1, half)
        or sin.shape != cos.shape
        or half != 32
        or x.dtype != torch.bfloat16
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or not all(t.is_cuda and t.is_contiguous() for t in (x, cos, sin))
    ):
        raise ValueError("QSA RoPE requires contiguous BF16 rows and single FP32 position")
    output = torch.empty_like(x)
    _rope[(x.shape[0],)](
        x,
        cos,
        sin,
        output,
        x.shape[1],
        half,
        triton.next_power_of_2(x.shape[1] - 2 * half),
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


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
