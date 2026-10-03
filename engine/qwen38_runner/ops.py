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
    y = y.reshape(shape) * (weight.float() + (1.0 if gemma else 0.0))
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
