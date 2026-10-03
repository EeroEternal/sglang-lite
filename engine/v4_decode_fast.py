"""Graph-safe DeepSeek-V4 decode for one CUDA graph across token positions.

Official ``Transformer.forward`` takes ``start_pos`` as a Python int and branches
on it (RoPE slice, KV slot, compressor, indexer top-k width). A CUDA graph
records one kernel sequence, so the fast path:

* reads position from a 0-dim CUDA long (``model._decode_pos``)
* keeps every tensor shape fixed (window 128, compress pad, indexer top-k 512)
* runs the compressor math on every token and commits stores with ``where``
* gathers the 6 activated experts on device into one reused workspace

Prefill (seqlen != 1) stays on the original modules. The fast path is off until
``set_fast(True)``, which is the capture / replay window.

Does not import sglang or vllm.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from .v4_deep_gemm import grouped_fp4_gemm_nt, grouped_mk_alignment, scale_to_f32, weight_as_packed_i8
    from .v4_moe_fast import _local_experts
except ImportError:
    from v4_deep_gemm import (  # type: ignore
        grouped_fp4_gemm_nt,
        grouped_mk_alignment,
        scale_to_f32,
        weight_as_packed_i8,
    )
    from v4_moe_fast import _local_experts  # type: ignore

_FAST = {"on": False}
_POS: dict[str, Optional[torch.Tensor]] = {"t": None}
_ORIG: dict[str, Any] = {}
_READY = {"moe": False}


def set_fast(enabled: bool) -> None:
    """Turn the fixed-shape decode path on or off. Capture and replay need it on."""
    _FAST["on"] = bool(enabled)


def fast_is_on() -> bool:
    return bool(_FAST["on"])


def window_indices(pos: torch.Tensor, win: int, idx: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Decode sliding-window indices, length ``win``.

    ``pos >= win - 1`` is the ring buffer order used by the official cat.
    Earlier positions are ``0..pos`` padded with -1. ``pos`` is a 0-dim tensor
    so this launches the same kernels at every position.
    """
    sp = torch.remainder(pos, win)
    past = pos >= (win - 1)
    ring_cut = (win - 1) - sp
    ring = torch.where(idx < ring_cut, sp + 1 + idx, idx - ring_cut)
    partial = torch.where(idx <= pos, idx, neg)
    return torch.where(past, ring, partial)


def compress_indices(pos: torch.Tensor, ratio: int, offset: int, idx: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Fixed-length compressed-KV indices. Invalid slots are -1."""
    n_valid = torch.div(pos + 1, ratio, rounding_mode="floor")
    return torch.where(idx < n_valid, idx + offset, neg)


def check_index_formulas() -> None:
    """CPU checks of the window and compress index rules."""
    win = 4
    idx = torch.arange(win)
    neg = torch.full((win,), -1)

    def at(pos: int) -> list[int]:
        return window_indices(torch.tensor(pos), win, idx, neg).tolist()

    assert at(5) == [2, 3, 0, 1], at(5)
    assert at(2) == [0, 1, 2, -1], at(2)
    assert at(4) == [1, 2, 3, 0], at(4)
    assert at(3) == [0, 1, 2, 3], at(3)
    assert at(7) == [0, 1, 2, 3], at(7)

    cidx = torch.arange(4)
    cneg = torch.full((4,), -1)
    got = compress_indices(torch.tensor(5), 4, 128, cidx, cneg).tolist()
    assert got == [128, -1, -1, -1], got
    got = compress_indices(torch.tensor(7), 4, 128, cidx, cneg).tolist()
    assert got == [128, 129, -1, -1], got


def _pos() -> torch.Tensor:
    t = _POS["t"]
    if t is None:
        raise RuntimeError("v4 decode pos tensor is not set")
    return t


def _use_fast_seq(x: torch.Tensor) -> bool:
    return bool(_FAST["on"]) and x.shape[0] == 1 and x.shape[1] == 1


class _MoeWS:
    """One reused grouped-GEMM workspace for every MoE layer (T=1, K=6)."""

    def __init__(self) -> None:
        self.ready = False
        self.groups = 0
        self.align = 0
        self.m = 0
        self.slots: Optional[torch.Tensor] = None
        self.layout: Optional[torch.Tensor] = None
        self.a: Optional[torch.Tensor] = None
        self.a_s: Optional[torch.Tensor] = None
        self.gate: Optional[torch.Tensor] = None
        self.up: Optional[torch.Tensor] = None
        self.down: Optional[torch.Tensor] = None
        self.hidden: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.w1: Optional[torch.Tensor] = None
        self.s1: Optional[torch.Tensor] = None
        self.w3: Optional[torch.Tensor] = None
        self.s3: Optional[torch.Tensor] = None
        self.w2: Optional[torch.Tensor] = None
        self.s2: Optional[torch.Tensor] = None


_WS = _MoeWS()


def _blob(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = weight_as_packed_i8(weight.data)
    if not packed.is_contiguous():
        packed = packed.contiguous()
    scale = weight.scale.data
    if not scale.is_contiguous():
        scale = scale.contiguous()
    return packed, scale


def _ptr_table(blobs: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in blobs], dtype=torch.int64, device=blobs[0].device)


def _alloc_stack(sample: torch.Tensor, groups: int) -> torch.Tensor:
    return torch.empty((groups, *sample.shape), dtype=sample.dtype, device=sample.device)


def prepare_decode(model: Any) -> dict:
    """Bind KV views, build expert pointer tables, allocate the shared workspace.

    Call after weights are loaded and before warmup. Pointer tables alias the
    live expert parameters. The workspace holds 6 experts, not all 32.
    """
    import model as M

    device = next(model.parameters()).device
    pos = torch.zeros((), dtype=torch.long, device=device)
    model._decode_pos = pos
    _POS["t"] = pos

    n_attn = 0
    for mod in model.modules():
        if not isinstance(mod, M.Attention):
            continue
        n_attn += 1
        win = int(mod.window_size)
        mod._win_idx = torch.arange(win, device=device, dtype=torch.long)
        mod._win_neg = torch.full((win,), -1, device=device, dtype=torch.long)
        ratio = int(mod.compress_ratio or 0)
        if ratio:
            mod.compressor.kv_cache = mod.kv_cache[:, win:]
            mod.compressor.freqs_cis = mod.freqs_cis
            if mod.compressor.overlap:
                b, _n, h = mod.compressor.kv_state.shape
                mod.compressor._rot_kv = torch.empty(
                    b, ratio, h, dtype=mod.compressor.kv_state.dtype, device=device
                )
                mod.compressor._rot_sc = torch.empty_like(mod.compressor._rot_kv)
            if mod.indexer is not None:
                mod.indexer.freqs_cis = mod.freqs_cis
                mod.indexer.compressor.kv_cache = mod.indexer.kv_cache
                mod.indexer.compressor.freqs_cis = mod.freqs_cis
                tlen = int(mod.indexer.kv_cache.shape[1])
                mod.indexer._t_idx = torch.arange(tlen, device=device, dtype=torch.long)
                ic = mod.indexer.compressor
                if ic.overlap and not hasattr(ic, "_rot_kv"):
                    b, _n, h = ic.kv_state.shape
                    ic._rot_kv = torch.empty(b, int(ic.compress_ratio), h, dtype=ic.kv_state.dtype, device=device)
                    ic._rot_sc = torch.empty_like(ic._rot_kv)
            else:
                n_comp = int(mod.kv_cache.shape[1] - win)
                mod._cmp_idx = torch.arange(n_comp, device=device, dtype=torch.long)
                mod._cmp_neg = torch.full((n_comp,), -1, device=device, dtype=torch.long)

    moes = [m for m in model.modules() if isinstance(m, M.MoE)]
    if not moes:
        raise RuntimeError("no MoE modules")
    groups = int(moes[0].n_activated_experts)
    align = int(grouped_mk_alignment())
    local0 = _local_experts(moes[0])
    if not local0:
        raise RuntimeError("MoE layer 0 has no local FP4 experts")
    w1, s1 = _blob(local0[0].w1.weight)
    _w3, s3 = _blob(local0[0].w3.weight)
    w2, s2 = _blob(local0[0].w2.weight)
    inter = int(w1.shape[0])
    hidden = int(w2.shape[0])
    _WS.groups = groups
    _WS.align = align
    _WS.m = align * groups
    _WS.slots = torch.arange(groups, device=device, dtype=torch.long) * align
    _WS.layout = torch.full((_WS.m,), -1, dtype=torch.int32, device=device)
    for g in range(groups):
        _WS.layout[g * align] = g
    _WS.w1 = _alloc_stack(w1, groups)
    _WS.s1 = _alloc_stack(s1, groups)
    _WS.w3 = _alloc_stack(_w3, groups)
    _WS.s3 = _alloc_stack(s3, groups)
    _WS.w2 = _alloc_stack(w2, groups)
    _WS.s2 = _alloc_stack(s2, groups)
    k_act = int(w1.shape[-1]) * 2
    _WS.a = torch.zeros(_WS.m, k_act, dtype=torch.float8_e4m3fn, device=device)
    _WS.a_s = torch.ones(_WS.m, k_act // int(M.block_size), dtype=torch.float32, device=device)
    _WS.gate = torch.empty(_WS.m, inter, dtype=torch.bfloat16, device=device)
    _WS.up = torch.empty(_WS.m, inter, dtype=torch.bfloat16, device=device)
    _WS.down = torch.empty(_WS.m, hidden, dtype=torch.bfloat16, device=device)
    _WS.hidden = torch.zeros(_WS.m, inter, dtype=torch.bfloat16, device=device)
    _WS.y = torch.zeros(1, hidden, dtype=torch.float32, device=device)

    kept = 0
    for moe in moes:
        if int(moe.n_activated_experts) != groups:
            raise RuntimeError("MoE top-k differs across layers")
        local = _local_experts(moe)
        if not local:
            raise RuntimeError(f"MoE layer {moe.layer_id} is not FP4")
        packs = {name: [] for name in ("w1", "s1", "w3", "s3", "w2", "s2")}
        anchors: list[torch.Tensor] = []
        for expert in local:
            for name, weight in (
                ("w1", expert.w1.weight),
                ("w3", expert.w3.weight),
                ("w2", expert.w2.weight),
            ):
                packed, scale = _blob(weight)
                anchors.append(packed)
                anchors.append(scale)
                packs[name].append(packed)
                packs["s" + name[-1]].append(scale)
        moe._g_keep = anchors
        moe._g_w1 = _ptr_table(packs["w1"])
        moe._g_s1 = _ptr_table(packs["s1"])
        moe._g_w3 = _ptr_table(packs["w3"])
        moe._g_s3 = _ptr_table(packs["s3"])
        moe._g_w2 = _ptr_table(packs["w2"])
        moe._g_s2 = _ptr_table(packs["s2"])
        moe._g_nlocal = len(local)
        kept += 1

    # Compile the gather kernel before capture. A mismatch here is a hard failure.
    from v4_fast_ops import gather_bytes

    src = torch.arange(32, device=device, dtype=torch.uint8)
    dst = torch.empty(1, 32, dtype=torch.uint8, device=device)
    gather_bytes(
        torch.tensor([src.data_ptr()], dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        dst,
    )
    torch.cuda.synchronize()
    if int((dst - src).abs().sum().item()) != 0:
        raise RuntimeError("gather_bytes self-check failed")

    head = model.head
    if M.world_size > 1:
        head._ag_buf = torch.empty(M.world_size, head.part_vocab_size, dtype=torch.float32, device=device)

    _WS.ready = True
    _READY["moe"] = True
    return {
        "attn": n_attn,
        "moe": kept,
        "groups": groups,
        "align": align,
        "workspace_mib": round(sum(t.numel() * t.element_size() for t in (
            _WS.w1, _WS.s1, _WS.w3, _WS.s3, _WS.w2, _WS.s2, _WS.a, _WS.a_s,
            _WS.gate, _WS.up, _WS.down, _WS.hidden, _WS.y,
        ) if t is not None) / (1024 * 1024), 1),
    }


def snapshot_buffers(model: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Clone every buffer. Restore with ``copy_`` so KV views keep their storage."""
    return [(buf, buf.detach().clone()) for _name, buf in model.named_buffers()]


def restore_buffers(snaps: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for buf, saved in snaps:
        buf.copy_(saved)


def _compressor_decode(comp: Any, x: torch.Tensor, pos: torch.Tensor) -> None:
    """Decode compressor. Kernel sequence does not depend on the position."""
    import model as M

    bsz = x.shape[0]
    ratio = int(comp.compress_ratio)
    overlap = bool(comp.overlap)
    d = int(comp.head_dim)
    rd = int(comp.rope_head_dim)
    dtype = x.dtype
    xf = x.float()
    kv = comp.wkv(xf)
    score = comp.wgate(xf)
    slot = torch.remainder(pos, ratio)
    ape = comp.ape.index_select(0, slot.reshape(1)).view(1, 1, -1)
    score = score + ape
    dest = (slot + (ratio if overlap else 0)).reshape(1)
    state = comp.kv_state[:bsz]
    score_state = comp.score_state[:bsz]
    state.index_copy_(1, dest, kv)
    score_state.index_copy_(1, dest, score)

    if overlap:
        kv_cat = torch.cat([state[:, :ratio, :d], state[:, ratio:, d:]], dim=1)
        sc_cat = torch.cat([score_state[:, :ratio, :d], score_state[:, ratio:, d:]], dim=1)
        pooled = (kv_cat * torch.softmax(sc_cat, dim=1)).sum(dim=1, keepdim=True)
        pred = torch.remainder(pos + 1, ratio) == 0
        rot_kv = comp._rot_kv[:bsz]
        rot_sc = comp._rot_sc[:bsz]
        rot_kv.copy_(state[:, ratio:])
        rot_sc.copy_(score_state[:, ratio:])
        state[:, :ratio] = torch.where(pred, rot_kv, state[:, :ratio])
        score_state[:, :ratio] = torch.where(pred, rot_sc, score_state[:, :ratio])
    else:
        pooled = (state * torch.softmax(score_state, dim=1)).sum(dim=1, keepdim=True)
        pred = torch.remainder(pos + 1, ratio) == 0

    cand = comp.norm(pooled.to(dtype))
    freq_i = torch.clamp(pos + 1 - ratio, min=0).reshape(1)
    freqs = comp.freqs_cis.index_select(0, freq_i)
    M.apply_rotary_emb(cand[..., -rd:], freqs)
    if comp.rotate:
        cand = M.rotate_activation(cand)
        M.fp4_act_quant(cand, M.fp4_block_size, True)
    else:
        M.act_quant(cand[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
    cache = comp.kv_cache[:bsz]
    comp_i = torch.div(pos, ratio, rounding_mode="floor").reshape(1)
    old = cache.index_select(1, comp_i)
    cache.index_copy_(1, comp_i, torch.where(pred, cand, old))


def _indexer_decode(indexer: Any, x: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, offset: int) -> torch.Tensor:
    import model as M

    bsz = x.shape[0]
    rd = int(indexer.rope_head_dim)
    ratio = int(indexer.compress_ratio)
    freqs = indexer.freqs_cis.index_select(0, pos.reshape(1))
    q = indexer.wq_b(qr)
    q = q.unflatten(-1, (indexer.n_local_heads, indexer.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs)
    q = M.rotate_activation(q)
    M.fp4_act_quant(q, M.fp4_block_size, True)
    _compressor_decode(indexer.compressor, x, pos)
    weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)
    score = torch.einsum("bshd,btd->bsht", q, indexer.kv_cache[:bsz])
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if M.world_size > 1:
        M.dist.all_reduce(score)
    n_valid = torch.div(pos + 1, ratio, rounding_mode="floor")
    score = score.masked_fill(indexer._t_idx.view(1, 1, -1) >= n_valid, float("-inf"))
    topk = score.topk(int(indexer.index_topk), dim=-1)[1]
    topk = torch.where(topk >= n_valid, torch.full_like(topk, -1), topk + offset)
    return topk.to(torch.int32)


def _attn_decode(attn: Any, x: torch.Tensor) -> torch.Tensor:
    import model as M
    import kernel as K

    bsz, seqlen, _ = x.shape
    pos = _pos()
    win = int(attn.window_size)
    rd = int(attn.rope_head_dim)
    freqs = attn.freqs_cis.index_select(0, pos.reshape(1))
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    M.apply_rotary_emb(q[..., -rd:], freqs)

    kv = attn.kv_norm(attn.wkv(x))
    M.apply_rotary_emb(kv[..., -rd:], freqs)
    M.act_quant(kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)

    window = window_indices(pos, win, attn._win_idx, attn._win_neg).to(torch.int32).view(1, 1, win)
    ratio = int(attn.compress_ratio or 0)
    if ratio:
        if attn.indexer is not None:
            compress = _indexer_decode(attn.indexer, x, qr, pos, win)
        else:
            compress = compress_indices(pos, ratio, win, attn._cmp_idx, attn._cmp_neg).to(torch.int32).view(1, 1, -1)
        topk = torch.cat([window, compress], dim=-1).contiguous()
    else:
        topk = window.contiguous()

    slot = torch.remainder(pos, win).reshape(1)
    attn.kv_cache[:bsz].index_copy_(1, slot, kv)
    if ratio:
        _compressor_decode(attn.compressor, x, pos)
    o = K.sparse_attn(q, attn.kv_cache[:bsz], attn.attn_sink, topk, attn.softmax_scale)
    M.apply_rotary_emb(o[..., -rd:], freqs, True)
    o = o.view(bsz, seqlen, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return attn.wo_b(o.flatten(2))


def _embed_decode(emb: Any, x: torch.Tensor) -> torch.Tensor:
    import model as M

    if M.world_size > 1:
        mask = (x < emb.vocab_start_idx) | (x >= emb.vocab_end_idx)
        shifted = torch.where(mask, torch.zeros_like(x), x - emb.vocab_start_idx)
        y = F.embedding(shifted, emb.weight)
        y = torch.where(mask.unsqueeze(-1), torch.zeros_like(y), y)
        M.dist.all_reduce(y)
        return y
    return F.embedding(x, emb.weight)


def _head_decode(head: Any, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: Any) -> torch.Tensor:
    import model as M

    hidden = head.hc_head(x, hc_fn, hc_scale, hc_base)
    logits = head.get_logits(norm(hidden))
    if M.world_size <= 1:
        return logits
    buf = head._ag_buf
    M.dist.all_gather_into_tensor(buf, logits)
    world = int(M.world_size)
    return buf.view(world, logits.shape[0], -1).permute(1, 0, 2).reshape(logits.shape[0], -1).contiguous()


def _gather_group(ptrs: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor) -> None:
    from v4_fast_ops import gather_bytes

    gather_bytes(ptrs, idx, dst)


def _moe_decode(moe: Any, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    import model as M
    import kernel as K

    shape = x.shape
    flat = x.reshape(-1, moe.dim)
    weights, indices = moe.gate(flat, input_ids.flatten())
    ids = indices.reshape(-1).to(torch.int64)
    local = ids - int(moe.experts_start_idx)
    n_local = int(moe._g_nlocal)
    valid = (local >= 0) & (local < n_local)
    safe = torch.where(valid, local, torch.zeros_like(local))
    rw = torch.where(valid, weights.reshape(-1), torch.zeros_like(weights.reshape(-1)))

    block = int(M.block_size)
    xq, xs = K.act_quant(flat, block, M.scale_fmt, M.scale_dtype)
    xs_f = scale_to_f32(xs)
    assert _WS.a is not None and _WS.slots is not None
    # index_copy has no Float8 kernel. The bit pattern is one byte per value.
    src_u8 = xq.reshape(-1).view(torch.uint8).expand(_WS.groups, -1).contiguous()
    _WS.a.view(torch.uint8).index_copy_(0, _WS.slots, src_u8)
    _WS.a_s.index_copy_(0, _WS.slots, xs_f.expand(_WS.groups, -1).contiguous())

    _gather_group(moe._g_w1, safe, _WS.w1)
    _gather_group(moe._g_s1, safe, _WS.s1)
    _gather_group(moe._g_w3, safe, _WS.w3)
    _gather_group(moe._g_s3, safe, _WS.s3)
    gate = grouped_fp4_gemm_nt(_WS.a, _WS.a_s, _WS.w1, scale_to_f32(_WS.s1), _WS.layout, out=_WS.gate)
    up = grouped_fp4_gemm_nt(_WS.a, _WS.a_s, _WS.w3, scale_to_f32(_WS.s3), _WS.layout, out=_WS.up)

    lim = float(getattr(moe.experts[moe.experts_start_idx], "swiglu_limit", 0) or 0)
    _WS.hidden.zero_()
    align = _WS.align
    for g in range(_WS.groups):
        row = g * align
        gate_g = gate[row].float()
        up_g = up[row].float()
        if lim > 0:
            up_g = up_g.clamp(min=-lim, max=lim)
            gate_g = gate_g.clamp(max=lim)
        hidden = F.silu(gate_g) * up_g
        _WS.hidden[row].copy_((hidden * rw[g]).to(dtype=_WS.hidden.dtype))

    hq, hs = K.act_quant(_WS.hidden, block, M.scale_fmt, M.scale_dtype)
    _gather_group(moe._g_w2, safe, _WS.w2)
    _gather_group(moe._g_s2, safe, _WS.s2)
    down = grouped_fp4_gemm_nt(hq, scale_to_f32(hs), _WS.w2, scale_to_f32(_WS.s2), _WS.layout, out=_WS.down)
    _WS.y.zero_()
    for g in range(_WS.groups):
        _WS.y[0].add_(down[g * align].float())
    y = _WS.y
    if M.world_size > 1:
        M.dist.all_reduce(y)
    y = y + moe.shared_experts(flat).float()
    return y.type_as(flat).view(shape)


def _wrap_attn(self, x, start_pos):
    if _use_fast_seq(x):
        return _attn_decode(self, x)
    return _ORIG["attn"](self, x, start_pos)


def _wrap_moe(self, x, input_ids):
    flat_tokens = x.reshape(-1, self.dim).shape[0]
    if _FAST["on"] and _READY["moe"] and flat_tokens == 1 and x.shape[0] == 1:
        return _moe_decode(self, x, input_ids)
    return _ORIG["moe"](self, x, input_ids)


def _wrap_embed(self, x):
    if _FAST["on"] and x.shape[0] == 1:
        return _embed_decode(self, x)
    return _ORIG["embed"](self, x)


def _wrap_head(self, x, hc_fn, hc_scale, hc_base, norm):
    if _FAST["on"] and x.shape[0] == 1 and getattr(self, "_ag_buf", None) is not None:
        return _head_decode(self, x, hc_fn, hc_scale, hc_base, norm)
    return _ORIG["head"](self, x, hc_fn, hc_scale, hc_base, norm)


def attach_v4_decode_fast(model: Any) -> dict:
    """Wrap Attention, MoE, embedding, and head. Fast path stays off.

    ``attach_v4_moe_fast`` must already have replaced ``MoE.forward``. The
    wrapper calls that implementation whenever the fast path is off.
    """
    import model as M

    if getattr(M.Attention.forward, "_v4_decode_fast", False):
        return {"attached": True, "already": True}
    _ORIG["attn"] = M.Attention.forward
    _ORIG["moe"] = M.MoE.forward
    _ORIG["embed"] = M.ParallelEmbedding.forward
    _ORIG["head"] = M.ParallelHead.forward
    for fn in (_wrap_attn, _wrap_moe, _wrap_embed, _wrap_head):
        fn._v4_decode_fast = True  # type: ignore[attr-defined]
    M.Attention.forward = _wrap_attn
    M.MoE.forward = _wrap_moe
    M.ParallelEmbedding.forward = _wrap_embed
    M.ParallelHead.forward = _wrap_head
    del model
    return {"attached": True, "already": False}


def capture_decode(
    model: Any, static_tokens: torch.Tensor, *, autoregressive: bool = False
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Warm up the fast path and capture one decode step.

    ``set_fast(True)`` is required. ``static_tokens`` is ``[1, 1]`` long CUDA.
    Position is ``model._decode_pos``. Both are read at replay time; fill them
    before each ``graph.replay()`` unless ``autoregressive`` is set. In that
    mode the graph writes its next token and advances position on the GPU, so
    consecutive replays do not synchronize with the host. Warmup and capture
    mutate KV. The caller restores the post-prefill snapshot afterwards.
    """
    if not _FAST["on"]:
        raise RuntimeError("set_fast(True) before capture")
    if static_tokens.shape != (1, 1):
        raise RuntimeError(f"static tokens must be [1, 1], got {tuple(static_tokens.shape)}")
    # Warm up on the capture stream. A side-stream NCCL warmup leaves the
    # communicator on the wrong stream and capture then fails.
    torch.cuda.synchronize()
    for _ in range(3):
        model(static_tokens, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(static_tokens, 1)
        if autoregressive:
            static_tokens.copy_(logits[0].argmax().reshape(1, 1))
            model._decode_pos.add_(1)
    model._v4_graph = graph
    model._v4_static_logits = logits
    model._v4_static_tokens = static_tokens
    return graph, logits


def graph_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_CUDA_GRAPH", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
