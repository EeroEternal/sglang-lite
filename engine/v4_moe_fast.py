"""Faster MoE dispatch for official DeepSeek-V4 Hybrid (decode-friendly).

Official ``MoE.forward`` loops **all local experts** with ``bincount`` +
``torch.where`` (host syncs). For decode (few tokens × top-k), almost all
experts are idle — we only walk **activated** expert ids.

Also fuses act_quant for expert SwiGLU gate/up (same input × two FP4 GEMMs).

FP4 GEMM itself is replaced by DeepGEMM SM120 when ``attach_v4_deep_gemm``
has patched ``kernel.fp4_gemm`` (see ``v4_deep_gemm``). Activated experts in
one layer share one grouped FP4 GEMM per projection (gate, up, down) when
``SGLANG_LITE_V4_MOE_GROUPED`` is on. This module does not import sglang/vllm.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    # Package import (`sglang_lite.v4_moe_fast`).
    from .v4_deep_gemm import (
        deep_gemm_enabled,
        ensure_weight_dg_cache,
        grouped_fp4_gemm_nt,
        grouped_mk_alignment,
        is_armed as deep_gemm_is_armed,
        scale_to_f32,
    )
except ImportError:
    # Bench scripts put `engine/` on sys.path and import this file as a top-level module.
    from v4_deep_gemm import (  # type: ignore
        deep_gemm_enabled,
        ensure_weight_dg_cache,
        grouped_fp4_gemm_nt,
        grouped_mk_alignment,
        is_armed as deep_gemm_is_armed,
        scale_to_f32,
    )

logger = logging.getLogger("sglang_lite.v4_moe_fast")


def moe_fast_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_MOE_FAST", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def moe_grouped_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_MOE_GROUPED", "1").strip().lower()
    return deep_gemm_enabled() and raw not in ("0", "false", "no", "off")


def _is_fp4_weight(w: torch.Tensor) -> bool:
    if w.dtype == torch.float4_e2m1fn_x2:
        return True
    # HF/int8-packed FP4: int8 with companion .scale shaped [N, K//32]
    if w.dtype == torch.int8 and getattr(w, "scale", None) is not None:
        try:
            return int(w.scale.shape[-1]) == (int(w.shape[-1]) * 2) // 32
        except Exception:
            return False
    return False


def _expert_forward_fused(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
    """Expert SwiGLU with single act_quant for FP4 w1/w3."""
    dtype = x.dtype
    w1 = self.w1.weight
    w3 = self.w3.weight
    # Globals live on vendor ``model`` (set in Transformer.__init__), not kernel.
    import model as M  # type: ignore
    import kernel as K  # type: ignore

    block = int(getattr(M, "block_size", 128))
    scale_fmt = getattr(M, "scale_fmt", None)
    scale_dtype = getattr(M, "scale_dtype", torch.float32)
    # FP4 experts: quantize activation once for both gate and up projections.
    # K.fp4_gemm is DeepGEMM when attach_v4_deep_gemm has run.
    if _is_fp4_weight(w1):
        xq, s = K.act_quant(x, block, scale_fmt, scale_dtype)
        gate = K.fp4_gemm(xq, s, w1, w1.scale, scale_dtype).float()
        up = K.fp4_gemm(xq, s, w3, w3.scale, scale_dtype).float()
    elif w1.dtype == torch.float8_e4m3fn:
        xq, s = K.act_quant(x, block, scale_fmt, scale_dtype)
        gate = K.fp8_gemm(xq, s, w1, w1.scale, scale_dtype).float()
        up = K.fp8_gemm(xq, s, w3, w3.scale, scale_dtype).float()
    else:
        gate = self.w1(x).float()
        up = self.w3(x).float()

    lim = float(getattr(self, "swiglu_limit", 0) or 0)
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return self.w2(h.to(dtype))


def _moe_forward_activated_only(self, x: torch.Tensor, input_ids: torch.Tensor):
    """Only run experts that appear in top-k indices (local shard)."""
    import model as M  # type: ignore  # vendor globals: world_size, dist, rank

    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, input_ids.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)

    # Unique activated experts (device → host once). Decode: ≤ topk (e.g. 6).
    uniq = torch.unique(indices)
    # Host list of a few ints — far fewer than n_local_experts (32).
    for i in uniq.tolist():
        ei = int(i)
        if ei < self.experts_start_idx or ei >= self.experts_end_idx:
            continue
        expert = self.experts[ei]
        if expert is None:
            continue
        tok_idx, top = torch.where(indices == ei)
        if tok_idx.numel() == 0:
            continue
        y[tok_idx] += expert(x[tok_idx], weights[tok_idx, top, None])

    if M.world_size > 1:
        M.dist.all_reduce(y)
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


def _local_experts(moe) -> Optional[list]:
    local = []
    for i in range(moe.experts_start_idx, moe.experts_end_idx):
        expert = moe.experts[i]
        if expert is None or not _is_fp4_weight(expert.w1.weight):
            return None
        if not _is_fp4_weight(expert.w2.weight) or not _is_fp4_weight(expert.w3.weight):
            return None
        local.append(expert)
    return local


def grouped_fp4_swiglu(
    xq: torch.Tensor,
    xs: torch.Tensor,
    indices: torch.Tensor,
    route_w: torch.Tensor,
    experts: list,
    *,
    experts_start: int,
    align: int,
    swiglu_limit: float,
    act_quant,
    quant_block: int,
    scale_fmt,
    scale_dtype,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Routed SwiGLU for one MoE layer via three grouped FP4 GEMMs.

    Returns ``[T, dim]`` float32, or ``None`` when a group is larger than
    ``align`` tokens (caller uses the per-expert loop).
    """
    t_count = int(xq.shape[0])
    n_local = len(experts)
    dim_hint = None
    # Decode is a handful of routes. One host read beats a where() per expert.
    if indices.numel() <= 256:
        idx_cpu = indices.detach().to(device="cpu", dtype=torch.int64)
        buckets: dict[int, list[tuple[int, int]]] = {}
        for t in range(t_count):
            for s in range(int(indices.shape[-1])):
                li = int(idx_cpu[t, s]) - experts_start
                if 0 <= li < n_local:
                    buckets.setdefault(li, []).append((t, s))
        groups = []
        for li in sorted(buckets):
            pairs = buckets[li]
            if len(pairs) > align:
                return None
            tok_idx = torch.tensor([p[0] for p in pairs], device=xq.device, dtype=torch.int64)
            top = torch.tensor([p[1] for p in pairs], device=xq.device, dtype=torch.int64)
            groups.append((experts[li], tok_idx, top, len(pairs)))
            dim_hint = experts[li].w2.weight.shape[0]
    else:
        groups = []
        for li, expert in enumerate(experts):
            gid = experts_start + li
            tok_idx, top = torch.where(indices == gid)
            count = int(tok_idx.shape[0])
            if count == 0:
                continue
            if count > align:
                return None
            groups.append((expert, tok_idx, top, count))
            dim_hint = expert.w2.weight.shape[0]
    if not groups:
        if dim_hint is None and experts:
            dim_hint = experts[0].w2.weight.shape[0]
        if dim_hint is None:
            return xq.new_zeros(t_count, 0, dtype=torch.float32)
        return torch.zeros(t_count, int(dim_hint), device=xq.device, dtype=torch.float32)

    n_g = len(groups)
    m_rows = align * n_g
    k = int(xq.shape[-1])
    a = torch.zeros(m_rows, k, device=xq.device, dtype=xq.dtype)
    xs_f = scale_to_f32(xs)
    a_s = torch.ones(m_rows, int(xs_f.shape[-1]), device=xq.device, dtype=torch.float32)
    layout = torch.full((m_rows,), -1, device=xq.device, dtype=torch.int32)
    w1s, s1s, w3s, s3s = [], [], [], []
    for g, (expert, tok_idx, _top, count) in enumerate(groups):
        base = g * align
        a[base : base + count] = xq[tok_idx]
        a_s[base : base + count] = xs_f[tok_idx]
        # Every occupied row, not only the block start. The kernel reads
        # grouped_layout at each BLOCK_M, which can be smaller than `align`.
        layout[base : base + count] = g
        w1, s1 = ensure_weight_dg_cache(expert.w1.weight)
        w3, s3 = ensure_weight_dg_cache(expert.w3.weight)
        w1s.append(w1)
        s1s.append(s1)
        w3s.append(w3)
        s3s.append(s3)

    gate = grouped_fp4_gemm_nt(a, a_s, torch.stack(w1s), torch.stack(s1s), layout)
    up = grouped_fp4_gemm_nt(a, a_s, torch.stack(w3s), torch.stack(s3s), layout)

    hidden_rows = []
    w2s, s2s = [], []
    lim = float(swiglu_limit or 0)
    for g, (expert, tok_idx, top, count) in enumerate(groups):
        base = g * align
        gate_g = gate[base : base + count].float()
        up_g = up[base : base + count].float()
        if lim > 0:
            up_g = torch.clamp(up_g, min=-lim, max=lim)
            gate_g = torch.clamp(gate_g, max=lim)
        hidden = F.silu(gate_g) * up_g
        hidden = hidden * route_w[tok_idx, top].to(dtype=hidden.dtype).unsqueeze(-1)
        hidden_rows.append(hidden.to(dtype=out_dtype))
        w2, s2 = ensure_weight_dg_cache(expert.w2.weight)
        w2s.append(w2)
        s2s.append(s2)

    hidden_cat = torch.cat(hidden_rows, dim=0)
    hq, hs = act_quant(hidden_cat, quant_block, scale_fmt, scale_dtype)
    inter = int(hidden_cat.shape[-1])
    a2 = torch.zeros(m_rows, inter, device=xq.device, dtype=hq.dtype)
    hs_f = scale_to_f32(hs)
    a2_s = torch.ones(m_rows, int(hs_f.shape[-1]), device=xq.device, dtype=torch.float32)
    offset = 0
    for g, (_expert, _tok, _top, count) in enumerate(groups):
        base = g * align
        a2[base : base + count] = hq[offset : offset + count]
        a2_s[base : base + count] = hs_f[offset : offset + count]
        offset += count

    down = grouped_fp4_gemm_nt(a2, a2_s, torch.stack(w2s), torch.stack(s2s), layout)
    y = torch.zeros(t_count, int(down.shape[-1]), device=xq.device, dtype=torch.float32)
    for g, (_expert, tok_idx, _top, count) in enumerate(groups):
        base = g * align
        y[tok_idx] += down[base : base + count].float()
    return y


def _moe_forward_grouped(self, x: torch.Tensor, input_ids: torch.Tensor):
    """Grouped FP4 MoE. Returns None when this batch should use the per-expert loop."""
    import model as M  # type: ignore
    import kernel as K  # type: ignore

    local = _local_experts(self)
    if not local:
        return None
    align = grouped_mk_alignment()
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, input_ids.flatten())
    block = int(getattr(M, "block_size", 128))
    scale_fmt = getattr(M, "scale_fmt", None)
    scale_dtype = getattr(M, "scale_dtype", torch.float32)
    xq, xs = K.act_quant(x, block, scale_fmt, scale_dtype)
    lim = float(getattr(local[0], "swiglu_limit", 0) or 0)
    y = grouped_fp4_swiglu(
        xq,
        xs,
        indices,
        weights,
        local,
        experts_start=int(self.experts_start_idx),
        align=align,
        swiglu_limit=lim,
        act_quant=K.act_quant,
        quant_block=block,
        scale_fmt=scale_fmt,
        scale_dtype=scale_dtype,
        out_dtype=x.dtype,
    )
    if y is None:
        return None
    if M.world_size > 1:
        M.dist.all_reduce(y)
    y = y + self.shared_experts(x).float()
    return y.type_as(x).view(shape)


def _moe_forward_fast(self, x: torch.Tensor, input_ids: torch.Tensor):
    if (
        moe_grouped_enabled()
        and deep_gemm_is_armed()
        and not getattr(self, "_sglang_lite_grouped_off", False)
    ):
        try:
            out = _moe_forward_grouped(self, x, input_ids)
        except Exception as exc:
            logger.warning("v4 grouped MoE disabled after error: %s", exc)
            self._sglang_lite_grouped_off = True
            out = None
        if out is not None:
            return out
    return _moe_forward_activated_only(self, x, input_ids)


def attach_v4_moe_fast(model: Any) -> dict:
    """Patch MoE/Expert methods on a loaded official Transformer.

    Returns stats dict with counts of patched modules.
    """
    if not moe_fast_enabled():
        return {"enabled": False, "moe": 0, "expert": 0}

    import model as M  # type: ignore

    MoE = M.MoE
    Expert = M.Expert

    n_moe = 0
    n_exp = 0
    # Bind optimized methods on classes (all instances).
    if getattr(MoE.forward, "_sglang_lite_moe_fast", False) is not True:
        MoE.forward = _moe_forward_fast
        MoE.forward._sglang_lite_moe_fast = True  # type: ignore[attr-defined]
        n_moe = 1
    if getattr(Expert.forward, "_sglang_lite_moe_fast", False) is not True:
        Expert.forward = _expert_forward_fused
        Expert.forward._sglang_lite_moe_fast = True  # type: ignore[attr-defined]
        n_exp = 1

    # Count modules for logging
    n_moe_mod = sum(1 for m in model.modules() if isinstance(m, MoE))
    n_exp_mod = sum(1 for m in model.modules() if isinstance(m, Expert))
    logger.info(
        "v4 MoE fast: patched MoE.forward + Expert.forward (mods moe=%s expert=%s grouped=%s)",
        n_moe_mod,
        n_exp_mod,
        moe_grouped_enabled(),
    )
    print(
        "[sglang-lite] v4 MoE fast armed (grouped FP4 GEMM + activated-expert fallback); "
        f"moe_modules={n_moe_mod} expert_modules={n_exp_mod} grouped={moe_grouped_enabled()}"
    )
    return {
        "enabled": True,
        "moe_class_patched": n_moe,
        "expert_class_patched": n_exp,
        "moe_modules": n_moe_mod,
        "expert_modules": n_exp_mod,
    }
