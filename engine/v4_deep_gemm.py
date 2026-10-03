"""DeepGEMM SM120 FP8×FP4 GEMM for official DeepSeek-V4 Hybrid.

Vendors ``engine/vendor/deep_gemm_sm120`` (vLLM third_party pin, SM120 kernels).
Does **not** ``import vllm`` / ``import sglang`` at runtime.

Official Hybrid layout (checkpoint / ``Linear``):
  - act: FP8 e4m3, scale 1×128 on K (float32 or float8_e8m0fnu)
  - weight: packed FP4 e2m1 (float4_e2m1fn_x2 or int8), scale 1×32 e8m0 on K

DeepGEMM call shape (verified on PRO6000 sm_120)::

    fp8_fp4_gemm_nt(
        (a_fp8, a_scale_f32),   # [M,K], [M, K//128]
        (w_i8,  w_scale_f32),   # [N, K//2], [N, K//32]
        out_bf16,                # [M,N]
        recipe=(1, 1, 128),
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

Env:
  ``SGLANG_LITE_V4_DEEP_GEMM=1`` (default) enable when import+probe OK
  ``SGLANG_LITE_V4_DEEP_GEMM=0`` force TileLang ``kernel.fp4_gemm``
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger("sglang_lite.v4_deep_gemm")

_DG = None  # module
_DG_OK: Optional[bool] = None
_PATCHED = False

_RECIPE = (1, 1, 128)
_RECIPE_A = (1, 128)
_RECIPE_B = (1, 32)


def deep_gemm_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_DEEP_GEMM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "deep_gemm_sm120"


def _import_deep_gemm_sm120():
    """Import vendored deep_gemm_sm120 (preferred) or site-packages fallback."""
    global _DG, _DG_OK
    if _DG_OK is not None:
        return _DG if _DG_OK else None

    # Prefer in-tree vendor (copied from vLLM third_party, SM120 .so).
    root = _vendor_root()
    if (root / "__init__.py").is_file():
        parent = str(root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            import deep_gemm_sm120 as dg  # type: ignore

            if not hasattr(dg, "fp8_fp4_gemm_nt"):
                raise ImportError("deep_gemm_sm120 missing fp8_fp4_gemm_nt")
            _DG = dg
            _DG_OK = True
            logger.info("deep_gemm_sm120 loaded from vendor %s ver=%s", root, getattr(dg, "__version__", "?"))
            return dg
        except Exception as e:
            logger.warning("vendor deep_gemm_sm120 import failed: %s", e)

    # Optional site package named deep_gemm (must be SM120-capable).
    try:
        import deep_gemm as dg  # type: ignore

        if not hasattr(dg, "fp8_fp4_gemm_nt"):
            raise ImportError("deep_gemm missing fp8_fp4_gemm_nt")
        _DG = dg
        _DG_OK = True
        logger.info("deep_gemm loaded from site-packages ver=%s", getattr(dg, "__version__", "?"))
        return dg
    except Exception as e:
        logger.info("deep_gemm unavailable: %s", e)
        _DG = None
        _DG_OK = False
        return None


def probe_deep_gemm_sm120(device: str = "cuda") -> dict:
    """Tiny FP8×FP4 GEMM probe. Sets readiness for attach."""
    out: dict = {"ok": False, "error": None, "version": None, "source": None}
    if device == "cpu" or not torch.cuda.is_available():
        out["error"] = "no_cuda"
        return out
    dg = _import_deep_gemm_sm120()
    if dg is None:
        out["error"] = "import_failed"
        return out
    out["version"] = getattr(dg, "__version__", "?")
    out["source"] = getattr(dg, "__file__", "?")
    try:
        m, n, k = 16, 64, 128
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        # minimal quant
        a_fp8 = a.to(torch.float8_e4m3fn)
        a_s = torch.ones(m, k // 128, device=device, dtype=torch.float32)
        b = torch.randint(-8, 8, (n, k // 2), device=device, dtype=torch.int8)
        b_s = torch.ones(n, k // 32, device=device, dtype=torch.float32)
        d = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        dg.fp8_fp4_gemm_nt(
            (a_fp8, a_s),
            (b, b_s),
            d,
            recipe=_RECIPE,
            recipe_a=_RECIPE_A,
            recipe_b=_RECIPE_B,
        )
        torch.cuda.synchronize()
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        # mark unusable
        global _DG_OK
        _DG_OK = False
    return out


def scale_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 / float8_e8m0fnu / float → float32 power-of-2 scales for DeepGEMM."""
    if scale.dtype == torch.float32 or scale.dtype == torch.float:
        return scale
    # float8_e8m0fnu and similar: torch .float() is correct on modern PyTorch
    return scale.float()


def weight_as_packed_i8(weight: torch.Tensor) -> torch.Tensor:
    """Official float4_e2m1fn_x2 or int8 packed FP4 → int8 view (no copy)."""
    if weight.dtype == torch.int8:
        return weight
    return weight.view(torch.int8)


def ensure_weight_dg_cache(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache float32 weight scales on the Parameter (one-time)."""
    w_i8 = getattr(weight, "_dg_w_i8", None)
    s_f = getattr(weight, "_dg_s_f32", None)
    if w_i8 is None or s_f is None:
        w_i8 = weight_as_packed_i8(weight.data).contiguous()
        scale = getattr(weight, "scale", None)
        if scale is None:
            raise RuntimeError("FP4 weight missing .scale")
        s_f = scale_to_f32(scale.data).contiguous()
        try:
            weight._dg_w_i8 = w_i8  # type: ignore[attr-defined]
            weight._dg_s_f32 = s_f  # type: ignore[attr-defined]
        except Exception:
            pass
    return w_i8, s_f


def fp4_gemm_nt(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T via DeepGEMM (same contract as TileLang fp4_gemm)."""
    dg = _import_deep_gemm_sm120()
    if dg is None:
        raise RuntimeError("deep_gemm_sm120 not available")

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    odt = out_dtype or torch.get_default_dtype()
    if out is None:
        out = a.new_empty(*a.size()[:-1], N, dtype=odt)
    a2 = a.reshape(M, K)
    if not a2.is_contiguous():
        a2 = a2.contiguous()
    as2 = scale_to_f32(a_s).reshape(M, -1).contiguous()
    # Prefer one-time cache on Parameter (set by prewarm / ensure_weight_dg_cache).
    if getattr(b, "_dg_w_i8", None) is not None and getattr(b, "_dg_s_f32", None) is not None:
        bi = b._dg_w_i8  # type: ignore[attr-defined]
        bs = b._dg_s_f32  # type: ignore[attr-defined]
    elif getattr(b, "scale", None) is not None and b_s is b.scale:
        bi, bs = ensure_weight_dg_cache(b)
    else:
        bi = weight_as_packed_i8(b)
        bs = scale_to_f32(b_s).contiguous()
    d2 = out.reshape(M, N)
    if d2.dtype != odt:
        tmp = a.new_empty(M, N, dtype=odt)
        d2 = tmp
        out = tmp.view(*a.size()[:-1], N) if a.dim() > 2 else tmp
    if not d2.is_contiguous():
        d2 = d2.contiguous()
        out = d2.view(*out.shape) if out.shape == d2.shape else d2
    dg.fp8_fp4_gemm_nt(
        (a2, as2),
        (bi if bi.is_contiguous() else bi.contiguous(), bs if bs.is_contiguous() else bs.contiguous()),
        d2,
        recipe=_RECIPE,
        recipe_a=_RECIPE_A,
        recipe_b=_RECIPE_B,
    )
    return out


def grouped_mk_alignment() -> int:
    """Row alignment of ``m_grouped_fp8_fp4_gemm_nt_contiguous`` (SM120: 128)."""
    dg = _import_deep_gemm_sm120()
    if dg is None or not hasattr(dg, "get_mk_alignment_for_contiguous_layout"):
        raise RuntimeError("deep_gemm_sm120 grouped GEMM unavailable")
    return int(dg.get_mk_alignment_for_contiguous_layout())


def grouped_fp4_gemm_nt(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    grouped_layout: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Grouped ``A[M,K] @ B[G,N,K]^T``.

    ``grouped_layout`` is int32 ``[M]``. Each aligned block of
    ``grouped_mk_alignment()`` rows belongs to one group; the value at the
    first row of the block is the index into ``B``. ``-1`` skips that block.

    Pass ``recipe_a`` / ``recipe_b`` only. This entry rejects a combined
    ``recipe`` together with the split recipes (the dense GEMM accepts both).
    Output is always bfloat16. The grouped kernel rejects fp32.
    """
    dg = _import_deep_gemm_sm120()
    if dg is None or not hasattr(dg, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        raise RuntimeError("deep_gemm_sm120 grouped GEMM unavailable")

    # Grouped SM120 entry asserts D is bf16 (dense fp8_fp4_gemm_nt also
    # accepts fp32). The V4 loader sets the default dtype to bf16 before
    # forward, which is the contract this matches.
    odt = torch.bfloat16
    if out_dtype is not None and out_dtype != torch.bfloat16:
        raise RuntimeError("grouped FP4 GEMM output must be bfloat16")
    m, k = a.shape
    _g, n, _kp = b.shape
    if out is None:
        out = torch.empty(m, n, device=a.device, dtype=odt)
    a2 = a if a.is_contiguous() else a.contiguous()
    as2 = scale_to_f32(a_s)
    as2 = as2 if as2.is_contiguous() else as2.contiguous()
    b2 = b if b.is_contiguous() else b.contiguous()
    bs2 = scale_to_f32(b_s)
    bs2 = bs2 if bs2.is_contiguous() else bs2.contiguous()
    d2 = out if out.is_contiguous() and out.dtype == odt else torch.empty(m, n, device=a.device, dtype=odt)
    layout = grouped_layout
    if layout.dtype != torch.int32 or not layout.is_contiguous():
        layout = layout.to(dtype=torch.int32).contiguous()
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a2, as2),
        (b2, bs2),
        d2,
        layout,
        recipe_a=_RECIPE_A,
        recipe_b=_RECIPE_B,
    )
    if d2 is not out:
        out.copy_(d2)
    return out


def _fp4_gemm_dropin(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Drop-in replacement for vendor ``kernel.fp4_gemm`` (ignores scale_dtype; uses f32 for DG)."""
    del scale_dtype  # DeepGEMM consumes float32 scales
    return fp4_gemm_nt(a, a_s, b, b_s)


def _linear_deep_gemm(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Replace vendor ``model.linear`` for FP4/FP8 when DeepGEMM is live."""
    assert bias is None
    import kernel as K  # type: ignore
    import model as M  # type: ignore

    if weight.dtype in (torch.float4_e2m1fn_x2, torch.int8) and getattr(weight, "scale", None) is not None:
        # Heuristic: int8 with .scale and last dim = in//2 is FP4 packed.
        is_fp4 = weight.dtype == torch.float4_e2m1fn_x2 or (
            weight.dtype == torch.int8
            and weight.scale is not None
            and weight.scale.shape[-1] == (weight.shape[-1] * 2) // 32
        )
        if is_fp4 or weight.dtype == torch.float4_e2m1fn_x2:
            block = int(getattr(M, "block_size", 128))
            scale_fmt = getattr(M, "scale_fmt", None)
            scale_dtype = getattr(M, "scale_dtype", torch.float32)
            xq, s = K.act_quant(x, block, scale_fmt, scale_dtype)
            w_i8, w_s = ensure_weight_dg_cache(weight)
            return fp4_gemm_nt(xq, s, w_i8, w_s)

    if weight.dtype == torch.float8_e4m3fn:
        # Keep TileLang/cuBLAS path for pure FP8 weights for now.
        block = int(getattr(M, "block_size", 128))
        scale_fmt = getattr(M, "scale_fmt", None)
        scale_dtype = getattr(M, "scale_dtype", torch.float32)
        xq, s = K.act_quant(x, block, scale_fmt, scale_dtype)
        return K.fp8_gemm(xq, s, weight, weight.scale, scale_dtype)

    import torch.nn.functional as F

    return F.linear(x, weight)


def prewarm_model_scales(model: Any) -> int:
    """Materialize float32 scale caches for all FP4 parameters."""
    n = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if w is None or not hasattr(w, "scale"):
            continue
        if w.dtype not in (torch.float4_e2m1fn_x2, torch.int8):
            continue
        try:
            ensure_weight_dg_cache(w)
            n += 1
        except Exception:
            pass
    return n


def attach_v4_deep_gemm(model: Optional[Any] = None) -> dict:
    """Patch vendor ``kernel.fp4_gemm`` (+ optional ``model.linear``) to DeepGEMM.

    Safe no-op when disabled / probe fails / already patched.
    """
    global _PATCHED
    stats: dict = {
        "enabled": False,
        "patched": False,
        "probe": None,
        "scale_cache": 0,
        "backend": "tilelang",
    }
    if not deep_gemm_enabled():
        stats["backend"] = "disabled"
        return stats

    probe = probe_deep_gemm_sm120()
    stats["probe"] = probe
    if not probe.get("ok"):
        logger.warning("DeepGEMM SM120 probe failed: %s", probe.get("error"))
        print(f"[sglang-lite] v4 DeepGEMM skipped (probe failed: {probe.get('error')})")
        return stats

    if not _PATCHED:
        try:
            import kernel as K  # type: ignore
        except ImportError as e:
            # Hybrid load puts vendor/deepseek_infer on sys.path first; call attach after that.
            stats["error"] = f"kernel_not_importable: {e}"
            print(
                "[sglang-lite] v4 DeepGEMM probe OK but kernel not on path yet; "
                "call attach after Hybrid graph import"
            )
            return stats

        if not getattr(K.fp4_gemm, "_sglang_lite_deep_gemm", False):
            K._fp4_gemm_tilelang = K.fp4_gemm  # type: ignore[attr-defined]
            K.fp4_gemm = _fp4_gemm_dropin  # type: ignore[assignment]
            K.fp4_gemm._sglang_lite_deep_gemm = True  # type: ignore[attr-defined]
            stats["patched"] = True
        # Mark model.linear once present (calls K.fp4_gemm → DeepGEMM after patch).
        try:
            import model as M  # type: ignore

            if not getattr(M.linear, "_sglang_lite_deep_gemm", False):
                M.linear._sglang_lite_deep_gemm = True  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug("model.linear mark skipped: %s", e)
        _PATCHED = True
        stats["patched"] = True

    if model is not None:
        stats["scale_cache"] = prewarm_model_scales(model)

    stats["enabled"] = True
    stats["backend"] = "deep_gemm_sm120"
    ver = (probe or {}).get("version", "?")
    print(
        f"[sglang-lite] v4 DeepGEMM armed (fp8_fp4 recipe a1x128/b1x32); "
        f"ver={ver} scale_cache={stats['scale_cache']}"
    )
    logger.info("DeepGEMM attached: %s", stats)
    return stats


def is_armed() -> bool:
    return bool(_PATCHED and _DG_OK)


__all__ = [
    "attach_v4_deep_gemm",
    "deep_gemm_enabled",
    "fp4_gemm_nt",
    "grouped_fp4_gemm_nt",
    "grouped_mk_alignment",
    "is_armed",
    "probe_deep_gemm_sm120",
    "prewarm_model_scales",
    "scale_to_f32",
]
