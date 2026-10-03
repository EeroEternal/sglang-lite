#!/usr/bin/env python3
"""Short thruput probe for Phase-A MoE performance baselines (no multi-node).

Measures wall-clock **after model load**:
  - prefill + decode cold (first request)
  - decode-ish warm (second request, optional prefix reuse)
  - tok/s = completion_tokens / generate_s

  python scripts/moe_thruput_probe.py \\
    --model ~/models/Qwen3-30B-A3B-Instruct --device cuda \\
    --max-new 128 --cases 1x128,1x64 --out ~/bench/thru_qwen3_30b.json

Env defaults: SGLANG_LITE_V4_DISABLE_FI_SPARSE=1 (harmless for non-V4).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _parse_cases(s: str) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"bad case {part!r}, want batchxmax_new e.g. 1x128")
        b, n = part.lower().split("x", 1)
        out.append({"batch": int(b), "max_new": int(n)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument(
        "--cases",
        default="1x64,1x128",
        help="comma list of batchxmax_new (batch>1 runs sequential for now)",
    )
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--warm-runs", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.warm_runs < 1:
        ap.error("--warm-runs must be positive")

    _ensure_path()
    os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
    os.environ.setdefault("SGLANG_LITE_LOG_JSON", "0")
    # Default thruput path: HF cache + batched_mm + torch.compile (~84 tok/s).
    # Radix-native stack (PRO6000 Qwen3-30B ~101 warm): paged + native layer loop
    # + CUDA graph + cutlass fused MoE.
    if os.environ.get("SGLANG_LITE_RADIX_NATIVE", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("SGLANG_LITE_FORCE_HF_CACHE", "0")
        os.environ.setdefault("SGLANG_LITE_TORCH_COMPILE", "0")
        os.environ.setdefault("SGLANG_LITE_CUDA_GRAPH_DECODE", "1")
        os.environ.setdefault("SGLANG_LITE_FUSED_MOE", "1")
        os.environ.setdefault("SGLANG_LITE_NATIVE_DECODE", "1")
    else:
        os.environ.setdefault("SGLANG_LITE_FORCE_HF_CACHE", "1")
        os.environ.setdefault("SGLANG_LITE_TORCH_COMPILE", "1")
        # cutlass fused slower than compile on FORCE_HF path (~44 vs ~84).
        os.environ.setdefault("SGLANG_LITE_FUSED_MOE", "0")
    # Large burst: fewer scheduler hops; runner warms compile at load.
    os.environ.setdefault("SGLANG_LITE_DECODE_BURST", "128")
    # batched_mm: ~2× default grouped_mm; + torch.compile(mode=default) ~84 tok/s.
    os.environ.setdefault("SGLANG_LITE_EXPERTS_IMPL", "batched_mm")

    from sglang_lite import LiteEngine

    cases = _parse_cases(args.cases)
    t_load0 = time.perf_counter()
    # start_loop=False → pump_until_idle uses pump_once (decode burst enabled).
    eng = LiteEngine(
        model_name=args.model,
        device=args.device,
        max_batch_size=max(args.max_batch, max(c["batch"] for c in cases)),
        allow_stub=False,
        start_loop=False,
    )
    # Thruput path: skip per-token detokenize + ignore EOS for fixed-length runs.
    eng._gen_skip_streaming_text = True
    eng._gen_ignore_eos = True
    load_s = round(time.perf_counter() - t_load0, 3)

    results: List[Dict[str, Any]] = []
    try:
        ids = eng.tokenize(args.prompt)
        for case in cases:
            max_new = case["max_new"]
            batch = case["batch"]
            # Sequential batch for baseline (true concurrent batch is a later gate).
            cold_s = 0.0
            warm_s = 0.0
            cold_tok = 0
            warm_tok = 0
            sample = ""
            for i in range(batch):
                t0 = time.perf_counter()
                out = eng.generate(
                    f"thru-cold-{case['batch']}x{max_new}-{i}",
                    ids,
                    max_tokens=max_new,
                    temperature=0.0,
                )
                cold_s += time.perf_counter() - t0
                usage = out.get("usage") or {}
                cold_tok += int(usage.get("completion_tokens") or 0)
                if i == 0:
                    sample = eng.detokenize(out.get("output_ids") or [])[:160]
                if out.get("error"):
                    raise RuntimeError(out["error"])
            warm_runs = []
            warm_text = ""
            for run in range(args.warm_runs):
                run_s = 0.0
                run_tok = 0
                for i in range(batch):
                    t0 = time.perf_counter()
                    out = eng.generate(
                        f"thru-warm-{case['batch']}x{max_new}-{run}-{i}",
                        ids,
                        max_tokens=max_new,
                        temperature=0.0,
                    )
                    run_s += time.perf_counter() - t0
                    usage = out.get("usage") or {}
                    run_tok += int(usage.get("completion_tokens") or 0)
                    if out.get("error"):
                        raise RuntimeError(out["error"])
                    if i == 0:
                        warm_text = eng.detokenize(out.get("output_ids") or [])[:160]
                warm_runs.append({"seconds": run_s, "tokens": run_tok})
            warm_s = statistics.median(r["seconds"] for r in warm_runs)
            warm_tok = warm_runs[0]["tokens"]
            row = {
                "case": f"{batch}x{max_new}",
                "batch": batch,
                "max_new": max_new,
                "prompt_tokens": len(ids),
                "completion_tokens_cold": cold_tok,
                "completion_tokens_warm": warm_tok,
                "generate_cold_s": round(cold_s, 3),
                "generate_warm_s": round(warm_s, 3),
                "tok_s_cold": round(cold_tok / cold_s, 2) if cold_s > 0 else 0.0,
                "tok_s_warm": round(warm_tok / warm_s, 2) if warm_s > 0 else 0.0,
                "sample_text": sample,
                "warm_sample_text": warm_text,
                "warm_runs": warm_runs,
            }
            results.append(row)
            print(
                f"[thru] {row['case']} cold={row['tok_s_cold']} tok/s "
                f"warm={row['tok_s_warm']} tok/s load_s={load_s}"
            )
        stats = eng.get_stats()
    finally:
        try:
            eng.begin_drain()
            eng.shutdown()
        except Exception:
            pass

    payload = {
        "model": args.model,
        "device": args.device,
        "prompt": args.prompt,
        "load_s": load_s,
        "cases": results,
        "stats_tail": {
            "steps": (stats or {}).get("steps"),
            "cache_hit_count": ((stats or {}).get("cache") or {}).get("hit_count"),
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[thru] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
