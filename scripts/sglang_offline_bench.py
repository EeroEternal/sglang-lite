#!/usr/bin/env python3
"""SGLang offline thruput probe (run inside sglang docker or host with sglang).

  python scripts/sglang_offline_bench.py \\
    --model /model --out /bench/thru_sglang.json --max-new 128
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--warm-runs", type=int, default=1)
    args = ap.parse_args()
    if args.warm_runs < 1:
        ap.error("--warm-runs must be positive")

    payload = {
        "engine": "sglang",
        "model": args.model,
        "cases": [],
        "error": None,
    }
    try:
        from sglang import Engine

        t0 = time.perf_counter()
        engine = Engine(
            model_path=args.model,
            tp_size=1,
            trust_remote_code=True,
            mem_fraction_static=args.mem_fraction,
            disable_cuda_graph=False,
        )
        payload["load_s"] = round(time.perf_counter() - t0, 3)

        # warmup
        engine.generate(args.prompt, {"max_new_tokens": 8, "temperature": 0})

        lengths = [64, 128] if args.max_new >= 128 else [args.max_new]
        for n in lengths:
            t1 = time.perf_counter()
            o1 = engine.generate(
                args.prompt,
                {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
            )
            cold_s = time.perf_counter() - t1
            warm_runs = []
            for _ in range(args.warm_runs):
                t2 = time.perf_counter()
                o2 = engine.generate(
                    args.prompt,
                    {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
                )
                warm_runs.append(time.perf_counter() - t2)
            warm_s = statistics.median(warm_runs)

            def ntok(o, fallback: int) -> int:
                if isinstance(o, dict):
                    m = o.get("meta_info") or o.get("meta") or {}
                    for k in (
                        "completion_tokens",
                        "completion_token_num",
                        "n_output_tokens",
                    ):
                        if k in m:
                            return int(m[k])
                return fallback

            ctok, wtok = ntok(o1, n), ntok(o2, n)
            row = {
                "case": f"1x{n}",
                "batch": 1,
                "max_new": n,
                "completion_tokens_cold": ctok,
                "completion_tokens_warm": wtok,
                "generate_cold_s": round(cold_s, 3),
                "generate_warm_s": round(warm_s, 3),
                "tok_s_cold": round(ctok / cold_s, 2) if cold_s > 0 else 0.0,
                "tok_s_warm": round(wtok / warm_s, 2) if warm_s > 0 else 0.0,
                "sample_text": str(
                    (o1.get("text") if isinstance(o1, dict) else o1) or ""
                )[:120],
                "warm_sample_text": str(
                    (o2.get("text") if isinstance(o2, dict) else o2) or ""
                )[:160],
                "warm_runs_s": warm_runs,
            }
            payload["cases"].append(row)
            print("[sglang]", row)
        try:
            engine.shutdown()
        except Exception:
            pass
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        payload["traceback"] = traceback.format_exc()[-2000:]
        print(payload["error"])

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 2 if payload.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
