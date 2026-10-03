#!/usr/bin/env python3
"""GPU-resident Qwen3.8 NVFP4 baseline, separate from the lite engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path


def engine_args(model: str, tp: int, attention_dp: int = 1) -> dict:
    """Use SM120 NVFP4 kernels without PLE, weight, or unified-memory offload."""
    if tp not in (2, 4, 8):
        raise ValueError("NVFP4 baseline requires --tp 2, 4, or 8")
    if attention_dp not in (1, 2) or tp % attention_dp:
        raise ValueError("attention DP must be 1 or 2 and divide TP")
    return {
        "model_path": model,
        "tp_size": tp,
        "ep_size": tp,
        "dp_size": attention_dp,
        "enable_dp_attention": attention_dp > 1,
        "dtype": "bfloat16",
        "quantization": "modelopt_fp4",
        "moe_runner_backend": "flashinfer_cutlass",
        "fp4_gemm_runner_backend": "flashinfer_cutlass",
        "page_size": 64,
        "mamba_radix_cache_strategy": "extra_buffer",
        "mamba_track_interval": 64,
        "context_length": 4096,
        "chunked_prefill_size": 2048,
        "max_running_requests": 8,
        "cuda_graph_max_bs_decode": 8,
        "mem_fraction_static": 0.80,
        "ple_offload_embedding": False,
        "cpu_offload_gb": 0,
        "enable_unified_memory": False,
        "speculative_algorithm": None,
    }


def checkpoint_identity(model: Path) -> dict:
    """Reject partially downloaded checkpoints before allocating GPU memory."""
    index_bytes = (model / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError("checkpoint index has no weight shards")
    missing = [name for name in shards if not (model / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint incomplete: {len(missing)} missing shards")
    return {
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "shard_sizes": {name: (model / name).stat().st_size for name in shards},
    }


def checked_generation(engine, prompt: str, length: int) -> tuple[dict, float]:
    """Measure a fixed-length greedy generation and reject incomplete outputs."""
    start = time.perf_counter()
    output = engine.generate(
        prompt, {"max_new_tokens": length, "temperature": 0, "ignore_eos": True}
    )
    elapsed = time.perf_counter() - start
    tokens = (output.get("meta_info") or {}).get("completion_tokens")
    if tokens != length:
        raise RuntimeError(f"expected {length} completion tokens, got {tokens!r}")
    return output, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--attention-dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prompt", default="Write a short explanation of why the sky is blue.")
    parser.add_argument("--lengths", default="64,128,256")
    parser.add_argument("--warm-runs", type=int, default=3)
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(",")]
    if args.warm_runs < 1 or not lengths or any(n < 1 for n in lengths):
        parser.error("warm runs and generation lengths must be positive")

    model = Path(args.model).expanduser().resolve()
    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "qwen4_exp":
        parser.error("expected a Qwen3.8-Flash-Next qwen4_exp checkpoint")
    if (config.get("quantization_config") or {}).get("quant_algo") != "NVFP4":
        parser.error("expected the RadixArk NVFP4 checkpoint")
    kwargs = engine_args(str(model), args.tp, args.attention_dp)
    identity = checkpoint_identity(model)

    import torch
    from sglang import Engine

    if torch.cuda.device_count() < args.tp:
        parser.error(f"need {args.tp} visible GPUs, got {torch.cuda.device_count()}")
    payload = {
        "engine": "sglang",
        "sglang_version": importlib.metadata.version("sglang"),
        "torch_version": torch.__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(args.tp)],
        "model": str(model),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint": identity,
        "timing_scope": "end-to-end generation, including prefill and scheduling",
        "engine_args": kwargs,
        "gpu_only_requested": True,
        "text_only_requests": True,
        "prompt": args.prompt,
        "cases": [],
    }
    engine = None
    try:
        started = time.perf_counter()
        engine = Engine(**kwargs)
        payload["load_s"] = time.perf_counter() - started
        checked_generation(engine, args.prompt, 8)
        for length in lengths:
            _output, cold_s = checked_generation(engine, args.prompt, length)
            runs = []
            texts = []
            for _ in range(args.warm_runs):
                output, elapsed = checked_generation(engine, args.prompt, length)
                runs.append(elapsed)
                texts.append(output.get("text", ""))
            row = {
                "case": f"1x{length}",
                "completion_tokens": length,
                "cold_s": cold_s,
                "warm_runs_s": runs,
                "warm_median_s": statistics.median(runs),
                "warm_median_tok_s": length / statistics.median(runs),
                "greedy_repeat_text_match": all(t == texts[0] for t in texts),
                "sample_text": texts[0][:400],
                "output_text": texts[0],
                "output_sha256": hashlib.sha256(texts[0].encode("utf-8")).hexdigest(),
            }
            payload["cases"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
