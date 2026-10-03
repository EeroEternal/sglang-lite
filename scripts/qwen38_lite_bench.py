"""Fixed-length owned-runner speed probe, separate from numerical diagnostics."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path


def parse_lengths(value):
    lengths = [int(n) for n in value.split(",")]
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError("generation lengths must be positive")
    return lengths


def summarize(samples, length):
    if not samples or any(len(s["ids"]) != length for s in samples):
        raise ValueError("incomplete generation cannot be a throughput sample")
    times = [s["generation_s"] for s in samples]
    decode = [s["decode_s"] for s in samples]
    median = statistics.median(times)
    decode_median = statistics.median(decode)
    return {
        "case": f"1x{length}",
        "completion_tokens": length,
        "warm_runs_s": times,
        "warm_median_s": median,
        "warm_median_tok_s": length / median,
        "decode_steps": length - 1,
        "warm_decode_runs_s": decode,
        "warm_decode_median_s": decode_median,
        "warm_decode_tok_s": (length - 1) / decode_median if length > 1 else None,
        "greedy_repeat_ids_match": all(s["ids"] == samples[0]["ids"] for s in samples),
        "ids": samples[0]["ids"],
        "text": samples[0]["text"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="64,128,256")
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--full-logits", action="store_true", help="controlled old logits gather")
    parser.add_argument(
        "--full-qsa-selection", action="store_true", help="controlled unpruned QSA index scoring"
    )
    parser.add_argument("--capacity", type=int, default=4096)
    args = parser.parse_args()
    lengths = parse_lengths(args.lengths)
    if args.warm_runs < 1:
        parser.error("warm runs must be positive")

    import torch
    import torch.distributed as dist
    from tokenizers import Tokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
    from qwen38_runner.model import Qwen38Runner

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    model = Path(args.model).resolve()
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    prompt = "Write a short explanation of why the sky is blue."
    prompt_ids = tokenizer.encode(prompt).ids
    required = max(max(lengths) + (4 if args.profile else 0), 5)
    if not prompt_ids or len(prompt_ids) + required > args.capacity:
        parser.error("prompt/generation exceeds context")
    execution_limit = None if args.full_qsa_selection else len(prompt_ids) + required
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "engine": "owned-qwen38-runner",
        "timing_scope": "runner generation including prefill, tokenization and output decode; no serving scheduler; reset and rank barrier excluded",
        "decode_timing_scope": "GPU events for N-1 graph decode steps; no per-token host synchronization",
        "acceptance": "experimental speed only; full numerical acceptance pending",
        "parallel": {"tp": world, "ep": world, "attention_dp": 1, "pp": 1, "shared_expert_rank": 0},
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(rank),
        "capacity": args.capacity,
        "qsa_execution_limit": execution_limit,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(
            (model / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "cases": [],
        "greedy_selection": "full-logits" if args.full_logits else "rank-candidates",
        "gdn_conv": "eager",
    }
    dist.init_process_group("nccl")
    graph = None
    try:
        runner = Qwen38Runner(
            str(model), rank, world, args.capacity, execution_limit=execution_limit
        )
        prompt_gpu = torch.tensor(prompt_ids, device=runner.device, dtype=torch.int64)
        generated = torch.empty(args.capacity, device=runner.device, dtype=torch.int64)

        def step():
            if args.full_logits:
                return runner.step(runner.token).argmax(-1)
            return runner.step_greedy(runner.token)

        runner.token.copy_(prompt_gpu[:1])
        for _ in range(4):
            runner.token.copy_(step())
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.token.copy_(step())
            generated.index_copy_(0, runner.position - 1, runner.token)
        torch.cuda.synchronize()

        def run(length):
            runner.reset()
            torch.cuda.synchronize()
            dist.barrier()
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            if tokenizer.encode(prompt).ids != prompt_ids:
                raise RuntimeError("tokenization changed between runs")
            for index in range(len(prompt_ids)):
                runner.token.copy_(prompt_gpu[index : index + 1])
                graph.replay()
            decode_start.record()
            for _ in range(length - 1):
                graph.replay()
            decode_end.record()
            # Exactly one output transfer after all replay submissions.
            ids = generated[len(prompt_ids) - 1 : len(prompt_ids) + length - 1].cpu().tolist()
            text = tokenizer.decode(ids)
            elapsed = time.perf_counter() - started
            return {
                "generation_s": elapsed,
                "decode_s": decode_start.elapsed_time(decode_end) / 1000,
                "ids": ids,
                "text": text,
            }

        for length in lengths:
            cold = run(length)
            samples = [run(length) for _ in range(args.warm_runs)]
            row = summarize(samples, length)
            row["cold_s"] = cold["generation_s"]
            report["cases"].append(row)
            if rank == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            runner.reset()
            for index in range(len(prompt_ids)):
                runner.token.copy_(prompt_gpu[index : index + 1])
                graph.replay()
            for _ in range(max(lengths) - 1):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                report["profile_start_position"] = len(prompt_ids) + max(lengths) - 1
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(4):
                        graph.replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(output.with_suffix(".trace.json")))
                report["kernel_profile"] = prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30
                )
                print(report["kernel_profile"], flush=True)
            else:
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
            dist.barrier()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if graph is not None:
            graph.reset()
        if rank == 0:
            output.write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
