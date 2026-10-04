"""Owned GPU Qwen3.8 prototype: correctness first, not a serving entrypoint."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from qwen38_runner.model import Qwen38Runner


def tensor_audit(root):
    """Inspect owned persistent tensors, not CUDA temporaries/NCCL host staging."""
    seen, rows = set(), []

    def visit(value, name):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            rows.append(
                {
                    "name": name,
                    "device": str(value.device),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{name}.{index}")
        elif hasattr(value, "__dict__"):
            visit(vars(value), name)

    visit(root, "runner")
    bad = [r for r in rows if not r["device"].startswith("cuda:")]
    if bad:
        raise RuntimeError(f"non-CUDA persistent tensors: {bad}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--golden")
    p.add_argument("--teacher-force", action="store_true")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument("--attention-tp", type=int, choices=(2, 4, 8), default=8)
    p.add_argument("--serial-shared-expert", action="store_true", help="paired control")
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    prompt = "Write a short explanation of why the sky is blue."
    ids = tokenizer.encode(prompt).ids
    required = max(args.tokens, 4) if args.graph else args.tokens
    if args.tokens < 1 or len(ids) + required > args.capacity:
        p.error("generation exceeds the context capacity")
    reference = None
    if args.golden:
        reference = json.loads(Path(args.golden).read_text())["runs"][0]
        if len(reference["ids"]) < args.tokens:
            p.error("golden reference is shorter than the requested sample")
    if args.teacher_force and reference is None:
        p.error("teacher forcing requires --golden")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    report = {
        "runner": "owned-qwen38-prototype",
        "prompt_ids": ids,
        "ids": [],
        "logprobs": [],
        "parallel": {
            "tp": world,
            "ep": world,
            "attention_tp": args.attention_tp,
            "attention_replicas": world // args.attention_tp,
            "attention_dp": 1,
            "pp": 1,
            "shared_expert_rank": 0,
            "shared_expert_stream": not args.serial_shared_expert,
        },
        "graph": args.graph,
        "capacity": args.capacity,
        "teacher_forced": args.teacher_force,
        "qsa_execution_limit": len(ids) + required,
    }
    graph = None
    try:
        load_start = time.perf_counter()
        runner = Qwen38Runner(
            args.model,
            rank,
            world,
            args.capacity,
            execution_limit=len(ids) + required,
            attention_tp=args.attention_tp,
            shared_expert_stream=not args.serial_shared_expert,
        )
        report["load_s"] = time.perf_counter() - load_start
        audit = tensor_audit(runner)
        audit_path = Path(args.out).with_suffix(f".rank{rank}.audit.json")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        report["persistent_cuda_tensors_per_rank"] = len(audit)
        logits = None
        runner.reset()
        for token in ids:
            runner.token.fill_(token)
            logits = runner.step(runner.token)
        # Compile/warm operators before optional capture; restore all mutable state.
        if args.graph:
            for _ in range(3):
                runner.token.copy_(logits.argmax(-1))
                logits = runner.step(runner.token)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_logits = runner.step(runner.token)
                runner.token.copy_(captured_logits.argmax(-1))
            torch.cuda.synchronize()
            runner.reset()
            for token in ids:
                runner.token.fill_(token)
                logits = runner.step(runner.token)
        next_id = logits.argmax(-1)
        runner.token.copy_(next_id)
        started = time.perf_counter()
        for index in range(args.tokens):
            token = int(next_id.item())
            report["ids"].append(token)
            selected = reference["ids"][index] if args.teacher_force else token
            report["logprobs"].append(float(logits.log_softmax(-1)[0, selected].item()))
            if index + 1 == args.tokens:
                break
            if args.teacher_force:
                runner.token.fill_(selected)
            if args.graph:
                graph.replay()
                logits = captured_logits
                next_id = runner.token.clone()
            else:
                logits = runner.step(runner.token)
                next_id = logits.argmax(-1)
                runner.token.copy_(next_id)
        torch.cuda.synchronize()
        report["generation_s"] = time.perf_counter() - started
        report["timing_scope"] = "correctness probe with host token/logprob synchronization"
        report["text"] = tokenizer.decode(report["ids"])
        if reference is not None:
            expected = reference["ids"][: args.tokens]
            differences = [i for i, (a, b) in enumerate(zip(expected, report["ids"])) if a != b]
            report["first_difference"] = differences[0] if differences else None
            report["golden_ids"] = expected
            report["golden_logprobs"] = [v[0] for v in reference["logprobs"][: args.tokens]]
            report["exact_ids_match"] = len(expected) == args.tokens and not differences
            report["top1_matches"] = args.tokens - len(differences)
            if args.teacher_force:
                report["max_abs_logprob_error"] = max(
                    abs(a - b) for a, b in zip(report["logprobs"], report["golden_logprobs"])
                )
        if rank == 0:
            print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if rank == 0:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        # NCCL graph references must be released before communicator teardown.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
