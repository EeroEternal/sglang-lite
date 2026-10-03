#!/usr/bin/env python3
"""Separate correctness/profile probe; these timings are not the baseline KPI."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

from qwen38_sglang_baseline import checkpoint_identity, engine_args


def first_difference(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def install_worker_audit():
    """Observe parameters and persistent pools without copying tensor contents."""
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.forward

    @functools.wraps(original)
    def audited_forward(self, *args, **kwargs):
        if not getattr(self, "_qwen38_audited", False):
            tensors = {}
            for name, tensor in self.model.named_parameters():
                tensors["parameter." + name] = tensor
            for name, tensor in self.model.named_buffers():
                tensors["buffer." + name] = tensor
            seen = set()

            def visit(value, prefix, depth=0):
                if depth > 6 or id(value) in seen:
                    return
                seen.add(id(value))
                if isinstance(value, torch.Tensor):
                    tensors[prefix] = value
                elif isinstance(value, dict):
                    for key, item in value.items():
                        visit(item, f"{prefix}.{key}", depth + 1)
                elif isinstance(value, (tuple, list)):
                    for key, item in enumerate(value):
                        visit(item, f"{prefix}.{key}", depth + 1)
                elif hasattr(value, "__dict__"):
                    for key, item in vars(value).items():
                        visit(item, f"{prefix}.{key}", depth + 1)

            visit(self.req_to_token_pool, "request_pool")
            visit(self.token_to_kv_pool, "kv_pool")
            rows = {
                name: {"device": str(t.device), "dtype": str(t.dtype), "shape": list(t.shape)}
                for name, t in tensors.items()
            }
            forbidden = [
                name for name, t in tensors.items()
                if t.device.type != "cuda"
                and (name.startswith("parameter.") or t.is_floating_point())
            ]
            rank = self.ps.tp_rank
            Path(os.environ["QWEN38_AUDIT_DIR"], f"rank-{rank}.json").write_text(
                json.dumps({"tensors": rows, "non_cuda_weights_or_float_states": forbidden}, indent=2)
            )
            if forbidden:
                raise RuntimeError(f"non-GPU model weights/state: {forbidden[:10]}")
            self._qwen38_audited = True
        return original(self, *args, **kwargs)

    ModelRunner.forward = audited_forward


# Spawned workers re-import this module before loading the model runner.
if os.environ.get("QWEN38_AUDIT_DIR"):
    install_worker_audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    target = Path(args.out).resolve()
    target.mkdir(parents=True, exist_ok=True)
    audits = target / "residency"
    audits.mkdir(exist_ok=True)
    os.environ["QWEN38_AUDIT_DIR"] = str(audits)
    install_worker_audit()

    from sglang import Engine

    model = Path(args.model).resolve()
    payload = {"checkpoint": checkpoint_identity(model), "runs": []}
    options = engine_args(str(model), args.tp)
    options["enable_deterministic_inference"] = args.deterministic
    payload["engine_args"] = options
    engine = None
    try:
        engine = Engine(**options)
        info = engine.get_server_info()
        payload["resolved"] = {
            key: info.get(key) for key in (
                "attention_backend", "linear_attn_backend", "sampling_backend",
                "moe_runner_backend", "ple_offload_embedding", "cpu_offload_gb",
                "enable_unified_memory", "mamba_ssm_dtype",
            )
        }
        prompt = "Write a short explanation of why the sky is blue."
        for label, flush in [("fresh-0", True), ("cached-0", False),
                             ("fresh-1", True), ("cached-1", False),
                             ("fresh-2", True)]:
            if flush:
                engine.flush_cache()
            output = engine.generate(
                prompt, {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True},
                return_logprob=True, top_logprobs_num=2,
            )
            meta = output["meta_info"]
            if meta["completion_tokens"] != 64:
                raise RuntimeError("incomplete correctness sample")
            ids = [entry[1] for entry in meta["output_token_logprobs"]]
            payload["runs"].append({
                "label": label, "ids": ids, "text": output["text"],
                "logprobs": meta["output_token_logprobs"],
                "top2": meta["output_top_logprobs"],
                "cached_tokens": meta.get("cached_tokens"),
            })
        reference = payload["runs"][0]["ids"]
        for row in payload["runs"]:
            row["first_difference"] = first_difference(reference, row["ids"])
        if args.profile:
            engine.start_profile(output_dir=str(target / "profile"), activities=["CPU", "GPU"],
                                 with_stack=False, record_shapes=True)
            engine.generate(prompt, {"temperature": 0, "max_new_tokens": 32, "ignore_eos": True})
            engine.stop_profile()
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            (target / "diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
