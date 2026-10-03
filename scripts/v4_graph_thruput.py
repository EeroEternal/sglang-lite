#!/usr/bin/env python3
"""Real V4 decode throughput: one CUDA graph, GPU expert gather, optional one-shot allreduce.

Position advances. This is not the position-0 loop in v4_deep_gemm_thruput.py.

  CUDA_HOME=/usr/local/cuda torchrun --nproc-per-node=8 scripts/v4_graph_thruput.py --max-new 128

Do not set CPLUS_INCLUDE_PATH. System nvcc (CUDA 13.3) compiles the inline ops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

def _select_rank_device() -> None:
    """Give each torchrun worker one visible GPU before importing CUDA."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) > 1:
        rank = int(local_rank)
        if rank >= len(devices):
            raise ValueError(f"LOCAL_RANK={rank} exceeds {len(devices)} visible GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    elif not devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank


_select_rank_device()

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")


def _stage(rank: int, name: str) -> None:
    if rank == 0:
        print(f"STAGE {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--check", type=int, default=8)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--deep-gemm", type=int, default=1)
    ap.add_argument("--fast-ar", type=int, default=1)
    ap.add_argument("--profile", type=int, default=0)
    args = ap.parse_args()
    if not args.deep_gemm:
        ap.error("CUDA graph decode requires grouped DeepGEMM; use v4_deep_gemm_thruput.py for TileLang")

    root = Path(__file__).resolve().parents[1]
    vendor = root / "engine" / "vendor" / "deepseek_infer"
    if not (vendor / "model.py").is_file():
        print("missing vendor deepseek_infer", file=sys.stderr)
        return 2
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(root / "engine"))
    os.environ["SGLANG_LITE_V4_DEEP_GEMM"] = "1" if args.deep_gemm else "0"
    os.environ.setdefault("SGLANG_LITE_V4_MOE_FAST", "1")
    os.environ.setdefault("SGLANG_LITE_V4_MOE_GROUPED", "1")

    from v4_decode_fast import check_index_formulas

    check_index_formulas()

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    oneshot = False
    if args.fast_ar and world > 1:
        _stage(rank, "oneshot")
        from v4_fast_ops import install_oneshot_allreduce, oneshot_self_test, uninstall_oneshot_allreduce

        try:
            oneshot = bool(oneshot_self_test())
        except Exception:
            uninstall_oneshot_allreduce()
            if rank == 0:
                traceback.print_exc()
            oneshot = False
        if not oneshot:
            uninstall_oneshot_allreduce()

    _stage(rank, "load")
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from model import ModelArgs, Transformer

    from v4_moe_fast import attach_v4_moe_fast
    from v4_decode_fast import (
        attach_v4_decode_fast,
        capture_decode,
        prepare_decode,
        restore_buffers,
        set_fast,
        snapshot_buffers,
    )

    cfg = json.loads((vendor / "config.json").read_text(encoding="utf-8"))
    if isinstance(cfg.get("compress_ratios"), list):
        cfg["compress_ratios"] = tuple(cfg["compress_ratios"])
    names = {f.name for f in fields(ModelArgs)}
    margs = ModelArgs(**{k: v for k, v in cfg.items() if k in names})
    with torch.device("cuda"):
        model = Transformer(margs)
    ckpt = Path(os.environ["SGLANG_LITE_DSV4_CONVERTED"]).expanduser()
    load_model(model, str(ckpt / f"model{rank}-mp{world}.safetensors"), strict=False)
    model.eval()
    attach_v4_moe_fast(model)
    if args.deep_gemm:
        from v4_deep_gemm import attach_v4_deep_gemm

        st = attach_v4_deep_gemm(model)
        if rank == 0:
            print("ATTACH", st, flush=True)
    attach_v4_decode_fast(model)

    _stage(rank, "prepare")
    prep = prepare_decode(model)
    if rank == 0:
        print("PREPARE", prep, flush=True)

    hf = Path(os.environ.get("SGLANG_LITE_DSV4_HF", os.path.expanduser("~/models/ds-v4-flash")))
    tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
    prompt_ids = tok.encode(encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat"))
    prompt_len = len(prompt_ids)

    @torch.inference_mode()
    def prefill():
        return model(torch.tensor([prompt_ids], dtype=torch.long, device="cuda"), start_pos=0)

    _stage(rank, "prefill")
    set_fast(False)
    prefill_logits = prefill()
    torch.cuda.synchronize()
    first = int(prefill_logits[0].argmax().item())
    if rank == 0:
        print(f"PROMPT_LEN {prompt_len} FIRST {first}", flush=True)

    snap = snapshot_buffers(model)
    n_check = int(args.check)
    eager_out: list[int] = []
    eager_logits = None
    nxt = first
    _stage(rank, "eager_ref")
    for step in range(n_check):
        logits = model(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), start_pos=prompt_len + step)
        if step == 0:
            eager_logits = logits.detach().float().clone()
        nxt = int(logits[0].argmax().item())
        eager_out.append(nxt)
    torch.cuda.synchronize()
    restore_buffers(snap)

    static_tokens = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    static_pos = model._decode_pos
    static_tokens.fill_(first)
    static_pos.fill_(prompt_len)
    set_fast(True)
    _stage(rank, "capture")
    if world > 1:
        dist.barrier()
    try:
        graph, static_logits = capture_decode(model, static_tokens, autoregressive=True)
    except Exception:
        if rank == 0:
            traceback.print_exc()
            print("CAPTURE_FAIL", flush=True)
        return 1
    if rank == 0:
        print("CAPTURE_OK", flush=True)
    restore_buffers(snap)
    if world > 1:
        dist.barrier()

    inputs = [first]
    for tok_id in eager_out[:-1]:
        inputs.append(tok_id)
    graph_out: list[int] = []
    graph_logits = None
    _stage(rank, "check")
    for step, tok_id in enumerate(inputs):
        static_tokens.fill_(tok_id)
        static_pos.fill_(prompt_len + step)
        graph.replay()
        torch.cuda.synchronize()
        if step == 0:
            graph_logits = static_logits.detach().float().clone()
        graph_out.append(int(static_logits[0].argmax().item()))
    if world > 1:
        dist.barrier()

    logit_max_abs = -1.0
    if eager_logits is not None and graph_logits is not None:
        logit_max_abs = float((graph_logits - eager_logits).abs().max().item())
    match = graph_out == eager_out
    if rank == 0:
        print(
            f"CHECK match={match} logit_max_abs={logit_max_abs:.3e} eager={eager_out} graph={graph_out}",
            flush=True,
        )
    if world > 1:
        flag = torch.tensor([0 if match else 1], device="cuda", dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        match = int(flag.item()) == 0
        dist.barrier()
    if not match:
        if rank == 0:
            print("CHECK_FAIL", flush=True)
        return 2

    restore_buffers(snap)
    static_tokens.fill_(first)
    static_pos.fill_(prompt_len)
    # One replay so the timed loop does not include graph-replay warmup.
    graph.replay()
    torch.cuda.synchronize()
    restore_buffers(snap)
    static_tokens.fill_(first)
    static_pos.fill_(prompt_len)

    _stage(rank, "time")
    if world > 1:
        dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.max_new):
        graph.replay()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    dt = time.perf_counter() - t0
    if rank == 0:
        print(
            "RESULT "
            + json.dumps(
                {
                    "tok_s": round(args.max_new / dt, 3),
                    "pure_decode_s": round(dt, 4),
                    "max_new": args.max_new,
                    "prompt_len": prompt_len,
                    "capture": True,
                    "oneshot": oneshot,
                    "check": n_check,
                    "logit_max_abs": None if logit_max_abs < 0 else round(logit_max_abs, 6),
                    "world": world,
                    "start_pos": "advancing",
                }
            ),
            flush=True,
        )
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                static_tokens.fill_(first)
                static_pos.fill_(prompt_len)
                graph.replay()
            torch.cuda.synchronize()
        if rank == 0:
            print("PROFILE", flush=True)
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25),
                flush=True,
            )
        if world > 1:
            dist.barrier()
    # destroy_process_group hangs after a CUDA graph that captured NCCL.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
