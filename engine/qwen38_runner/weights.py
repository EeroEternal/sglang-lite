# Copyright 2026 sglang-lite contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit checkpoint slices; CPU is used only during weight loading."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


class Weights:
    def __init__(self, path, device):
        self.path = Path(path)
        self.device = device
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.files = {}

    def source(self, name):
        filename = self.index[name]
        if filename not in self.files:
            self.files[filename] = safe_open(self.path / filename, framework="pt", device="cpu")
        return self.files[filename].get_slice(name)

    def get(self, name, rows=None, cols=None, dtype=None):
        view = self.source(name)
        if not view.get_shape():
            if rows is not None or cols is not None:
                raise ValueError(f"cannot slice scalar {name}")
            value = self.files[self.index[name]].get_tensor(name)
        else:
            value = view[:] if rows is None else view[rows]
        if cols is not None:
            value = value[:, cols]
        return value.to(device=self.device, dtype=dtype).contiguous()

    def shape(self, name):
        return self.source(name).get_shape()

    def close(self):
        self.files.clear()


def row_slice(size, rank, world):
    if size % world:
        raise ValueError(f"{size} rows cannot be split across {world} ranks")
    count = size // world
    return slice(rank * count, (rank + 1) * count)


def cutlass_up_gate(gate, up):
    """CUTLASS's SwiGLU uses [Up, Gate], including the blockscale rows."""
    return torch.cat([up, gate], dim=1).contiguous()


class NVFP4Experts:
    """FlashInfer leaf kernel with checkpoint-preserving W4A4 scales."""

    def __init__(self, weights, prefix, rank, world, config):
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

        self.rank, self.world = rank, world
        count = config["num_experts"] // world
        packed, scales, globals_, inputs = {}, {}, {}, {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names = [
                f"{prefix}.experts.{e}.{projection}"
                for e in range(rank * count, (rank + 1) * count)
            ]
            packed[projection] = torch.stack([weights.get(n + ".weight") for n in names])
            scales[projection] = torch.stack([weights.get(n + ".weight_scale") for n in names])
            globals_[projection] = torch.stack(
                [weights.get(n + ".weight_scale_2") for n in names]
            ).flatten()
            inputs[projection] = torch.stack(
                [weights.get(n + ".input_scale") for n in names]
            ).flatten()
        if not torch.equal(globals_["gate_proj"], globals_["up_proj"]):
            raise ValueError("CUTLASS requires identical gate/up global weight scales")
        w13 = cutlass_up_gate(packed["gate_proj"], packed["up_proj"])
        sf13 = cutlass_up_gate(scales["gate_proj"], scales["up_proj"])
        sf2 = scales["down_proj"]
        # Each expert has an independent scale layout; do not interleave across E.
        sf13 = torch.stack(
            [nvfp4_block_scale_interleave(s.view(torch.uint8)).reshape(s.shape) for s in sf13]
        )
        sf2 = torch.stack(
            [nvfp4_block_scale_interleave(s.view(torch.uint8)).reshape(s.shape) for s in sf2]
        )
        a1 = torch.cat([inputs["gate_proj"], inputs["up_proj"]]).max()
        a2 = inputs["down_proj"].max()
        self.w13 = w13.view(torch.int64)
        self.w2 = packed["down_proj"].contiguous().view(torch.int64)
        self.quant = [
            (1 / a1).float(),
            sf13.view(torch.int32),
            (a1 * globals_["gate_proj"]).float(),
            (1 / a2).float(),
            sf2.view(torch.int32),
            (a2 * globals_["down_proj"]).float(),
        ]
        self.output = torch.empty(
            (1, config["hidden_size"]), dtype=torch.bfloat16, device=weights.device
        )

    def __call__(self, x, ids, probabilities):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.tllm_enums import ActivationType

        return cutlass_fused_moe(
            input=x,
            token_selected_experts=ids.int(),
            token_final_scales=probabilities,
            fc1_expert_weights=self.w13,
            fc2_expert_weights=self.w2,
            output_dtype=torch.bfloat16,
            quant_scales=self.quant,
            ep_size=self.world,
            ep_rank=self.rank,
            tp_size=1,
            tp_rank=0,
            output=self.output,
            tune_max_num_tokens=1,
            activation_type=ActivationType.Swiglu,
        )[0]
