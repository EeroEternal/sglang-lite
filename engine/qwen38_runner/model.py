# Copyright 2026 sglang-lite contributors
# Licensed under the Apache License, Version 2.0.
"""Owned single-sequence Qwen3.8 state and complete text decode graph.

Equations/layouts are adapted from SGLang v0.5.20 (Apache-2.0), pinned at
94602c9c2b7cbdb8efd5c52802dac6a1c180089e. No runtime SGLang dependency.
This path must pass real-weight validation before becoming a serving default.
"""

import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import ops
from .config import attention_layout, validate_config
from .weights import NVFP4Experts, Weights, row_slice


class HyperConnection:
    def __init__(self, weights, prefix):
        self.norm = weights.get(prefix + ".hc_norm.weight")
        self.down = weights.get(prefix + ".input_mix_weight_down.weight")
        self.up = weights.get(prefix + ".input_mix_weight_up.weight")
        name = prefix + ".block_inject_weight.weight"
        self.inject = weights.get(name) if name in weights.index else None

    def mix(self, value):
        mixed, normalized = ops.hc_mix(value, self.norm, self.down, self.up, 2560, 4)
        return mixed, (value, normalized)

    def combine(self, value, residual):
        return ops.hc_combine(value, *residual, self.inject, 2560, 4)


class GDN:
    def __init__(self, weights, prefix, rank, world):
        key, value = 2048, 6144
        ks, vs = row_slice(key, rank, world), row_slice(value, rank, world)
        sections = [
            ks,
            slice(key + ks.start, key + ks.stop),
            slice(2 * key + vs.start, 2 * key + vs.stop),
        ]
        self.input = torch.cat([weights.get(prefix + ".in_proj_qkv.weight", s) for s in sections])
        self.z = weights.get(prefix + ".in_proj_z.weight", vs)
        hs = row_slice(48, rank, world)
        self.a = weights.get(prefix + ".in_proj_a.weight", hs)
        self.b = weights.get(prefix + ".in_proj_b.weight", hs)
        self.bias = weights.get(prefix + ".dt_bias", hs, dtype=torch.float32)
        self.log_a = weights.get(prefix + ".A_log", hs, dtype=torch.float32)
        self.conv = (
            torch.cat([weights.get(prefix + ".conv1d.weight", s) for s in sections])
            .squeeze(1)
            .float()
        )
        self.norm = weights.get(prefix + ".norm.weight")
        self.out = weights.get(prefix + ".out_proj.weight", cols=vs)
        self.keys, self.values = 16 // world, 48 // world
        self.history = torch.zeros(
            (self.input.shape[0], 3), device=weights.device, dtype=torch.bfloat16
        )
        self.state = torch.zeros(
            (self.values, 128, 128), device=weights.device, dtype=torch.float32
        )

    def reset(self):
        self.history.zero_()
        self.state.zero_()

    def __call__(self, value):
        projected = F.linear(value, self.input)
        conv = F.silu(ops.gdn_conv_sum(projected, self.history, self.conv)).to(value.dtype)
        q, k, v = conv.split([self.keys * 128, self.keys * 128, self.values * 128])
        output = ops.gdn_update(
            q.reshape(-1, 128),
            k.reshape(-1, 128),
            v.reshape(-1, 128),
            F.linear(value, self.z).reshape(-1, 128),
            F.linear(value, self.b).flatten(),
            F.linear(value, self.a).flatten(),
            self.bias,
            self.log_a,
            self.state,
            self.norm,
        )
        return F.linear(output, self.out)


class QSA:
    def __init__(self, weights, prefix, rank, world, capacity, execution_limit=None):
        self.heads = 24 // world
        self.q = weights.get(prefix + ".q_proj.weight", row_slice(12288, rank, world))
        kv_rank = rank // (world // 2)
        kv_rows = row_slice(512, kv_rank, 2)
        self.k = weights.get(prefix + ".k_proj.weight", kv_rows)
        self.v = weights.get(prefix + ".v_proj.weight", kv_rows)
        self.out = weights.get(prefix + ".o_proj.weight", cols=row_slice(6144, rank, world))
        self.qnorm = weights.get(prefix + ".q_norm.weight").float() + 1.0
        self.knorm = weights.get(prefix + ".k_norm.weight").float() + 1.0
        self.i_proj = weights.get(prefix + ".indexer.index_qk_proj.weight")
        self.iqnorm = weights.get(prefix + ".indexer.q_layernorm.weight").float() + 1.0
        self.iknorm = weights.get(prefix + ".indexer.k_layernorm.weight").float() + 1.0
        self.keys = torch.zeros((capacity, 256), device=weights.device, dtype=torch.bfloat16)
        self.values = torch.zeros_like(self.keys)
        self.index_raw = torch.zeros((capacity, 128), device=weights.device, dtype=torch.bfloat16)
        blocks = (capacity + 3) // 4
        self.blocks = torch.arange(blocks, device=weights.device)
        self.slot_offsets = torch.arange(4, device=weights.device)
        self.compressed = torch.zeros((blocks, 128), device=weights.device, dtype=torch.bfloat16)
        self.capacity = capacity
        self.dense_limit = execution_limit if execution_limit and execution_limit <= 2048 else 0
        self.dense_selected = (
            torch.full((1,), -1, device=weights.device, dtype=torch.int32)
            if self.dense_limit
            else None
        )

    def reset(self):
        for state in (self.keys, self.values, self.index_raw, self.compressed):
            state.zero_()

    def __call__(self, value, position, cos, sin, rope_cos, rope_sin):
        q_gate = F.linear(value, self.q).reshape(self.heads, 512)
        q, gate = q_gate.chunk(2, -1)
        q = ops.rope(ops.rms(q, self.qnorm, gemma=False), cos, sin)
        k = ops.rope(ops.rms(F.linear(value, self.k), self.knorm, gemma=False), cos, sin)
        self.keys.index_copy_(0, position, k)
        self.values.index_copy_(0, position, F.linear(value, self.v))
        index_qk = F.linear(value, self.i_proj).reshape(5, 128)
        self.index_raw.index_copy_(0, position, index_qk[4:])
        block = position // 4
        raw = ops.qsa_block_mean(self.index_raw, position)
        normalized = ops.rms(raw, self.iknorm, gemma=False)
        block_start = block * 4
        compressed = ops.rope(
            normalized, rope_cos.index_select(0, block_start), rope_sin.index_select(0, block_start)
        )
        ops.store_qsa_compressed(self.compressed, position, compressed)
        if self.dense_limit:
            selected = self.dense_selected
        else:
            iq = ops.rope(ops.rms(index_qk[:4], self.iqnorm, gemma=False), cos, sin)
            scores = F.relu(iq.float() @ self.compressed.float().T).sum(0)
            scores = scores.masked_fill(self.blocks >= (position + 1) // 4, -float("inf"))
            chosen = torch.topk(scores, min(512, scores.numel())).indices
            selected = (chosen[:, None] * 4 + self.slot_offsets).flatten()
            pending = ((position + 1) // 4) * 4 + self.slot_offsets
            selected = torch.cat([selected, pending])
            selected = torch.where(selected <= position, selected, -1).int()
        output = ops.attention(
            q, self.keys, self.values, position, selected, dense_limit=self.dense_limit
        )
        return F.linear((output * torch.sigmoid(gate)).reshape(1, -1), self.out)


class PLE:
    def __init__(self, weights, prefix, rank, world):
        base = prefix + ".ple_embedding"
        self.multipliers = weights.get(base + ".layer_multipliers", dtype=torch.int64)
        self.sizes = weights.get(base + ".ngram_heads_vocab_sizes", dtype=torch.int64)
        self.offsets = weights.get(base + ".ngram_heads_offsets", dtype=torch.int64)
        total = int((self.sizes[-1] + self.offsets[-1]).item())
        padded = (total + 127) // 128 * 128
        section = row_slice(padded, rank, world)
        self.start = section.start
        self.table = torch.zeros(
            (section.stop - section.start, 160), device=weights.device, dtype=torch.float8_e4m3fn
        )
        names = [
            n
            for n in weights.index
            if n.startswith(base + ".ngram_embedding.shard_") and n.endswith(".weight")
        ]
        parts = {int(re.search(r"shard_(\d+)", n).group(1)) for n in names}
        if parts != set(range(128)):
            raise ValueError("incomplete PLE table index")
        shard_rows = (padded + 127) // 128
        for name in names:
            number = int(re.search(r"shard_(\d+)", name).group(1))
            start = number * shard_rows
            end = start + weights.shape(name)[0]
            left, right = max(start, section.start), min(end, section.stop)
            if left < right:
                self.table[left - section.start : right - section.start].copy_(
                    weights.get(name, slice(left - start, right - start))
                )
        self.scale = weights.get(base + ".ngram_embedding.weight_scale", dtype=torch.bfloat16)
        self.key = weights.get(prefix + ".key_proj.weight")
        self.value = weights.get(prefix + ".value_proj.weight")
        self.key_norm = weights.get(prefix + ".norm_key.weight").float() + 1.0
        self.query_norm = weights.get(prefix + ".norm_query.weight").float() + 1.0
        self.conv_norm = weights.get(prefix + ".norm_conv.weight").float() + 1.0
        self.conv = weights.get(prefix + ".conv1d.weight")
        self.history = torch.zeros((1, 10240, 9), device=weights.device, dtype=torch.bfloat16)
        self.world = world

    def reset(self):
        self.history.zero_()

    def __call__(self, hidden, token, history):
        context = torch.cat([token, history])
        mixed = context * self.multipliers
        two = mixed[0] ^ mixed[1]
        three = two ^ mixed[2]
        ids = torch.cat([two.expand(8), three.expand(8)]) % self.sizes + self.offsets
        embedding = ops.table_lookup(self.table, ids, self.start).reshape(1, 2560)
        dist.all_reduce(embedding)
        embedding = embedding * self.scale
        key = ops.rms(
            F.linear(embedding, self.key), self.key_norm, group=2560, gemma=False
        ).reshape(1, 4, 2560)
        query = ops.rms(hidden, self.query_norm, group=2560, gemma=False).reshape(1, 4, 2560)
        gate = (key * query).sum(-1, keepdim=True) / (2560**0.5)
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        gated = (gate * F.linear(embedding, self.value).unsqueeze(1)).flatten(-2)
        normalized = ops.rms(gated, self.conv_norm, group=2560, gemma=False)
        window = torch.cat([self.history, normalized.unsqueeze(-1)], -1)
        conv = F.conv1d(window, self.conv, dilation=3, groups=10240).squeeze(-1)
        self.history.copy_(window[:, :, 1:])
        return gated + F.silu(conv)


class Layer:
    def __init__(
        self,
        weights,
        index,
        rank,
        world,
        config,
        capacity,
        execution_limit=None,
        moe_workspace=None,
        attention_tp=None,
        attention_rank=None,
        attention_group=None,
    ):
        prefix = f"model.language_model.layers.{index}"
        self.hc_attn = HyperConnection(weights, prefix + ".attn_hyper_connection")
        self.hc_mlp = HyperConnection(weights, prefix + ".mlp_hyper_connection")
        linear = config["layer_types"][index] == "linear_attention"
        attention_tp = world if attention_tp is None else attention_tp
        attention_rank = rank if attention_rank is None else attention_rank
        self.attention_group = attention_group
        self.attn = (
            GDN(weights, prefix + ".linear_attn", attention_rank, attention_tp)
            if linear
            else QSA(
                weights,
                prefix + ".self_attn",
                attention_rank,
                attention_tp,
                capacity,
                execution_limit,
            )
        )
        self.linear = linear
        self.ple = (
            PLE(weights, prefix + ".ple", rank, world)
            if index + 1 in config["ple_layer_ids"]
            else None
        )
        self.router = weights.get(prefix + ".mlp.gate.weight")
        self.experts = NVFP4Experts(
            weights, prefix + ".mlp", rank, world, config, workspace=moe_workspace
        )
        self.shared = None
        if rank == 0:
            self.shared = [
                weights.get(prefix + ".mlp." + name + ".weight")
                for name in (
                    "shared_expert.gate_proj",
                    "shared_expert.up_proj",
                    "shared_expert.down_proj",
                    "shared_expert_gate",
                )
            ]

    def reset(self):
        self.attn.reset()
        if self.ple:
            self.ple.reset()

    def __call__(self, hidden, token, history, position, cos, sin, rope_cos, rope_sin):
        if self.ple:
            hidden = hidden + self.ple(hidden, token, history)
        value, residual = self.hc_attn.mix(hidden)
        output = (
            self.attn(value)
            if self.linear
            else self.attn(value, position, cos, sin, rope_cos, rope_sin)
        )
        dist.all_reduce(output, group=self.attention_group)
        hidden = self.hc_attn.combine(output, residual)
        value, residual = self.hc_mlp.mix(hidden)
        probabilities = F.softmax(F.linear(value, self.router).float(), -1)
        probabilities, ids = probabilities.topk(10, -1)
        probabilities, ids = ops.normalize_topk(probabilities, ids)
        output = self.experts(value, ids, probabilities)
        if self.shared is not None:
            gate, up, down, shared_gate = self.shared
            shared = F.linear(F.silu(F.linear(value, gate)) * F.linear(value, up), down)
            output = output + shared * torch.sigmoid(F.linear(value, shared_gate))
        dist.all_reduce(output)
        return self.hc_mlp.combine(output, residual)


class Qwen38Runner:
    def __init__(self, model, rank, world, capacity=4096, *, execution_limit=None, attention_tp=8):
        attention_ranks, attention_rank = attention_layout(rank, world, attention_tp)
        if execution_limit is not None and (
            type(execution_limit) is not int or not 0 < execution_limit <= capacity
        ):
            raise ValueError("execution limit must be positive and within context capacity")
        config = json.loads((Path(model) / "config.json").read_text())
        self.config = validate_config(config, world, capacity)
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        self.rank, self.world, self.capacity = rank, world, capacity
        self.attention_tp = attention_tp
        self.attention_group = None
        if attention_tp < world:
            # Every rank creates the groups in the same order, including nonmembers.
            for start in range(0, world, attention_tp):
                members = list(range(start, start + attention_tp))
                group = dist.new_group(ranks=members, backend="nccl")
                if tuple(members) == attention_ranks:
                    self.attention_group = group
        weights = Weights(model, self.device)
        vocab = self.config["vocab_size"]
        rows = row_slice(vocab, rank, world)
        self.vocab_start = rows.start
        self.embedding = weights.get("model.language_model.embed_tokens.weight", rows)
        self.head = weights.get("lm_head.weight", rows)
        self.layers = []
        self.moe_workspace = None
        for index in range(48):
            self.layers.append(
                Layer(
                    weights,
                    index,
                    rank,
                    world,
                    self.config,
                    capacity,
                    execution_limit,
                    self.moe_workspace,
                    attention_tp,
                    attention_rank,
                    self.attention_group,
                )
            )
            # This runner executes one token on one stream; layers never overlap.
            self.moe_workspace = self.layers[-1].experts.workspace
            if rank == 0:
                print(f"loaded Qwen3.8 layer {index + 1}/48", flush=True)
        self.mixer = HyperConnection(weights, "model.language_model.hyper_connection_mixer")
        weights.close()
        angles = torch.arange(capacity, device=self.device).float()[:, None] * (
            10000000 ** (-torch.arange(0, 64, 2, device=self.device).float() / 64)
        )
        self.cos, self.sin = angles.cos(), angles.sin()
        self.position = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.history = torch.full((2,), 248044, device=self.device, dtype=torch.int64)
        self.token = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.logits_parts = [
            torch.empty((1, rows.stop - rows.start), device=self.device, dtype=torch.float32)
            for _ in range(world)
        ]
        self.candidates = torch.empty((world, 2), device=self.device, dtype=torch.float32)

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.position.zero_()
        self.history.fill_(248044)

    @torch.inference_mode()
    def _local_logits(self, token):
        embedding = ops.table_lookup(self.embedding, token, self.vocab_start).reshape(1, 2560)
        dist.all_reduce(embedding)
        hidden = embedding.repeat(1, 4)
        cos, sin = self.cos.index_select(0, self.position), self.sin.index_select(0, self.position)
        for layer in self.layers:
            hidden = layer(hidden, token, self.history, self.position, cos, sin, self.cos, self.sin)
        hidden, _ = self.mixer.mix(hidden)
        logits = F.linear(hidden, self.head).float()
        old = self.history.clone()
        self.history.copy_(
            torch.where(token == 248044, torch.full_like(old, 248044), torch.cat([token, old[:1]]))
        )
        self.position.add_(1)
        return logits

    @torch.inference_mode()
    def step(self, token):
        logits = self._local_logits(token)
        dist.all_gather(self.logits_parts, logits)
        return torch.cat(self.logits_parts, -1)

    @torch.inference_mode()
    def step_greedy(self, token):
        logits = self._local_logits(token)
        candidate = ops.local_greedy_candidate(logits, self.vocab_start)
        dist.all_gather_into_tensor(self.candidates.view(-1), candidate)
        return ops.greedy_from_candidates(self.candidates)
