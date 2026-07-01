"""
MoEModelRunner composed of smaller pieces for composability.

High-level components (unigateway can compose or replace):
- ModelLoader
- MoERouter
- PrefillExecutor / DecodeExecutor
- KernelBackend

sglang-lite only provides default implementations focused on popular MoE patterns.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .kv_cache import RadixCache, PastKV
from .scheduler import Sequence


class MoERouter:
    """Default MoE router. Can be replaced for custom routing strategies."""
    def route(self, input_ids: List[int]) -> List[int]:
        # Placeholder: real implementation would use the model's router weights.
        # For now return dummy expert ids.
        return [0] * len(input_ids)


class ModelRunner:  # kept for backward compat in examples; prefer MoEModelRunner
    def __init__(self, model_name: str = "stub", device: str = "cpu", max_batch: int = 4):
        self.model_name = model_name
        self.device = device
        self.max_batch = max_batch

        self.model = None
        self.tokenizer = None
        self._is_real = False
        self.vocab_size = 32000  # fallback

        if model_name and model_name != "stub":
            self._try_load_real(model_name)
        else:
            self._init_tiny_stub_model()

    def _try_load_real(self, model_name: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[sglang-lite] Attempting to load real model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=True
            )
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="cpu" if self.device == "cpu" else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            self._is_real = True
            self.vocab_size = self.model.config.vocab_size
            print(f"[sglang-lite] Real model '{model_name}' loaded on {self.device}.")
        except Exception as e:
            print(f"[sglang-lite] Failed to load real model ({e}). Using tiny stub model.")
            self._is_real = False
            self._init_tiny_stub_model()

    def _init_tiny_stub_model(self):
        """A tiny causal LM for demos when no real model is available."""
        import torch.nn as nn

        class TinyLM(nn.Module):
            def __init__(self, vocab=32000, dim=256, layers=4, heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.layers = nn.ModuleList(
                    [nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True) for _ in range(layers)]
                )
                self.ln = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.vocab_size = vocab

            def forward(self, input_ids, past_key_values=None, use_cache=False):
                x = self.embed(input_ids)
                # Very simplified: we ignore real past KV for the stub transformer
                # and just run on the last few tokens. Good enough for demo.
                for layer in self.layers:
                    x = layer(x)
                x = self.ln(x)
                logits = self.head(x)
                # Return dummy past for interface compatibility
                dummy_past = None
                if use_cache:
                    dummy_past = [(x[:, -1:, :].clone(), x[:, -1:, :].clone()) for _ in range(len(self.layers))]
                return type("Obj", (), {"logits": logits, "past_key_values": dummy_past})()

        self.model = TinyLM(vocab=self.vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False
        print("[sglang-lite] Using built-in tiny stub model for CPU demo.")

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        # Fallback for stub
        return [hash(c) % self.vocab_size for c in text[:128]]

    def detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join([chr(97 + (t % 26)) for t in token_ids[:20]])

    @torch.no_grad()
    def run_batch(
        self,
        batch: List[Sequence],
        radix: RadixCache,
        is_prefill: List[bool],
    ) -> List[Optional[int]]:
        """
        Execute one step for the batch.

        Returns next token id for each sequence (or None if finished).
        Updates seq.kv_state and seq.cached_len via scheduler hooks.
        """
        if not batch:
            return []

        results: List[Optional[int]] = []

        for i, seq in enumerate(batch):
            if seq.finished:
                results.append(None)
                continue

            do_prefill = is_prefill[i]

            if do_prefill:
                next_token = self._prefill(seq, radix)
            else:
                next_token = self._decode_one(seq, radix)

            results.append(next_token)

        return results

    def _prefill(self, seq: Sequence, radix: RadixCache) -> int:
        """Run prefill for the unprocessed part of the prompt."""
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]

        if not new_tokens:
            # Already fully cached, treat as decode
            return self._decode_one(seq, radix)

        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)

        if self._is_real and hasattr(self.model, "forward"):
            # Real transformers path
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=seq.kv_state,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            new_kv = outputs.past_key_values
        else:
            # Stub model path
            outputs = self.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            new_kv = outputs.past_key_values

        # Sample (greedy for simplicity and determinism in demo)
        next_id = int(torch.argmax(logits).item())

        # Merge the new KV into the existing prefix KV
        if seq.kv_state is not None and new_kv is not None:
            merged_kv = self._merge_kv(seq.kv_state, new_kv)
        else:
            merged_kv = new_kv

        # Let scheduler know
        # (we update here because scheduler may be called from outside)
        seq.kv_state = merged_kv
        seq.cached_len = len(prompt)

        return next_id

    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        """Decode exactly one new token using cached KV."""
        # We only need the last generated token as input
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)

        if self._is_real and hasattr(self.model, "forward"):
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=seq.kv_state,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            new_kv = outputs.past_key_values
        else:
            outputs = self.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            new_kv = outputs.past_key_values

        next_id = int(torch.argmax(logits).item())

        # Update KV (append the new step)
        if seq.kv_state is not None and new_kv is not None:
            seq.kv_state = self._merge_kv(seq.kv_state, new_kv)
        else:
            seq.kv_state = new_kv

        return next_id

    def _merge_kv(self, old: PastKV, new_step: PastKV) -> PastKV:
        """Concatenate old KV with the new token's KV along sequence dimension."""
        if old is None:
            return new_step
        if new_step is None:
            return old

        merged = []
        for (old_k, old_v), (new_k, new_v) in zip(old, new_step):
            merged.append((
                torch.cat([old_k, new_k], dim=2),
                torch.cat([old_v, new_v], dim=2),
            ))
        return merged

    def sample_next(self, logits: torch.Tensor, temperature: float = 0.7) -> int:
        """Simple sampling (greedy is used by default in runner for demo)."""
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
