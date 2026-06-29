"""
Realistic continuous batching scheduler for sglang-lite Phase 0.

Responsibilities:
- Accept new requests and immediately try Radix prefix match.
- Maintain waiting queue and running set.
- In each `step()` decide which sequences need prefill vs decode.
- Return a batch that the ModelRunner can execute together.
- Track per-sequence cached_len so the runner knows what is new.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from .kv_cache import RadixCache, PastKV


@dataclass
class Sequence:
    seq_id: int
    request_id: str

    # Full original prompt token ids
    input_ids: List[int]

    # Tokens generated so far
    output_ids: List[int] = field(default_factory=list)

    # How far into input_ids has been processed (including prefix cache hit)
    cached_len: int = 0

    # Current KV state for this sequence (owned or forked from radix)
    kv_state: Optional[PastKV] = None

    created_ts: float = field(default_factory=time.time)
    last_token_ts: float = field(default_factory=time.time)

    finished: bool = False
    finish_reason: str = ""

    # Simple stats for this sequence
    prefill_tokens: int = 0
    decode_tokens: int = 0


class Scheduler:
    """
    Continuous batching scheduler with Radix awareness.

    In each step we:
    1. Admit as many waiting sequences as capacity allows (they may have prefix hits).
    2. All admitted sequences that still have unprocessed prompt do prefill.
    3. Sequences that have finished prompt do decode (one token).
    4. The runner is responsible for updating the KV on the sequence.
    """

    def __init__(
        self,
        radix_cache: RadixCache,
        max_batch_size: int = 8,          # smaller on CPU
        max_tokens_per_batch: int = 512,
        max_waiting: int = 128,
    ):
        self.radix = radix_cache
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_waiting = max_waiting

        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self._next_seq_id = 1

    def add_request(self, request_id: str, input_ids: List[int]) -> Sequence:
        if len(self.waiting) >= self.max_waiting:
            # Simple queue limit: evict oldest waiting if possible
            if self.waiting:
                old = self.waiting.popleft()
                self.radix.release_blocks(getattr(old, 'allocated_blocks', []))
            else:
                raise RuntimeError("queue full and no eviction possible")

        seq = Sequence(
            seq_id=self._next_seq_id,
            request_id=request_id,
            input_ids=list(input_ids),
        )
        self._next_seq_id += 1

        # Radix prefix match — this is where we win on multi-turn
        matched, matched_len, remaining, cached_kv = self.radix.match_prefix(input_ids)

        seq.cached_len = matched_len
        if cached_kv is not None:
            seq.kv_state = self.radix.fork_kv(cached_kv)

        # Record the prefix in the tree (even if we haven't computed new KV yet)
        if matched:
            self.radix.insert_or_update(matched, seq.kv_state or [], matched_len)

        self.waiting.append(seq)
        return seq

    def step(self) -> Tuple[List[Sequence], List[bool]]:
        """
        Returns:
            batch: list of sequences to run this step
            is_prefill: parallel list, True if this sequence still needs prefill
        """
        batch: List[Sequence] = []
        is_prefill: List[bool] = []

        # 1. Admit new work from waiting (prefer those with cache hits if possible)
        while self.waiting and len(batch) < self.max_batch_size:
            seq = self.waiting.popleft()
            batch.append(seq)
            needs_prefill = seq.cached_len < len(seq.input_ids)
            is_prefill.append(needs_prefill)

        # 2. Keep running decode sequences (they have finished their prompt)
        # Promote decodes preferentially for continuous batching
        decodes = [s for s in self.running if not s.finished and s.cached_len >= len(s.input_ids)]
        for seq in decodes:
            if len(batch) < self.max_batch_size:
                batch.append(seq)
                is_prefill.append(False)

        # Fill remaining with other running if any
        for seq in list(self.running):
            if not seq.finished and len(batch) < self.max_batch_size and seq not in batch:
                batch.append(seq)
                is_prefill.append(False)

        # Simple cap
        if len(batch) > self.max_batch_size:
            excess = batch[self.max_batch_size :]
            batch = batch[: self.max_batch_size]
            is_prefill = is_prefill[: self.max_batch_size]
            for s in excess:
                if not s.finished and s not in self.running:
                    self.waiting.appendleft(s)

        self.running = [s for s, p in zip(batch, is_prefill) if not s.finished]
        return batch, is_prefill

    def update_after_prefill(
        self,
        seq: Sequence,
        new_tokens: List[int],
        new_kv_state: PastKV,
    ) -> None:
        """Called by runner after a prefill step."""
        seq.input_ids.extend(new_tokens)  # usually not needed, but for clarity
        seq.cached_len = len(seq.input_ids)
        seq.kv_state = new_kv_state
        seq.prefill_tokens += len(new_tokens)

        # Commit this prefix (including the newly computed part) into the radix tree
        full_prompt = seq.input_ids[: seq.cached_len]
        self.radix.insert_or_update(full_prompt, new_kv_state, seq.cached_len)

    def update_after_decode(
        self, seq: Sequence, new_token: int, new_kv_state: PastKV
    ) -> None:
        """Called by runner after decoding one token."""
        seq.output_ids.append(new_token)
        seq.cached_len += 1
        seq.kv_state = new_kv_state
        seq.decode_tokens += 1
        seq.last_token_ts = time.time()

    def mark_finished(self, seq: Sequence, reason: str = "stop") -> None:
        seq.finished = True
        seq.finish_reason = reason
        if seq in self.running:
            self.running.remove(seq)

        # Release any block tracking (future paged use)
        self.radix.release_blocks(getattr(seq, "allocated_blocks", []))
        if hasattr(seq, "allocated_blocks"):
            seq.allocated_blocks = []
