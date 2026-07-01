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
    """Lightweight sequence descriptor. Lifecycle mostly managed by the driver (unigateway)."""
    seq_id: int
    request_id: str
    input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    cached_len: int = 0
    kv_state: Optional[PastKV] = None
    created_ts: float = field(default_factory=time.time)
    last_token_ts: float = field(default_factory=time.time)
    finished: bool = False
    finish_reason: str = ""
    prefill_tokens: int = 0
    decode_tokens: int = 0


class SequenceTable:
    """Manages active sequences. Can be driven by unigateway."""
    def __init__(self):
        self.sequences: Dict[int, Sequence] = {}
        self._next_id = 1

    def create(self, request_id: str, input_ids: List[int]) -> Sequence:
        seq = Sequence(seq_id=self._next_id, request_id=request_id, input_ids=list(input_ids))
        self._next_id += 1
        self.sequences[seq.seq_id] = seq
        return seq

    def get(self, seq_id: int) -> Optional[Sequence]:
        return self.sequences.get(seq_id)

    def remove(self, seq_id: int):
        self.sequences.pop(seq_id, None)


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
        self.batch_former = BatchFormer(max_batch_size, max_tokens_per_batch)
        # Note: max_waiting / admission can be handled by unigateway driver

    def add_request(self, request_id: str, input_ids: List[int]) -> Sequence:
        # Admission logic (max_waiting, eviction on queue) is typically handled by unigateway.
        # The scheduler here assumes the request has already been admitted.

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
        Form next batch using BatchFormer (which unigateway can replace).
        """
        batch, is_prefill = self.batch_former.form_batch(self.waiting, self.running)

        # Update running list
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
