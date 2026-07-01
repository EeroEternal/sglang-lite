"""
RadixAttention-style KV Cache with actual prefix KV state management.

This is a core component. We implement prefix sharing here so that
multi-turn / agent workloads can reuse pre-computed KV for common prefixes.

For Phase 0 (CPU friendly):
- We store KV states directly on the tree nodes (list of (key, value) per layer).
- On prefix match we can reuse the stored KV up to the match point.
- When forking a new branch we copy the prefix KV (simple but correct).
- No full paged GPU blocks yet — we will evolve to paged later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class KVBlock:
    """Simple abstraction for a memory block (used for stats & future paged impl)."""
    block_id: int
    ref_count: int = 0


# Type for past_key_values used by transformers and our runner:
# List[Tuple[key, value]] where each is (batch, heads, seq, head_dim)
PastKV = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class RadixNode:
    """Node in the radix tree over token ids (pure tree structure)."""
    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0

    # The KV state for the *prefix ending at this node*.
    # None means this prefix has not been computed yet.
    kv_state: Optional[PastKV] = None

    # Rough memory usage estimate (number of tokens this prefix represents)
    prefix_len: int = 0


class RadixTree:
    """Pure token radix tree. KV concerns are handled by the caller (RadixCache)."""

    def __init__(self):
        self.root = RadixNode()

    def walk(self, token_ids: List[int]) -> Tuple[RadixNode, List[int], int]:
        """Walk the tree as far as possible. Returns (final_node, matched, matched_len)."""
        node = self.root
        matched: List[int] = []
        i = 0
        for tid in token_ids:
            if tid in node.children:
                node = node.children[tid]
                matched.append(tid)
                i += 1
            else:
                break
        return node, matched, i

    def insert_path(self, token_ids: List[int]) -> RadixNode:
        """Create nodes for the path if missing. Returns the leaf node."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = RadixNode(tokens=[tid])
            node = node.children[tid]
            node.ref_count += 1
        return node


class RadixCache:
    """
    Radix cache with KV state sharing.

    Key operations:
    - match_prefix: find longest matching prefix and return cached KV if available.
    - commit: after computing new tokens, store/update the KV for the path.
    - fork / share: when a new sequence branches from an existing prefix.
    """

    def __init__(self, max_tokens: int = 65536):
        self.max_tokens = max_tokens
        self.tree = RadixTree()
        self._allocated_blocks: Dict[int, KVBlock] = {}
        self._next_block_id = 0

        # Simple stats
        self.total_tokens_stored = 0
        self.hit_count = 0
        self.miss_count = 0

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[List[int], int, List[int], Optional[PastKV]]:
        """
        Find longest prefix match.

        Returns:
            matched_tokens, matched_len, remaining_tokens, cached_kv (or None)
        """
        node, matched, i = self.tree.walk(token_ids)
        cached_kv = node.kv_state if node.kv_state is not None else None

        remaining = token_ids[i:]
        if cached_kv is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1

        return matched, i, remaining, cached_kv

    def insert_or_update(
        self,
        token_ids: List[int],
        kv_state: PastKV,
        prefix_len: int,
    ) -> None:
        """
        Walk/create the path and store KV state at the end node.
        """
        node = self.tree.insert_path(token_ids)

        # Store KV on the leaf and propagate for prefix matching
        if kv_state is not None:
            node.kv_state = kv_state
            node.prefix_len = prefix_len + len(token_ids)

        # Rough accounting
        self.total_tokens_stored = max(self.total_tokens_stored, node.prefix_len)

    def fork_kv(self, src_kv: Optional[PastKV]) -> Optional[PastKV]:
        """
        When we hit a prefix, we usually need to copy the KV tensors
        because the new sequence will continue to append to it.
        Shallow copy of structure + clone tensors (CPU friendly).
        """
        if src_kv is None:
            return None
        return [(k.clone(), v.clone()) for k, v in src_kv]

    def get_cache_stats(self) -> Dict:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total)
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_tokens_stored": self.total_tokens_stored,
        }

    # --- The following are kept for future paged attention evolution ---
    def allocate_blocks(self, count: int) -> List[int]:
        ids = []
        for _ in range(count):
            bid = self._next_block_id
            self._next_block_id += 1
            self._allocated_blocks[bid] = KVBlock(block_id=bid, ref_count=1)
            ids.append(bid)
        return ids

    def release_blocks(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            blk = self._allocated_blocks.get(bid)
            if blk:
                blk.ref_count -= 1
                if blk.ref_count <= 0:
                    self._allocated_blocks.pop(bid, None)

    def evict(self, needed: int) -> int:
        # Placeholder — real version would walk tree by recency/refcount
        return min(needed, 16)
