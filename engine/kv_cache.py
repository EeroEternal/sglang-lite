"""RadixAttention-style KV cache with paged blocks, COW, and eviction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0


PastKV = List[Tuple[torch.Tensor, torch.Tensor]]


class KvLayoutKind(str, Enum):
    """Per-pool KV physical layout (Radix owns lifecycle; kernels consume pages)."""

    MHA = "mha"  # standard K/V: [blocks, page, kv_heads, head_dim]
    MLA_COMPRESSED = "mla_compressed"  # ckv + kpe pages
    SWA = "swa"  # sliding-window pool (V4)
    DSV4_PACKED = "dsv4_packed"  # FlashInfer SM120 packed uint8 last-dim 584


@dataclass(frozen=True)
class KvLayout:
    """Layout descriptor for one KV pool.

    Standard MHA uses a single pool. V4 sparse MLA needs SWA + compressed
    (and optionally packed) pools selected per layer via ``compress_ratios``.
    """

    kind: KvLayoutKind = KvLayoutKind.MHA
    num_kv_heads: int = 8
    head_dim: int = 128
    # MLA: latent ckv dim and rope dim (DeepSeek V2/V3/V4 style).
    ckv_dim: int = 0
    kpe_dim: int = 0
    # FlashInfer DSV4 packed page last-dim (SM120 path uses 584).
    packed_last_dim: int = 0
    dtype_name: str = "bfloat16"

    @staticmethod
    def mha(num_kv_heads: int, head_dim: int, dtype_name: str = "bfloat16") -> "KvLayout":
        return KvLayout(
            kind=KvLayoutKind.MHA,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_name=dtype_name,
        )

    @staticmethod
    def mla_compressed(
        ckv_dim: int = 512,
        kpe_dim: int = 64,
        dtype_name: str = "bfloat16",
    ) -> "KvLayout":
        return KvLayout(
            kind=KvLayoutKind.MLA_COMPRESSED,
            num_kv_heads=1,
            head_dim=ckv_dim,
            ckv_dim=ckv_dim,
            kpe_dim=kpe_dim,
            dtype_name=dtype_name,
        )

    @staticmethod
    def dsv4_packed(packed_last_dim: int = 584) -> "KvLayout":
        return KvLayout(
            kind=KvLayoutKind.DSV4_PACKED,
            num_kv_heads=1,
            head_dim=512,
            ckv_dim=512,
            kpe_dim=64,
            packed_last_dim=packed_last_dim,
            dtype_name="uint8",
        )


@dataclass
class RadixNode:
    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0
    block_ids: List[int] = field(default_factory=list)
    prefix_len: int = 0
    # Optional HF snapshot (legacy). Prefer rebuilding from paged tensors + last_logits.
    past_kv: Optional[PastKV] = None
    # Logits at the last prompt position — required for exact-prefix hits (no re-forward).
    last_logits: Optional[torch.Tensor] = None


class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def walk(self, token_ids: List[int]) -> Tuple[RadixNode, List[int], int]:
        node = self.root
        matched: List[int] = []
        for tid in token_ids:
            if tid in node.children:
                node = node.children[tid]
                matched.append(tid)
            else:
                break
        return node, matched, len(matched)

    def insert_path(self, token_ids: List[int]) -> RadixNode:
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = RadixNode(tokens=[tid])
            node = node.children[tid]
            node.ref_count += 1
        return node

    def find_node(self, token_ids: List[int]) -> Optional[RadixNode]:
        node, _, n = self.walk(token_ids)
        if n == len(token_ids) and token_ids:
            return node
        if not token_ids:
            return self.root
        return None


def cache_is_hybrid(past) -> bool:
    """True when the HF cache holds linear-attention state, not only K/V.

    Qwen3.5/3.6 Gated DeltaNet layers keep conv and recurrent state on
    ``LinearAttentionLayer``. Flattening those layers to ``(K, V)`` drops the
    state and the next forward rejects the cache.
    """
    layers = getattr(past, "layers", None)
    if not layers:
        return False
    try:
        from transformers.cache_utils import LinearAttentionCacheLayerMixin
    except Exception:
        return False
    return any(isinstance(layer, LinearAttentionCacheLayerMixin) for layer in layers)


class RadixCache:
    """Paged KV + radix prefix index with refcounted blocks and COW forks."""

    def __init__(
        self,
        max_tokens: int = 65536,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        layout: Optional[KvLayout] = None,
        # Optional secondary pools for V4 SWA / packed pages (allocated lazily).
        swa_layout: Optional[KvLayout] = None,
        compressed_layout: Optional[KvLayout] = None,
    ):
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.layout = layout or KvLayout.mha(num_kv_heads, head_dim)
        self.swa_layout = swa_layout
        self.compressed_layout = compressed_layout

        self.tree = RadixTree()
        self._allocated_blocks: Dict[int, KVBlock] = {}
        self._free_blocks: List[int] = []
        self._next_block_id = 0

        self.num_blocks = max(1, (max_tokens + block_size - 1) // block_size)
        self.k_cache = torch.zeros(
            (num_layers, self.num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, self.num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        # Optional MLA latent pages: [layers, blocks, page, ckv_dim] / kpe
        self.ckv_cache: Optional[torch.Tensor] = None
        self.kpe_cache: Optional[torch.Tensor] = None
        if self.layout.kind == KvLayoutKind.MLA_COMPRESSED or compressed_layout is not None:
            cl = compressed_layout or self.layout
            self.ckv_cache = torch.zeros(
                (num_layers, self.num_blocks, block_size, cl.ckv_dim),
                dtype=dtype,
                device=device,
            )
            self.kpe_cache = torch.zeros(
                (num_layers, self.num_blocks, block_size, cl.kpe_dim),
                dtype=dtype,
                device=device,
            )
        # Dual packed pools for V4 Phase 0c (SWA + compressed), uint8 last-dim 584.
        # ``packed_kv_cache`` remains an alias of the SWA pool for older callers.
        self.packed_swa_cache: Optional[torch.Tensor] = None
        self.packed_comp_cache: Optional[torch.Tensor] = None
        self.packed_kv_cache: Optional[torch.Tensor] = None
        need_swa_packed = self.layout.kind == KvLayoutKind.DSV4_PACKED or (
            swa_layout is not None and swa_layout.kind == KvLayoutKind.DSV4_PACKED
        )
        need_comp_packed = (
            compressed_layout is not None
            and compressed_layout.kind == KvLayoutKind.DSV4_PACKED
        )
        if need_swa_packed:
            pl = swa_layout if swa_layout and swa_layout.packed_last_dim else self.layout
            last = pl.packed_last_dim or 584
            self.packed_swa_cache = torch.zeros(
                (num_layers, self.num_blocks, 1, block_size, last),
                dtype=torch.uint8,
                device=device,
            )
            self.packed_kv_cache = self.packed_swa_cache
        # Phase 0c: pair a compressed packed pool whenever SWA packed exists
        # (or when an explicit compressed DSV4 layout is requested).
        if need_comp_packed or need_swa_packed:
            cl = (
                compressed_layout
                if compressed_layout and compressed_layout.packed_last_dim
                else (
                    swa_layout
                    if swa_layout and swa_layout.packed_last_dim
                    else self.layout
                )
            )
            last_c = cl.packed_last_dim or 584
            self.packed_comp_cache = torch.zeros(
                (num_layers, self.num_blocks, 1, block_size, last_c),
                dtype=torch.uint8,
                device=device,
            )
        # Phase 0c-3: bf16 restore pool [layers, blocks, page, 512] for Hybrid
        # page→official buffer restore (packed 584 is lossy / FI-only).
        self.restore_bf16_cache: Optional[torch.Tensor] = None
        if need_swa_packed or need_comp_packed:
            self.restore_bf16_cache = torch.zeros(
                (num_layers, self.num_blocks, block_size, 512),
                dtype=dtype if dtype != torch.uint8 else torch.bfloat16,
                device=device,
            )
        # Token occupancy per physical page (for append position)
        self._page_len: Dict[int, int] = {}

        self.total_tokens_stored = 0
        self.hit_count = 0
        self.miss_count = 0
        self.oom_reject_count = 0
        self.evict_count = 0
        # Phase 0c dual-write / dual-hit observability
        self.dual_write_count = 0
        self.dual_write_tokens = 0
        self.dual_hit_count = 0
        self.dual_append_count = 0
        self.dual_restore_count = 0
        # Phase 0c-4: pre-forward stage of official buffers from pages.
        self.dual_stage_count = 0

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[List[int], int, List[int], List[int], Optional[PastKV], Optional[torch.Tensor]]:
        """Return matched tokens/len/remaining/block_ids/past_kv/last_logits.

        A reusable hit requires paged block_ids covering the matched prefix.
        past_kv is optional; runners rebuild attention state from pages.
        Exact hits also need last_logits for the first sample.
        """
        node, matched, i = self.tree.walk(token_ids)
        remaining = token_ids[i:]
        block_ids = list(node.block_ids) if node.block_ids else []
        past_kv = self.fork_kv(node.past_kv) if node.past_kv is not None else None
        last_logits = node.last_logits.clone() if node.last_logits is not None else None

        # Walk upward for nearest ancestor with committed pages if leaf has none
        if (not block_ids) and i > 0:
            cur = self.tree.root
            best_blocks: List[int] = []
            best_logits = None
            best_kv = None
            best_len = 0
            for j, tid in enumerate(matched):
                cur = cur.children.get(tid)
                if cur is None:
                    break
                if cur.block_ids:
                    best_blocks = list(cur.block_ids)
                    best_logits = cur.last_logits
                    best_kv = cur.past_kv
                    best_len = j + 1
            if best_blocks and best_len == i:
                block_ids = best_blocks
                last_logits = best_logits.clone() if best_logits is not None else None
                past_kv = self.fork_kv(best_kv) if best_kv is not None else None
                matched = matched[:best_len]
                i = best_len
                remaining = token_ids[i:]

        pages_needed = (i + self.block_size - 1) // self.block_size if i > 0 else 0
        reusable = i > 0 and len(block_ids) >= pages_needed
        if remaining == [] and reusable and last_logits is None and past_kv is None:
            # Exact hit without logits cannot sample correctly
            reusable = False

        if reusable:
            self.hit_count += 1
            block_ids = block_ids[: max(pages_needed, 1)] if pages_needed else block_ids
        else:
            self.miss_count += 1
            return [], 0, token_ids, [], None, None

        for bid in block_ids:
            blk = self._allocated_blocks.get(bid)
            if blk:
                blk.ref_count += 1

        return matched, i, remaining, block_ids, past_kv, last_logits

    def insert_or_update(
        self,
        token_ids: List[int],
        kv_tensors: Optional[PastKV] = None,
        prefix_len: int = 0,
        block_ids: Optional[List[int]] = None,
        last_logits: Optional[torch.Tensor] = None,
    ) -> List[int]:
        if not token_ids:
            return list(block_ids or [])
        node = self.tree.insert_path(token_ids)
        if block_ids is not None:
            # Tree retains its own refs so finished sequences can release without dropping prefix KV.
            new_list = list(block_ids)
            if node.block_ids != new_list:
                if node.block_ids:
                    self.release_blocks(node.block_ids)
                node.block_ids = self.fork_blocks(new_list)
        if kv_tensors is not None:
            node.past_kv = self.fork_kv(kv_tensors)
        if last_logits is not None:
            node.last_logits = last_logits.detach().float().cpu().clone()
        node.prefix_len = prefix_len or len(token_ids)
        self.total_tokens_stored = max(self.total_tokens_stored, node.prefix_len)
        return list(node.block_ids)

    def write_packed_kv(
        self,
        block_table: List[int],
        start_pos: int,
        packed_tokens: torch.Tensor,
        *,
        pool: str = "swa",
        layer_idx: int = 0,
    ) -> None:
        """Write packed uint8 tokens ``[T, packed_last_dim]`` into a dual pool.

        ``pool`` is ``\"swa\"`` (default) or ``\"comp\"`` (compressed).
        Page layout is ``[layers, blocks, 1, page, packed_dim]``.
        """
        cache = self._packed_pool(pool)
        if cache is None:
            raise RuntimeError(f"packed pool {pool!r} is not allocated on this RadixCache")
        if not block_table:
            raise ValueError("write_packed_kv requires block_table")
        if packed_tokens.dim() != 2:
            raise ValueError(f"expected [T, D] packed tokens, got {tuple(packed_tokens.shape)}")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range ({self.num_layers})")
        n = int(packed_tokens.shape[0])
        if n == 0:
            return
        last = int(cache.shape[-1])
        if int(packed_tokens.shape[-1]) != last:
            raise ValueError(
                f"packed last dim {packed_tokens.shape[-1]} != pool dim {last}"
            )
        end_pos = start_pos + n
        pages_needed = (end_pos + self.block_size - 1) // self.block_size
        if pages_needed > len(block_table):
            raise RuntimeError(
                f"block_table too short for packed pos={end_pos - 1}: pages={len(block_table)}"
            )
        src = packed_tokens.to(device=self.device, dtype=torch.uint8).contiguous()
        t = 0
        while t < n:
            pos = start_pos + t
            page_i = pos // self.block_size
            slot = pos % self.block_size
            span = min(self.block_size - slot, n - t)
            bid = block_table[page_i]
            if bid >= self.num_blocks:
                raise RuntimeError(f"block id {bid} out of range ({self.num_blocks})")
            # cache[layer, bid, 0, slot:slot+span, :]
            cache[layer_idx, bid, 0, slot : slot + span].copy_(src[t : t + span])
            self._page_len[bid] = max(self._page_len.get(bid, 0), slot + span)
            t += span

    def read_packed_kv(
        self,
        block_table: List[int],
        length: int,
        *,
        pool: str = "swa",
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Read ``length`` packed tokens from a dual pool as ``[T, packed_dim]`` uint8."""
        cache = self._packed_pool(pool)
        if cache is None:
            raise RuntimeError(f"packed pool {pool!r} is not allocated on this RadixCache")
        if length <= 0:
            return torch.zeros(0, cache.shape[-1], dtype=torch.uint8, device=self.device)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range ({self.num_layers})")
        pages_needed = (length + self.block_size - 1) // self.block_size
        if len(block_table) < pages_needed:
            raise RuntimeError(
                f"read_packed_kv: need {pages_needed} pages, have {len(block_table)}"
            )
        last = int(cache.shape[-1])
        out = torch.empty(length, last, dtype=torch.uint8, device=self.device)
        t = 0
        while t < length:
            pos = t
            page_i = pos // self.block_size
            slot = pos % self.block_size
            span = min(self.block_size - slot, length - t)
            bid = block_table[page_i]
            out[t : t + span].copy_(cache[layer_idx, bid, 0, slot : slot + span])
            t += span
        return out

    def _packed_pool(self, pool: str) -> Optional[torch.Tensor]:
        name = (pool or "swa").lower()
        if name in ("swa", "window", "packed"):
            return self.packed_swa_cache if self.packed_swa_cache is not None else self.packed_kv_cache
        if name in ("comp", "compressed", "c"):
            return self.packed_comp_cache
        raise ValueError(f"unknown packed pool {pool!r}; use 'swa' or 'comp'")

    def write_restore_bf16(
        self,
        block_table: List[int],
        start_pos: int,
        tokens: torch.Tensor,
        *,
        layer_idx: int = 0,
    ) -> None:
        """Write bf16 ``[T, 512]`` into the Hybrid restore pool."""
        if self.restore_bf16_cache is None:
            raise RuntimeError("restore_bf16_cache is not allocated")
        if not block_table:
            raise ValueError("write_restore_bf16 requires block_table")
        if tokens.dim() != 2 or tokens.shape[-1] != 512:
            raise ValueError(f"expected [T, 512], got {tuple(tokens.shape)}")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range")
        n = int(tokens.shape[0])
        if n == 0:
            return
        end_pos = start_pos + n
        pages_needed = (end_pos + self.block_size - 1) // self.block_size
        if pages_needed > len(block_table):
            raise RuntimeError(
                f"block_table too short for restore bf16 pos={end_pos - 1}"
            )
        src = tokens.to(device=self.device, dtype=self.restore_bf16_cache.dtype).contiguous()
        t = 0
        while t < n:
            pos = start_pos + t
            page_i = pos // self.block_size
            slot = pos % self.block_size
            span = min(self.block_size - slot, n - t)
            bid = block_table[page_i]
            self.restore_bf16_cache[layer_idx, bid, slot : slot + span].copy_(
                src[t : t + span]
            )
            self._page_len[bid] = max(self._page_len.get(bid, 0), slot + span)
            t += span

    def read_restore_bf16(
        self,
        block_table: List[int],
        length: int,
        *,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Read ``length`` tokens from the Hybrid restore pool as ``[T, 512]``."""
        if self.restore_bf16_cache is None:
            raise RuntimeError("restore_bf16_cache is not allocated")
        if length <= 0:
            return torch.zeros(
                0, 512, dtype=self.restore_bf16_cache.dtype, device=self.device
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range")
        pages_needed = (length + self.block_size - 1) // self.block_size
        if len(block_table) < pages_needed:
            raise RuntimeError(
                f"read_restore_bf16: need {pages_needed} pages, have {len(block_table)}"
            )
        out = torch.empty(
            length, 512, dtype=self.restore_bf16_cache.dtype, device=self.device
        )
        t = 0
        while t < length:
            pos = t
            page_i = pos // self.block_size
            slot = pos % self.block_size
            span = min(self.block_size - slot, length - t)
            bid = block_table[page_i]
            out[t : t + span].copy_(
                self.restore_bf16_cache[layer_idx, bid, slot : slot + span]
            )
            t += span
        return out

    def write_kv(
        self,
        block_table: List[int],
        start_pos: int,
        layer_kvs: PastKV,
    ) -> None:
        """Write per-layer K/V for tokens starting at start_pos into pages."""
        if not layer_kvs or not block_table:
            raise ValueError("write_kv requires layer_kvs and block_table")
        for layer_idx, (k, v) in enumerate(layer_kvs):
            if layer_idx >= self.num_layers:
                break
            self.write_kv_layer(block_table, start_pos, layer_idx, k, v)

    def write_kv_layer(
        self,
        block_table: List[int],
        start_pos: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a single layer's K/V for tokens starting at start_pos into pages."""
        if not block_table:
            raise ValueError("write_kv_layer requires block_table")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range ({self.num_layers})")
        k_tok, v_tok = self.normalize_kv(k, v)
        n = k_tok.shape[0]
        end_pos = start_pos + n
        pages_needed = (end_pos + self.block_size - 1) // self.block_size
        if pages_needed > len(block_table):
            raise RuntimeError(
                f"block_table too short for pos={end_pos - 1}: pages={len(block_table)}"
            )
        k_tok = k_tok.to(dtype=self.dtype)
        v_tok = v_tok.to(dtype=self.dtype)
        t = 0
        while t < n:
            pos = start_pos + t
            page_i = pos // self.block_size
            slot = pos % self.block_size
            span = min(self.block_size - slot, n - t)
            bid = block_table[page_i]
            if bid >= self.num_blocks:
                raise RuntimeError(f"block id {bid} out of range ({self.num_blocks})")
            self.k_cache[layer_idx, bid, slot : slot + span].copy_(k_tok[t : t + span])
            self.v_cache[layer_idx, bid, slot : slot + span].copy_(v_tok[t : t + span])
            self._page_len[bid] = max(self._page_len.get(bid, 0), slot + span)
            t += span

    def read_kv(self, block_table: List[int], length: int) -> PastKV:
        """Read length tokens from pages as HF legacy list[(k,v)] with shape (1,H,S,D)."""
        if length <= 0:
            return []
        pages_needed = (length + self.block_size - 1) // self.block_size
        if len(block_table) < pages_needed:
            raise RuntimeError(
                f"read_kv: need {pages_needed} pages, have {len(block_table)} for length={length}"
            )
        pages = torch.tensor(
            block_table[:pages_needed], dtype=torch.long, device=self.device
        )
        out: PastKV = []
        for layer_idx in range(self.num_layers):
            # (pages, block_size, H, D) -> (1, H, length, D)
            k = (
                self.k_cache[layer_idx]
                .index_select(0, pages)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:length]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            v = (
                self.v_cache[layer_idx]
                .index_select(0, pages)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:length]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            out.append((k, v))
        return out

    def build_cache(self, block_table: List[int], length: int):
        """Rebuild a transformers DynamicCache (or legacy list) from paged KV."""
        legacy = self.read_kv(block_table, length)
        if not legacy:
            return None
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:
            return legacy

    def normalize_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (seq, num_kv_heads, head_dim)."""
        if k.dim() == 4:
            # (bs, heads, seq, dim) or (bs, seq, heads, dim)
            if k.shape[1] == self.num_kv_heads:
                k = k[0].transpose(0, 1).contiguous()
                v = v[0].transpose(0, 1).contiguous()
            else:
                k = k[0].contiguous()
                v = v[0].contiguous()
        elif k.dim() == 3:
            # (seq, heads, dim) or (heads, seq, dim)
            if k.shape[0] == self.num_kv_heads and k.shape[1] != self.num_kv_heads:
                k = k.transpose(0, 1).contiguous()
                v = v.transpose(0, 1).contiguous()
        return k.to(device=self.device), v.to(device=self.device)

    def append_kv_paged(
        self,
        block_ids: List[int],
        kv_state: PastKV,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> None:
        if not kv_state or not block_ids:
            return
        if isinstance(kv_state[0], tuple):
            self.write_kv(block_ids, start_pos, kv_state)
        else:
            self.write_kv(block_ids, start_pos, [kv_state])  # type: ignore[list-item]

    def fork_kv(self, src_kv: Optional[PastKV]) -> Optional[PastKV]:
        if src_kv is None:
            return None
        # Hybrid caches are not a list of (K, V). to_legacy_cache drops GDN state.
        if cache_is_hybrid(src_kv):
            import copy

            return copy.deepcopy(src_kv)  # type: ignore[return-value]
        # HF DynamicCache / Cache objects
        if hasattr(src_kv, "to_legacy_cache"):
            try:
                from transformers import DynamicCache

                legacy = src_kv.to_legacy_cache()
                cloned = [(k.clone(), v.clone()) for k, v in legacy]
                return DynamicCache.from_legacy_cache(cloned)  # type: ignore[return-value]
            except Exception:
                import copy

                return copy.deepcopy(src_kv)  # type: ignore[return-value]
        if isinstance(src_kv, (list, tuple)):
            return [(k.clone(), v.clone()) for k, v in src_kv]
        import copy

        return copy.deepcopy(src_kv)  # type: ignore[return-value]

    def fork_blocks(self, block_ids: List[int]) -> List[int]:
        """COW: bump refcounts for shared prefix pages."""
        out = list(block_ids)
        for bid in out:
            blk = self._allocated_blocks.get(bid)
            if blk:
                blk.ref_count += 1
        return out

    def cow_block_if_shared(self, block_id: int) -> int:
        """If page is shared, allocate a private copy and return new id."""
        blk = self._allocated_blocks.get(block_id)
        if blk is None or blk.ref_count <= 1:
            return block_id
        new_ids = self.allocate_blocks(1)
        new_id = new_ids[0]
        self.k_cache[:, new_id].copy_(self.k_cache[:, block_id])
        self.v_cache[:, new_id].copy_(self.v_cache[:, block_id])
        if self.ckv_cache is not None:
            self.ckv_cache[:, new_id].copy_(self.ckv_cache[:, block_id])
        if self.kpe_cache is not None:
            self.kpe_cache[:, new_id].copy_(self.kpe_cache[:, block_id])
        if self.packed_swa_cache is not None:
            self.packed_swa_cache[:, new_id].copy_(self.packed_swa_cache[:, block_id])
        if self.packed_comp_cache is not None:
            self.packed_comp_cache[:, new_id].copy_(self.packed_comp_cache[:, block_id])
        if self.restore_bf16_cache is not None:
            self.restore_bf16_cache[:, new_id].copy_(self.restore_bf16_cache[:, block_id])
        self._page_len[new_id] = self._page_len.get(block_id, 0)
        blk.ref_count -= 1
        return new_id

    def get_cache_stats(self) -> Dict:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total)
        used = len(self._allocated_blocks)
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_tokens_stored": self.total_tokens_stored,
            "blocks_used": used,
            "blocks_free": len(self._free_blocks) + (self.num_blocks - self._next_block_id),
            "oom_reject_count": self.oom_reject_count,
            "evict_count": self.evict_count,
            "dual_write_count": self.dual_write_count,
            "dual_write_tokens": self.dual_write_tokens,
            "dual_hit_count": self.dual_hit_count,
            "dual_append_count": self.dual_append_count,
            "dual_restore_count": self.dual_restore_count,
            "dual_stage_count": self.dual_stage_count,
            "has_packed_swa": self.packed_swa_cache is not None
            or self.packed_kv_cache is not None,
            "has_packed_comp": self.packed_comp_cache is not None,
            "has_restore_bf16": self.restore_bf16_cache is not None,
        }

    def allocate_blocks(self, count: int) -> List[int]:
        ids: List[int] = []
        for _ in range(count):
            if self._free_blocks:
                bid = self._free_blocks.pop()
            else:
                if self._next_block_id >= self.num_blocks:
                    freed = self.evict(count - len(ids))
                    if freed <= 0 or not self._free_blocks:
                        self.oom_reject_count += 1
                        # release any partially allocated
                        self.release_blocks(ids)
                        raise MemoryError(
                            f"KV OOM: need {count} blocks, capacity={self.num_blocks}"
                        )
                    bid = self._free_blocks.pop()
                else:
                    bid = self._next_block_id
                    self._next_block_id += 1
            self._allocated_blocks[bid] = KVBlock(block_id=bid, ref_count=1)
            self._page_len[bid] = 0
            ids.append(bid)
        return ids

    def release_blocks(self, block_ids: List[int]) -> None:
        seen: Set[int] = set()
        for bid in block_ids:
            if bid in seen:
                continue
            seen.add(bid)
            blk = self._allocated_blocks.get(bid)
            if not blk:
                continue
            blk.ref_count -= 1
            if blk.ref_count <= 0:
                self._allocated_blocks.pop(bid, None)
                self._page_len.pop(bid, None)
                self.k_cache[:, bid].zero_()
                self.v_cache[:, bid].zero_()
                if self.ckv_cache is not None:
                    self.ckv_cache[:, bid].zero_()
                if self.kpe_cache is not None:
                    self.kpe_cache[:, bid].zero_()
                if self.packed_swa_cache is not None:
                    self.packed_swa_cache[:, bid].zero_()
                if self.packed_comp_cache is not None:
                    self.packed_comp_cache[:, bid].zero_()
                if self.restore_bf16_cache is not None:
                    self.restore_bf16_cache[:, bid].zero_()
                self._free_blocks.append(bid)

    def evict(self, needed: int) -> int:
        """Evict low-refcount leaves from the radix tree and free their private pages."""
        if needed <= 0:
            return 0
        candidates: List[Tuple[int, RadixNode, List[int]]] = []

        def walk(node: RadixNode, path: List[int]):
            if node.block_ids and node.ref_count <= 1 and not node.children:
                # private leaf
                private = [
                    b
                    for b in node.block_ids
                    if self._allocated_blocks.get(b) and self._allocated_blocks[b].ref_count <= 1
                ]
                if private:
                    candidates.append((node.ref_count, node, path[:]))
            for tid, child in node.children.items():
                path.append(tid)
                walk(child, path)
                path.pop()

        walk(self.tree.root, [])
        candidates.sort(key=lambda x: x[0])
        evicted = 0
        for _, node, path in candidates:
            if evicted >= needed:
                break
            blocks = list(node.block_ids)
            node.block_ids = []
            node.past_kv = None
            # unlink leaf
            if path:
                parent = self.tree.root
                for tid in path[:-1]:
                    parent = parent.children[tid]
                parent.children.pop(path[-1], None)
            self.release_blocks(blocks)
            evicted += len(blocks)
            self.evict_count += 1
        return evicted
