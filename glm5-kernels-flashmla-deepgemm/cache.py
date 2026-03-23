# KV cache with FlashMLA-compatible format support.
#
# Provides two modes:
# 1. Simple dynamic cache (like glm5-triton): concatenates KV along seq dim
# 2. FlashMLA paged cache: block-based paging with FP8 quantization
#
# The simple mode is used by default and for the non-absorbed attention path.
# The paged mode is used when FlashMLA kernels are active (absorbed weights).

import torch


class KVCache:
    """Dynamic KV cache for autoregressive decoding.

    Stores (key, value) per layer in [B, H, T, D] format.
    Compatible with both eager attention and FlashMLA.
    """

    def __init__(self, num_layers: int):
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states and return full (past + new) tensors."""
        if self._cache[layer_idx] is not None:
            prev_k, prev_v = self._cache[layer_idx]
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx][0].shape[2]

    def reset(self):
        self._cache = [None] * len(self._cache)


class PagedKVCache:
    """FlashMLA-compatible paged KV cache with FP8 support.

    Organizes KV storage into fixed-size pages addressed via block tables.
    Supports FlashMLA's 656-byte FP8 format (V32_FP8Sparse).

    Layout per page: [page_block_size, num_kv_heads, head_dim]
    FP8 layout per token: [512 FP8 nope][4×float32 scales][64 BF16 rope] = 656 bytes

    Args:
        num_layers: Number of decoder layers
        num_pages: Total number of pre-allocated pages
        page_block_size: Tokens per page (default 64, matches FlashMLA)
        num_kv_heads: Number of KV heads (1 for MLA compressed KV)
        head_dim: KV head dimension (576 for absorbed MLA: 512 nope + 64 rope)
        dtype: Storage dtype (torch.bfloat16 or torch.float8_e4m3fn)
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int = 1024,
        page_block_size: int = 64,
        num_kv_heads: int = 1,
        head_dim: int = 576,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_block_size = page_block_size
        self.head_dim = head_dim

        # Pre-allocate all pages for all layers
        self.kv_pages = torch.zeros(
            (num_layers, num_pages, page_block_size, num_kv_heads, head_dim),
            dtype=dtype, device=device,
        )

        # Block tables: [num_layers, max_batch, max_blocks_per_seq]
        # -1 means unallocated
        self.block_tables: list[torch.Tensor | None] = [None] * num_layers

        # Sequence lengths per layer
        self.seq_lengths: list[torch.Tensor | None] = [None] * num_layers

        # Free page tracking
        self._free_pages = list(range(num_pages))

    def allocate_page(self) -> int:
        """Allocate a single page and return its index."""
        if not self._free_pages:
            raise RuntimeError("No free pages available")
        return self._free_pages.pop()

    def free_page(self, page_idx: int):
        """Return a page to the free pool."""
        self._free_pages.append(page_idx)

    def get_kv_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the KV page tensor for a layer: [num_pages, block_size, heads, dim]."""
        return self.kv_pages[layer_idx]

    def get_block_table(self, layer_idx: int) -> torch.Tensor | None:
        """Return block table for a layer: [batch, max_blocks]."""
        return self.block_tables[layer_idx]

    def reset(self):
        """Clear all allocations."""
        self.kv_pages.zero_()
        self.block_tables = [None] * self.num_layers
        self.seq_lengths = [None] * self.num_layers
        self._free_pages = list(range(self.num_pages))
