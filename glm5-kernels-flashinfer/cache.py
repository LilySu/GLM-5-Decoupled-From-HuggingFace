# KV cache with FlashInfer-compatible paged format.
#
# FlashInfer MLA stores KV as two separate paged tensors:
#   ckv_cache: [num_pages, page_size, head_dim_ckv]  (512D compressed KV)
#   kpe_cache: [num_pages, page_size, head_dim_kpe]  (64D RoPE keys)
#
# The BatchMLAPagedAttentionWrapper addresses them via:
#   kv_indptr:  [batch+1] cumulative page counts per sequence
#   kv_indices: [total_pages] page IDs for all sequences
#   kv_len_arr: [batch] actual sequence lengths
#
# For FP8 mode via trtllm-gen, they are concatenated:
#   kv_cache: [num_pages, page_size, 576] (ckv + kpe contiguous)

import torch


class KVCache:
    """Simple dynamic KV cache (eager fallback, same as glm5-triton)."""

    def __init__(self, num_layers: int):
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def update(self, key_states, value_states, layer_idx):
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


class FlashInferPagedKVCache:
    """FlashInfer-native paged KV cache for absorbed MLA.

    Stores compressed KV (ckv) and RoPE keys (kpe) in separate paged tensors,
    matching FlashInfer's BatchMLAPagedAttentionWrapper expected format.

    Args:
        num_layers: decoder layers
        num_pages: total pre-allocated pages across all layers
        page_size: tokens per page (default 64, matches trtllm-gen constraint)
        head_dim_ckv: compressed KV dimension (512 for GLM-5)
        head_dim_kpe: RoPE key dimension (64 for GLM-5)
        dtype: BF16 or FP8
        device: CUDA device
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int = 1024,
        page_size: int = 64,
        head_dim_ckv: int = 512,
        head_dim_kpe: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe

        # Separate ckv and kpe page pools (FlashInfer's wrapper.run() expects them separate)
        self.ckv_pages = torch.zeros(
            (num_layers, num_pages, page_size, head_dim_ckv),
            dtype=dtype, device=device,
        )
        self.kpe_pages = torch.zeros(
            (num_layers, num_pages, page_size, head_dim_kpe),
            dtype=dtype, device=device,
        )

        self._free_pages = list(range(num_pages))

    def allocate_page(self) -> int:
        if not self._free_pages:
            raise RuntimeError("No free pages available")
        return self._free_pages.pop()

    def free_page(self, page_idx: int):
        self._free_pages.append(page_idx)

    def get_ckv_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_pages, page_size, head_dim_ckv]"""
        return self.ckv_pages[layer_idx]

    def get_kpe_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_pages, page_size, head_dim_kpe]"""
        return self.kpe_pages[layer_idx]

    def get_concatenated_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_pages, page_size, head_dim_ckv + head_dim_kpe] for trtllm-gen backend."""
        return torch.cat([
            self.ckv_pages[layer_idx],
            self.kpe_pages[layer_idx],
        ], dim=-1)

    def reset(self):
        self.ckv_pages.zero_()
        self.kpe_pages.zero_()
        self._free_pages = list(range(self.num_pages))
