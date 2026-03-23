# DSA Sparse Attention — FlashInfer kernel-accelerated version.
#
# FlashInfer's trtllm-gen backend supports sparse_mla_top_k parameter for
# token-level sparse attention during decode. For prefill, falls back to
# mask-based eager attention (same as FlashMLA path).
#
# Key difference from FlashMLA path:
#   FlashMLA: flash_mla_sparse_fwd() for prefill, flash_mla_with_kvcache(indices=...) for decode
#   FlashInfer: trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=2048) for decode only

import torch
import torch.nn.functional as F


def build_dsa_mask(topk_indices, attention_mask, query_states, total_len):
    """Build combined DSA sparse + causal attention mask (eager fallback)."""
    batch_size = topk_indices.shape[0]
    seq_length = topk_indices.shape[1]

    index_mask = torch.full(
        (batch_size, seq_length, total_len), float("-inf"),
        device=query_states.device, dtype=query_states.dtype,
    )
    index_mask.scatter_(-1, topk_indices, 0.0)
    index_mask = index_mask.unsqueeze(1)

    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None else index_mask
        )

    return combined_mask


def eager_attention_forward(query, key, value, attention_mask, scaling,
                            num_key_value_groups=1, dropout=0.0, training=False):
    """Standard eager attention with GQA expansion (fallback path)."""
    n_rep = num_key_value_groups
    if n_rep > 1:
        b, h_kv, t, d = key.shape
        key = key[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)
        b, h_kv, t, d = value.shape
        value = value[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
