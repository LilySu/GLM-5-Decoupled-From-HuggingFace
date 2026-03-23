# DSA Sparse Attention — attention over indexer-selected KV subset.
#
# After the DSAIndexer selects top-k token positions per query, this module
# builds a combined causal + sparse mask and runs standard attention through
# only the selected positions (by masking everything else to -inf).
#
# Ported from: glm5-raw-decoupled-from-hf/model.py (lines 299-334, inside MLAttention.forward)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# A fused Triton "Sparse Flash Attention" kernel would avoid materializing
# the full [B, H, S, T] attention matrix and instead gather only the
# DSA-selected KV entries, run attention on the reduced set, then scatter
# back. The paper describes this for Ascend NPUs: "handles the selection
# of TopK tokens from the KV cache and sparse attention computation in
# parallel" (Section 6).
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1.1 — DSA

import torch
import torch.nn.functional as F


def build_dsa_mask(topk_indices, attention_mask, query_states, total_len):
    """Build combined DSA sparse + causal attention mask.

    The DSA mask starts as all -inf, then allows only the top-k positions
    selected by the indexer. This is combined with the standard causal mask.

    Args:
        topk_indices:    [B, S, topk] — indices from DSAIndexer
        attention_mask:  [B, 1, S, T] — standard causal mask (0 for attend, -inf for mask)
        query_states:    used for device/dtype only
        total_len:       T — total key sequence length (past + current)

    Returns:
        combined_mask:   [B, 1, S, T] — merged causal + sparse mask
    """
    batch_size = topk_indices.shape[0]
    seq_length = topk_indices.shape[1]

    # Start with all -inf (block everything)
    index_mask = torch.full(
        (batch_size, seq_length, total_len), float("-inf"),
        device=query_states.device, dtype=query_states.dtype,
    )
    # Allow only the top-k positions selected by DSA indexer
    index_mask.scatter_(-1, topk_indices, 0.0)       # [B, S, T]
    index_mask = index_mask.unsqueeze(1)               # [B, 1, S, T]

    # Combine with causal mask (both must allow a position for it to be attended)
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None else index_mask
        )

    return combined_mask


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Standard eager attention with GQA expansion.

    This is the fallback attention used by the raw model. A Triton sparse
    attention kernel would replace this by gathering only the unmasked KV
    entries instead of computing the full QK^T matrix and masking.

    Args:
        module:          attention module (needs .num_key_value_groups attribute)
        query:           [B, H, S, qk_head_dim]
        key:             [B, H_kv, T, qk_head_dim]
        value:           [B, H_kv, T, v_head_dim]
        attention_mask:  [B, 1, S, T] — combined causal + DSA mask
        scaling:         float — 1/sqrt(qk_head_dim)
        dropout:         float — attention dropout rate

    Returns:
        attn_output:     [B, S, H * v_head_dim]
        attn_weights:    [B, H, S, T]
    """
    # Expand KV heads to match Q heads (GQA / MQA expansion)
    n_rep = module.num_key_value_groups
    if n_rep > 1:
        batch, n_kv_heads, slen, head_dim = key.shape
        key = key[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
        key = key.reshape(batch, n_kv_heads * n_rep, slen, head_dim)
        batch, n_kv_heads, slen, head_dim = value.shape
        value = value[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
        value = value.reshape(batch, n_kv_heads * n_rep, slen, head_dim)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
