# MLA (Multi-head Latent Attention) — FlashMLA kernel-accelerated version.
#
# Replaces the pure PyTorch eager attention from glm5-triton/mla_attention.py
# with FlashMLA CUDA kernels for H100 (SM90).
#
# FlashMLA provides:
#   - flash_mla_with_kvcache():  Dense/sparse decode with paged FP8 KV cache
#   - flash_mla_sparse_fwd():    Sparse prefill (non-paged)
#
# CRITICAL: Requires weight absorption. The kv_b_proj weights are absorbed into
# the query projection and output projection at load time. After absorption:
#   - KV cache = 576D (512 compressed nope + 64 BF16 rope)
#   - Q absorbed = 576D (512 absorbed nope + 64 rope)
#   - V output = 512D (kv_lora_rank, not v_head_dim=256)
#   - W_O absorbed to map [num_heads, 512] -> [hidden_size]
#
# When FlashMLA is not available (no SM90 GPU), falls back to PyTorch eager attention.
#
# Dependencies: pip install flash-mla (build from source, CUDA 12.8+, SM90)
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1

import torch
import torch.nn as nn
from .rope_partial import apply_rotary_pos_emb
from .dsa_indexer import DSAIndexer
from .cache import PagedKVCache

try:
    from flash_mla import (
        get_mla_metadata,
        flash_mla_with_kvcache,
        flash_mla_sparse_fwd,
    )
    FLASH_MLA_AVAILABLE = True
except ImportError:
    FLASH_MLA_AVAILABLE = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def _eager_attention_forward(query, key, value, attention_mask, scaling, num_key_value_groups=1, dropout=0.0, training=False):
    """Fallback eager attention when FlashMLA is not available."""
    n_rep = num_key_value_groups
    if n_rep > 1:
        b, h_kv, t, d = key.shape
        key = key[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)
        b, h_kv, t, d = value.shape
        value = value[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class MLAttention(nn.Module):
    """Multi-head Latent Attention with FlashMLA kernel acceleration.

    When FlashMLA is available (SM90), uses CUDA kernels for attention.
    Otherwise falls back to PyTorch eager attention.

    The projection chain is identical to glm5-triton/mla_attention.py.
    The difference is in the attention computation itself.
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        self.attention_dropout = cfg["attention_dropout"]
        self.num_heads = cfg["num_attention_heads"]

        self.q_lora_rank = cfg["q_lora_rank"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.v_head_dim = cfg["v_head_dim"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_head_dim = cfg["qk_head_dim"]

        self.is_causal = True
        self.use_flash_mla = FLASH_MLA_AVAILABLE

        # Query projection (LoRA-style compression path)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg["hidden_size"], self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(cfg["hidden_size"], cfg["q_lora_rank"], bias=cfg["attention_bias"])
            self.q_a_layernorm = RMSNorm(cfg["q_lora_rank"])
            self.q_b_proj = nn.Linear(cfg["q_lora_rank"], self.num_heads * self.qk_head_dim, bias=False)

        # KV projections (MLA compressed path)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg["hidden_size"], self.kv_lora_rank + self.qk_rope_head_dim, bias=cfg["attention_bias"],
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, cfg["hidden_size"], bias=cfg["attention_bias"])

        self.scaling = self.qk_head_dim ** -0.5

        self.indexer = DSAIndexer(cfg, layer_idx)

        # FlashMLA scheduler metadata (lazy init)
        self._flash_mla_metadata = None

    def absorb_weights(self):
        """Absorb kv_b_proj into Q and O projections for FlashMLA compatibility.

        After absorption:
        - KV cache stores 576D (512 compressed nope + 64 BF16 rope)
        - Query is projected to 576D (512 absorbed nope + 64 rope) per head
        - Output projection maps [num_heads, 512] -> [hidden_size]
        - kv_b_proj is no longer needed (removed)

        This enables FlashMLA kernels which require d_qk=576, d_v=512.
        """
        if hasattr(self, '_weights_absorbed') and self._weights_absorbed:
            return

        W_kv_b = self.kv_b_proj.weight  # [H*(qk_nope+v_head), kv_lora_rank]
        W_kv_b = W_kv_b.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        W_k_nope = W_kv_b[:, :self.qk_nope_head_dim, :]  # [H, qk_nope, kv_lora]
        W_v = W_kv_b[:, self.qk_nope_head_dim:, :]        # [H, v_head, kv_lora]

        # Absorb W_k_nope into Q_b_proj: Q_absorbed_nope = Q_nope @ W_k_nope
        # Q_b produces [H, qk_nope + qk_rope] per token
        # We replace the nope portion: new_q_nope[h] = old_q_nope[h] @ W_k_nope[h]
        # This makes q_nope match the kv_lora_rank (512) dimension
        if self.q_lora_rank is not None:
            W_qb = self.q_b_proj.weight  # [H*qk_head, q_lora_rank]
            W_qb = W_qb.view(self.num_heads, self.qk_head_dim, self.q_lora_rank)
            W_qb_nope = W_qb[:, :self.qk_nope_head_dim, :]  # [H, qk_nope, q_lora]
            W_qb_rope = W_qb[:, self.qk_nope_head_dim:, :]  # [H, qk_rope, q_lora]

            # absorbed_nope: [H, kv_lora, q_lora] = W_k_nope^T @ W_qb_nope
            W_qb_absorbed = torch.bmm(W_k_nope.transpose(1, 2), W_qb_nope)  # [H, kv_lora, q_lora]
            # New q_b weight: [H, (kv_lora + qk_rope), q_lora]
            W_qb_new = torch.cat([W_qb_absorbed, W_qb_rope], dim=1)
            new_q_b = nn.Linear(self.q_lora_rank, self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim), bias=False)
            new_q_b.weight.data = W_qb_new.reshape(-1, self.q_lora_rank).to(self.q_b_proj.weight.device)
            self.q_b_proj = new_q_b.to(device=self.q_b_proj.weight.device, dtype=self.q_b_proj.weight.dtype)

        # Absorb W_v into O_proj: O_absorbed = O @ W_v (per head)
        # W_o: [hidden, H*v_head] -> reshaped to [hidden, H, v_head]
        # W_v: [H, v_head, kv_lora]
        # O_new[:, h, :] = W_o[:, h, :] @ W_v[h] -> [hidden, H, kv_lora]
        hidden_dim = self.o_proj.weight.shape[0]
        W_o = self.o_proj.weight.view(hidden_dim, self.num_heads, self.v_head_dim)
        W_o_new = torch.einsum('dhv,hvl->dhl', W_o, W_v)  # [hidden, H, kv_lora]
        device = self.o_proj.weight.device
        dtype = self.o_proj.weight.dtype
        new_o = nn.Linear(self.num_heads * self.kv_lora_rank, hidden_dim, bias=False)
        new_o.weight.data = W_o_new.reshape(hidden_dim, -1).to(device)
        self.o_proj = new_o.to(device=device, dtype=dtype)

        # Update dimensions for absorbed path
        self._absorbed_qk_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self._absorbed_v_dim = self.kv_lora_rank  # 512

        # Remove kv_b_proj (no longer needed)
        del self.kv_b_proj

        self._weights_absorbed = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if hasattr(self, '_weights_absorbed') and self._weights_absorbed:
            return self._absorbed_forward(hidden_states, position_embeddings, attention_mask, past_key_values)
        return self._eager_forward(hidden_states, position_embeddings, attention_mask, past_key_values)

    def _absorbed_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        """FlashMLA absorbed forward path.

        After weight absorption:
        - Q: [B, S, H, 576] (512 absorbed nope + 64 rope)
        - KV cache: [num_pages, page_size, 1, 576] (512 compressed + 64 rope)
        - V output dim: 512 (kv_lora_rank)

        Uses flash_mla_with_kvcache for decode, eager for prefill.
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # --- Query (absorbed: nope is now kv_lora_rank=512) ---
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self._absorbed_qk_dim)
        q_nope, q_pe = torch.split(query_states, [self._absorbed_v_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe.transpose(1, 2), cos, sin, unsqueeze_dim=1).transpose(1, 2)
        query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, S, H, 576]

        # --- KV (compressed, no kv_b_proj expansion needed) ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        kv_cache_entry = torch.cat([k_compressed, k_pe], dim=-1)  # [B, S, 576]

        # --- DSA indexer ---
        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1) if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states, q_resid, position_embeddings, indexer_mask,
            use_cache=past_key_values is not None,
        )

        # --- FlashMLA decode path (B>=1, S=1) ---
        if self.use_flash_mla and seq_length == 1 and isinstance(past_key_values, PagedKVCache):
            kv_page_cache = past_key_values.get_kv_cache(self.layer_idx)
            block_table = past_key_values.get_block_table(self.layer_idx)
            cache_seqlens = past_key_values.seq_lengths[self.layer_idx]

            metadata, _ = get_mla_metadata(
                cache_seqlens,
                past_key_values.page_block_size * torch.ones(1, dtype=torch.int32, device=hidden_states.device),
            )

            attn_output, _ = flash_mla_with_kvcache(
                query_states,  # [B, 1, H, 576]
                kv_page_cache,  # [num_pages, page_size, 1, 576]
                block_table,
                cache_seqlens,
                head_dim_v=self._absorbed_v_dim,  # 512
                tile_scheduler_metadata=metadata,
                softmax_scale=self._absorbed_qk_dim ** -0.5,
                causal=False,  # causal handled by cache_seqlens
            )
            # attn_output: [B, 1, H, 512]
            attn_output = attn_output.reshape(batch_size, seq_length, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        # --- Eager fallback for prefill or non-paged cache ---
        kv_cache_entry = kv_cache_entry.unsqueeze(2)  # [B, S, 1, 576]
        k_states = kv_cache_entry.expand(-1, -1, self.num_heads, -1).transpose(1, 2)  # [B, H, S, 576]
        v_states = k_states[..., :self._absorbed_v_dim]  # [B, H, S, 512]
        query_states = query_states.transpose(1, 2)  # [B, H, S, 576]

        if past_key_values is not None and hasattr(past_key_values, 'update'):
            k_states, v_states = past_key_values.update(k_states, v_states, self.layer_idx)

        total_len = k_states.shape[2]
        combined_mask = _build_dsa_mask(topk_indices, attention_mask, query_states, total_len)

        attn_output, attn_weights = _eager_attention_forward(
            query_states, k_states, v_states, combined_mask,
            scaling=self._absorbed_qk_dim ** -0.5,
            num_key_value_groups=1,  # absorbed: all heads share same KV
            dropout=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _eager_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        """Non-absorbed eager forward path (identical to glm5-triton)."""
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # --- Query path ---
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        # --- KV path ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # --- DSA sparse token selection ---
        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1) if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states, q_resid, position_embeddings, indexer_mask,
            use_cache=past_key_values is not None,
        )

        # --- Eager attention with DSA sparse masking ---
        total_len = key_states.shape[2]
        combined_mask = _build_dsa_mask(topk_indices, attention_mask, query_states, total_len)

        attn_output, attn_weights = _eager_attention_forward(
            query_states, key_states, value_states, combined_mask,
            scaling=self.scaling,
            num_key_value_groups=self.num_key_value_groups,
            dropout=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _build_dsa_mask(topk_indices, attention_mask, query_states, total_len):
    """Build combined DSA sparse + causal attention mask."""
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
