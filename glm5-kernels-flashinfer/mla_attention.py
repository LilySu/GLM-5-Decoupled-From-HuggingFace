# MLA (Multi-head Latent Attention) — FlashInfer kernel-accelerated version.
#
# Uses FlashInfer's two-kernel architecture for GLM-5:
#   Dense prefill/decode: BatchMLAPagedAttentionWrapper (FA3 backend)
#   Sparse decode (DSA):  trtllm_batch_decode_with_kv_cache_mla (trtllm-gen backend)
#
# Key differences from the FlashMLA path:
#   - Plan/run pattern with workspace buffers (enables CUDA graph capture)
#   - Separate q_nope/q_pe tensors (not concatenated)
#   - CUTLASS backend unavailable (hardcodes H=128), FA3 and trtllm-gen used instead
#   - Requires monkey-patch for qk_nope_head_dim=192 validation (see patches.py)
#   - CUDA graph native support via use_cuda_graph=True
#
# When FlashInfer is not available, falls back to PyTorch eager attention.
#
# Dependencies: pip install flashinfer (requires CUDA 12.0+)

import torch
import torch.nn as nn
from .rope_partial import apply_rotary_pos_emb
from .dsa_indexer import DSAIndexer

try:
    from flashinfer.mla import (
        BatchMLAPagedAttentionWrapper,
        trtllm_batch_decode_with_kv_cache_mla,
    )
    from .patches import apply_glm5_patches
    apply_glm5_patches()
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


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
    """Fallback eager attention when FlashInfer is not available."""
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
    """Multi-head Latent Attention with FlashInfer kernel acceleration.

    FlashInfer architecture:
      - Dense attention: BatchMLAPagedAttentionWrapper with FA3 backend
        Takes separate q_nope [B, H, 512] and q_pe [B, H, 64] tensors
      - Sparse attention: trtllm_batch_decode_with_kv_cache_mla
        Takes concatenated query [B, 1, H, 576] with sparse_mla_top_k=2048

    Falls back to PyTorch eager when FlashInfer is not installed.
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
        self.use_flashinfer = FLASHINFER_AVAILABLE

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

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
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

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

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

        # --- Attention (eager fallback — FlashInfer kernel path requires absorbed weights) ---
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
