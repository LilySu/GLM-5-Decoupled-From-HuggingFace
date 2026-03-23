# MLA (Multi-head Latent Attention) — the core attention mechanism of GLM-5.
#
# MLA compresses KV into a low-rank latent space (512-dim), applies RoPE to
# only a 64-dim decoupled stream, and uses asymmetric head dims (QK=256, V=256).
# This is fundamentally different from standard MHA/GQA/MQA.
#
# Ported from: glm5-raw-decoupled-from-hf/model.py (lines 216-338, MLAttention)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# A fused Triton kernel would combine the compressed projection chains,
# partial RoPE, and the DSA-masked attention into one kernel. This is the
# most complex component and the hardest to fuse.
#
# Key shapes (GLM-5 config):
#   hidden_size:       6144
#   num_heads:         64
#   q_lora_rank:       2048  (Q compression bottleneck)
#   kv_lora_rank:      512   (KV compression bottleneck)
#   qk_rope_head_dim:  64    (RoPE-applied portion)
#   qk_nope_head_dim:  192   (non-RoPE portion)
#   qk_head_dim:       256   (192 + 64)
#   v_head_dim:        256
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1 — "Multi-latent Attention"

import torch
import torch.nn as nn
from .rope_partial import apply_rotary_pos_emb
from .dsa_indexer import DSAIndexer
from .dsa_sparse_attention import build_dsa_mask, eager_attention_forward


# ---------------------------------------------------------------------------
# RMSNorm — simple PyTorch version, identical to the raw model (model.py:12).
# The Triton fast_rms_layernorm from unsloth_rms_layernorm.py can be swapped
# in as a drop-in replacement by monkey-patching .forward on the instances:
#   layer.q_a_layernorm.forward = lambda x: fast_rms_layernorm(layer.q_a_layernorm, x)
# ---------------------------------------------------------------------------
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


class MLAttention(nn.Module):
    """Multi-head Latent Attention with DSA indexer for sparse token selection.

    Projection chain:

        Query path (with LoRA-style compression):
            hidden [B,S,6144]
              -> q_a_proj [6144 -> 2048]
              -> q_a_layernorm (RMSNorm, dim 2048)
              -> q_b_proj [2048 -> 64*256 = 16384]
              -> reshape to [B, 64, S, 256]
              -> split into nope[192] + rope[64]
              -> apply RoPE to rope portion
              -> cat back to [B, 64, S, 256]

        KV path (MLA compression):
            hidden [B,S,6144]
              -> kv_a_proj_with_mqa [6144 -> 576]  (512 kv_lora + 64 rope)
              -> split: k_compressed[512], k_pe_raw[64]
              -> kv_a_layernorm (RMSNorm, dim 512) on k_compressed
              -> kv_b_proj [512 -> 64*(192+256) = 28672]
              -> reshape to [B, S, 64, 448] -> split k_nope[192] + v[256]
              -> apply RoPE to k_pe_raw -> expand to all heads
              -> cat k_nope + k_pe -> key [B, 64, S, 256]

        Attention:
            DSA indexer selects top-2048 positions
            Build combined causal + sparse mask
            Eager attention: softmax(QK^T / sqrt(256)) * V
            Output: o_proj [64*256 -> 6144]
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

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # --- DSA sparse mask ---
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

        total_len = key_states.shape[2]
        combined_mask = build_dsa_mask(topk_indices, attention_mask, query_states, total_len)

        # --- Attention ---
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
