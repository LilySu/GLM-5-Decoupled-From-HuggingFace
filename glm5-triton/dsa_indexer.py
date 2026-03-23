# DSA Lightning Indexer — selects top-k tokens for sparse attention.
#
# For each query position, the indexer scores all available key positions
# and returns the top-2048 most relevant ones. The main attention then
# only computes over this sparse subset instead of the full sequence.
#
# Ported from: glm5-raw-decoupled-from-hf/model.py (lines 139-209, DSAIndexer)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# A fused Triton kernel would combine the score computation, ReLU, weighted
# sum across heads, and TopK into a single kernel (the paper calls this
# "Lightning Indexer" — see Section 2.1.1 and the Ascend NPU optimization
# in Section 6 which fuses "score calculation, ReLU, and TopK operations
# into a single kernel").
#
# IMPORTANT: The paper (Section 3.2, "DSA RL insights") notes that
# deterministic TopK is critical — non-deterministic CUDA topk caused
# "drastic performance degradation during RL after only a few steps."
# Use torch.topk (deterministic) not custom CUDA topk.
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_partial import apply_rotary_pos_emb


class DSAIndexer(nn.Module):
    """Selects top-k tokens for sparse attention via lightweight scoring.

    Architecture (GLM-5 config):
        index_n_heads:      32   (lightweight scoring heads)
        index_head_dim:     128  (dim per scoring head)
        index_topk:         2048 (max tokens to attend to)
        q_lora_rank:        2048 (input from MLA's compressed query)
        qk_rope_head_dim:   64   (partial RoPE applied to scoring Q/K)

    Forward flow:
        1. Query: q_resid [B,S,2048] -> wq_b -> [B,S,32*128] -> split rope/nope -> RoPE -> concat
        2. Key:   hidden  [B,S,6144] -> wk   -> [B,S,128]     -> LayerNorm -> split -> RoPE -> concat
        3. Score: einsum(q, k) * scale -> ReLU -> weighted sum across heads -> [B,S,T]
        4. TopK:  select top-2048 positions per query position -> [B,S,2048]

    The indexer maintains its own key cache separate from the main KV cache.
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = cfg["hidden_size"]        # 6144
        self.n_heads = cfg["index_n_heads"]           # 32
        self.head_dim = cfg["index_head_dim"]          # 128
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]  # 64
        self.index_topk = cfg["index_topk"]            # 2048
        self.q_lora_rank = cfg["q_lora_rank"]          # 2048

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        # Own key cache (separate from the main KVCache)
        self._cached_keys = None

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        """Returns top-k token indices [B, S, topk].

        Args:
            hidden_states: [B, S, hidden_size] — current layer input
            q_resid:       [B, S, q_lora_rank] — compressed query from MLA's q_a path
            position_embeddings: (cos, sin) from RotaryEmbedding
            attention_mask: [B, S, T] or None — causal mask for scoring
            use_cache: whether to maintain key cache for autoregressive decoding

        Returns:
            indices: [B, S, topk] — indices of selected tokens per query position
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # --- Queries from compressed MLA representation ---
        q = self.wq_b(q_resid)  # [B, S, n_heads * head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # --- Keys from full hidden states ---
        k = self.k_norm(self.wk(hidden_states))  # [B, S, head_dim]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # --- Key cache management ---
        if seq_len > 1:
            self._cached_keys = None  # Reset on prefill

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        # --- Scoring: q dot k per head, ReLU, weighted sum across heads ---
        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)  # [B, S, n_heads]
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        scores = F.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)  # [B, S, T]

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        # --- Deterministic TopK selection ---
        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices  # [B, S, topk]
