# DSA Lightning Indexer — DeepGEMM kernel-accelerated (same as FlashMLA path).
#
# FlashInfer does not provide a standalone indexer kernel. The DSA scoring is
# done by DeepGEMM fp8_mqa_logits, identical to the FlashMLA path.
# Only the sparse attention step differs (FlashInfer trtllm-gen vs FlashMLA sparse).
#
# Dependencies: pip install deep-gemm (build from source, CUDA 12.8+, SM90)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_partial import apply_rotary_pos_emb

try:
    import deep_gemm
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False


class DSAIndexer(nn.Module):
    """Selects top-k tokens for sparse attention via lightweight scoring.

    Identical to glm5-kernels-flashmla-deepgemm/dsa_indexer.py — the indexer
    is independent of the attention kernel choice (FlashMLA vs FlashInfer).
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg["hidden_size"]
        self.n_heads = cfg["index_n_heads"]
        self.head_dim = cfg["index_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.index_topk = cfg["index_topk"]
        self.q_lora_rank = cfg["q_lora_rank"]

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        self._cached_keys = None
        self.use_deepgemm = DEEP_GEMM_AVAILABLE

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if seq_len > 1:
            self._cached_keys = None
        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)

        if self.use_deepgemm and batch_size == 1:
            index_scores = self._deepgemm_score(q, k_cached, weights)
        else:
            scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
            scores = F.relu(scores)
            index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices

    def _deepgemm_score(self, q, k_cached, weights):
        q_2d = q.squeeze(0)
        k_2d = k_cached.squeeze(0)
        w_2d = weights.squeeze(0)

        q_fp8 = q_2d.to(torch.float8_e4m3fn)
        from .fp8_utils import quantize_activations_deepgemm
        k_fp8, k_scales = quantize_activations_deepgemm(k_2d)

        seq_len = q_2d.shape[0]
        seq_len_kv = k_2d.shape[0]

        cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device=q.device)
        cu_k_end = torch.arange(seq_len_kv - seq_len + 1, seq_len_kv + 1,
                                dtype=torch.int32, device=q.device)
        if cu_k_end.shape[0] < seq_len:
            cu_k_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device=q.device)

        logits = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scales), w_2d, cu_k_start, cu_k_end)
        return logits.unsqueeze(0)
