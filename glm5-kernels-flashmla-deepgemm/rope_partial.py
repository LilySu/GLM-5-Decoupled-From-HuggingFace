# Decoupled partial-dim RoPE for GLM-5 MLA.
#
# GLM-5's MLA applies RoPE only to a 64-dim slice (qk_rope_head_dim) out of
# the full 256-dim head (qk_head_dim = qk_nope_head_dim + qk_rope_head_dim).
# This is fundamentally different from standard RoPE which rotates all dims.
#
# Ported from: glm5-raw-decoupled-from-hf/model.py (lines 30-132)
#
# STATUS: Pure PyTorch reference implementation. No Triton kernel yet.
# The 64-dim RoPE is memory-bandwidth-bound at this size, so PyTorch is
# fast enough. A fused kernel would be warranted only if profiling shows
# the slice/cat overhead is significant at scale.
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1 — "Multi-latent Attention"
# "we increase the head dimension from 192 to 256 and decrease the number of
#  attention heads by 1/3"

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding to a single tensor.

    unsqueeze_dim=1 for [B, H, S, D] (BHSD), =2 for [B, S, H, D] (BSHD).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# RotaryEmbedding — precomputes inv_freq for the 64-dim rope portion only
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches inv_freq; returns (cos, sin) per forward call.

    GLM-5 config:
        qk_rope_head_dim = 64   (the dim this operates on)
        rope_theta = 10000.0
        max_position_embeddings = 202752
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        dim = cfg["qk_rope_head_dim"]  # 64
        base = cfg["rope_theta"]       # 10000.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [B, S, D] — only used for dtype/device
        # position_ids: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Partial-dim RoPE application patterns used by MLA
# ---------------------------------------------------------------------------

def apply_rope_to_query(query_states, cos, sin, qk_nope_head_dim, qk_rope_head_dim):
    """Split query into nope/rope parts, apply RoPE to rope part, recombine.

    Args:
        query_states: [B, H, S, qk_head_dim] where qk_head_dim = nope + rope
        cos, sin: [B, S, rope_dim] from RotaryEmbedding
        qk_nope_head_dim: 192 (no-position-embedding dims)
        qk_rope_head_dim: 64 (rotary-embedded dims)

    Returns:
        query_states: [B, H, S, qk_head_dim] with RoPE applied to last 64 dims
    """
    q_nope, q_pe = torch.split(query_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)
    return torch.cat([q_nope, q_pe], dim=-1)


def apply_rope_to_compressed_kv_key(k_pe_raw, cos, sin, num_heads):
    """Apply RoPE to the single-head rope stream from compressed KV, expand to all heads.

    Args:
        k_pe_raw: [B, S, qk_rope_head_dim] — the rope portion split from kv_a_proj output
        cos, sin: [B, S, rope_dim] from RotaryEmbedding
        num_heads: 64

    Returns:
        k_pe: [B, H, S, qk_rope_head_dim] — rotated and expanded to all heads
    """
    batch_size, seq_length, rope_dim = k_pe_raw.shape
    k_pe = k_pe_raw.view(batch_size, 1, seq_length, rope_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, num_heads, -1, -1)
    return k_pe
