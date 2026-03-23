# MoE sigmoid routing — pure PyTorch.
#
# GLM-5 uses n_group=1, topk_group=1, meaning group-level routing is a no-op.
# The routing simplifies to: sigmoid -> bias -> flat top-8 -> normalize -> scale.
#
# This is ~5 lines of core logic and runs in <10us — no custom CUDA kernel needed.
#
# If the full 744B model uses n_group>1 (like DeepSeek-V3's n_group=8), install
# sgl-kernel and use moe_fused_gate instead.
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopkRouter(nn.Module):
    """MoE routing layer: linear projection -> per-expert scores.

    Uses sigmoid activation (not softmax) — a key GLM-5 design choice.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_routed_experts = cfg["n_routed_experts"]
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        return F.linear(x.float(), self.weight.float())


def sigmoid_topk_route(
    router_logits: torch.Tensor,     # [num_tokens, n_routed_experts]
    correction_bias: torch.Tensor,   # [n_routed_experts]
    top_k: int = 8,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid-based expert routing with optional group selection.

    For GLM-5 (n_group=1, topk_group=1): flat sigmoid + top-k.
    For DeepSeek-V3 (n_group=8, topk_group=4): hierarchical group -> expert selection.

    Args:
        router_logits:  [N, E] raw router output
        correction_bias: [E] additive bias for load balancing
        top_k: number of experts to select per token
        n_group: number of expert groups
        topk_group: number of groups to select
        norm_topk_prob: whether to normalize routing weights
        routed_scaling_factor: scale factor applied after normalization

    Returns:
        topk_indices: [N, top_k] selected expert indices
        topk_weights: [N, top_k] routing weights (normalized and scaled)
    """
    scores = router_logits.sigmoid()
    scores_biased = scores + correction_bias.unsqueeze(0)

    if n_group > 1 and topk_group < n_group:
        # Hierarchical group-based routing (DeepSeek-V3 style)
        n_experts = scores_biased.shape[-1]
        experts_per_group = n_experts // n_group

        # Score each group by top-2 experts within it
        group_scores = (
            scores_biased.view(-1, n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        # Select top groups
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        # Expand to per-expert mask
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, experts_per_group)
            .reshape(-1, n_experts)
        )
        scores_biased = scores_biased.masked_fill(~score_mask.bool(), 0.0)

    # Select top-k experts
    topk_indices = torch.topk(scores_biased, k=top_k, dim=-1, sorted=False)[1]

    # Get weights from ORIGINAL scores (not bias-adjusted)
    topk_weights = scores.gather(1, topk_indices)

    # Normalize
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    # Scale
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights
