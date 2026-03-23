"""Test 13: MoE group routing with n_group > 1 (DeepSeek-V3 style).

Tests that our sigmoid_topk_route correctly implements hierarchical
group→expert selection when n_group=4, topk_group=2.
"""

import sys
import torch
from .conftest import assert_close


def test_group_routing_filters_groups():
    """With n_group=4, topk_group=2: experts in eliminated groups should never be selected."""
    print("\n[Test 13a] Group routing: eliminated groups excluded")
    from importlib import import_module
    router = import_module("glm5-kernels-flashmla-deepgemm.moe_router")

    torch.manual_seed(42)
    N, E = 8, 16  # 16 experts, 4 groups of 4
    n_group, topk_group, top_k = 4, 2, 4

    # Make group 0 and 1 have high scores, groups 2 and 3 low
    logits = torch.randn(N, E)
    logits[:, :4] += 5.0   # group 0: high
    logits[:, 4:8] += 5.0  # group 1: high
    logits[:, 8:12] -= 5.0  # group 2: low
    logits[:, 12:16] -= 5.0  # group 3: low

    bias = torch.zeros(E)
    indices, weights = router.sigmoid_topk_route(
        logits, bias, top_k=top_k, n_group=n_group,
        topk_group=topk_group, norm_topk_prob=True, routed_scaling_factor=1.0,
    )

    ok = True
    for i in range(N):
        selected = set(indices[i].tolist())
        # All selected experts should be in groups 0 or 1 (indices 0-7)
        for eidx in selected:
            group = eidx // (E // n_group)
            if group >= topk_group:
                print(f"  FAIL token {i}: expert {eidx} from eliminated group {group}")
                ok = False

    if ok:
        print("  PASS experts only from surviving groups")
    return ok


def test_group_routing_vs_flat():
    """With n_group=1, group routing should be identical to flat top-k."""
    print("\n[Test 13b] n_group=1 equivalent to flat top-k")
    from importlib import import_module
    router = import_module("glm5-kernels-flashmla-deepgemm.moe_router")

    torch.manual_seed(42)
    N, E = 16, 8
    logits = torch.randn(N, E)
    bias = torch.randn(E)

    # n_group=1 path
    idx_g1, w_g1 = router.sigmoid_topk_route(
        logits.clone(), bias, top_k=2, n_group=1, topk_group=1,
        norm_topk_prob=True, routed_scaling_factor=2.5,
    )

    # n_group=E path (every expert its own group, select all groups)
    # This should also be equivalent to flat when topk_group=n_group
    idx_gE, w_gE = router.sigmoid_topk_route(
        logits.clone(), bias, top_k=2, n_group=E, topk_group=E,
        norm_topk_prob=True, routed_scaling_factor=2.5,
    )

    ok = True
    for i in range(N):
        if set(idx_g1[i].tolist()) != set(idx_gE[i].tolist()):
            print(f"  FAIL token {i}: n_group=1 {set(idx_g1[i].tolist())} vs n_group=E {set(idx_gE[i].tolist())}")
            ok = False

    if ok:
        # Compare weights (sort by index first)
        s1 = idx_g1.sort(dim=-1)
        sE = idx_gE.sort(dim=-1)
        ok = assert_close("flat_vs_allgroups_weights",
                          w_g1.gather(1, s1.indices), w_gE.gather(1, sE.indices), atol=1e-5)
    return ok


if __name__ == "__main__":
    results = [test_group_routing_filters_groups(), test_group_routing_vs_flat()]
    sys.exit(0 if all(results) else 1)
