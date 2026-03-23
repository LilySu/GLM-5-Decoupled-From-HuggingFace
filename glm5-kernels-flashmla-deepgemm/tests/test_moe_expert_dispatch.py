"""Test 10: MoE expert dispatch — tokens route to correct experts with correct weights.

Verifies the fallback loop in moe_grouped_gemm_forward matches a manual computation.
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg


def test_expert_dispatch_single_expert():
    """All tokens routed to expert 0 — output = expert_0(input) * weight."""
    print("\n[Test 10a] Expert dispatch: all tokens to one expert")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashmla-deepgemm.moe_grouped_gemm")

    torch.manual_seed(42)
    N, D, I, E, K = 4, 32, 16, 4, 1

    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2 * I, D)
    down = torch.randn(E, D, I)
    indices = torch.zeros(N, K, dtype=torch.long)  # all to expert 0
    weights = torch.ones(N, K)

    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)

    # Manual: expert 0 forward
    gate, up = F.linear(hidden, gate_up[0]).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, down[0]) * 1.0

    return assert_close("single_expert", output, expected, atol=1e-5)


def test_expert_dispatch_no_tokens():
    """Some experts receive zero tokens — output should still be correct."""
    print("\n[Test 10b] Expert dispatch: sparse routing (most experts empty)")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashmla-deepgemm.moe_grouped_gemm")

    torch.manual_seed(42)
    N, D, I, E, K = 8, 32, 16, 8, 2

    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2 * I, D)
    down = torch.randn(E, D, I)
    # Only experts 0 and 1 receive tokens
    indices = torch.stack([torch.zeros(N, dtype=torch.long),
                           torch.ones(N, dtype=torch.long)], dim=1)
    weights = torch.full((N, K), 0.5)

    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)

    # Manual computation
    expected = torch.zeros_like(hidden)
    for eidx in [0, 1]:
        gate, up = F.linear(hidden, gate_up[eidx]).chunk(2, dim=-1)
        expected += F.linear(F.silu(gate) * up, down[eidx]) * 0.5

    return assert_close("sparse_routing", output, expected, atol=1e-5)


if __name__ == "__main__":
    results = [test_expert_dispatch_single_expert(), test_expert_dispatch_no_tokens()]
    sys.exit(0 if all(results) else 1)
