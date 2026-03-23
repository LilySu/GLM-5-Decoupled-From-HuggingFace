"""Test 9: DSA sparse mask construction correctness.

Validates that build_dsa_mask produces the right attention pattern:
only DSA-selected positions are unmasked, causal constraint is respected.
"""

import sys
import torch
from .conftest import assert_close


def test_dsa_mask_basic():
    """Selected positions are 0, everything else is -inf."""
    print("\n[Test 9a] DSA mask basic")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_sparse_attention")

    B, S, T, topk = 1, 4, 16, 3
    # Each query selects 3 specific positions
    indices = torch.tensor([[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13]]])
    query = torch.randn(B, 4, S, 32)  # dummy, used for device/dtype
    mask = dsa.build_dsa_mask(indices, attention_mask=None, query_states=query, total_len=T)

    ok = True
    # mask shape: [B, 1, S, T]
    if mask.shape != (1, 1, 4, 16):
        print(f"  FAIL shape: {mask.shape}")
        return False

    for s in range(S):
        selected = set(indices[0, s].tolist())
        for t in range(T):
            val = mask[0, 0, s, t].item()
            if t in selected:
                if val != 0.0:
                    print(f"  FAIL [s={s},t={t}]: selected but mask={val}")
                    ok = False
            else:
                if val != float("-inf"):
                    print(f"  FAIL [s={s},t={t}]: not selected but mask={val}")
                    ok = False

    if ok:
        print("  PASS basic DSA mask (selected=0, rest=-inf)")
    return ok


def test_dsa_mask_with_causal():
    """DSA mask combined with causal mask — both must allow a position."""
    print("\n[Test 9b] DSA mask + causal mask")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_sparse_attention")

    B, S, T = 1, 4, 8
    # DSA selects all positions (topk = T)
    indices = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, S, T)
    query = torch.randn(B, 4, S, 32)

    # Causal mask: query i can only attend to [0..i+past]
    causal = torch.full((1, 1, S, T), float("-inf"), dtype=query.dtype)
    for i in range(S):
        causal[0, 0, i, :i + 1] = 0.0

    mask = dsa.build_dsa_mask(indices, causal, query, T)

    ok = True
    for s in range(S):
        for t in range(T):
            val = mask[0, 0, s, t].item()
            if t <= s:
                # Both DSA and causal allow → should be 0
                if val != 0.0:
                    print(f"  FAIL [s={s},t={t}]: causal allows but mask={val}")
                    ok = False
            else:
                # Causal blocks → should be -inf (even though DSA allows)
                if val == 0.0:
                    print(f"  FAIL [s={s},t={t}]: causal blocks but mask=0")
                    ok = False

    if ok:
        print("  PASS causal + DSA mask intersection")
    return ok


if __name__ == "__main__":
    results = [test_dsa_mask_basic(), test_dsa_mask_with_causal()]
    sys.exit(0 if all(results) else 1)
