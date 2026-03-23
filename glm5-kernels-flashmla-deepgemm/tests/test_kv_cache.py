"""Test 8: KV cache correctness across multiple decode steps.

Validates that KVCache.update() correctly concatenates keys/values
across sequential decode steps and that PagedKVCache allocates correctly.
"""

import sys
import torch
from .conftest import assert_close


def test_kvcache_multistep():
    """Verify KV cache concatenation across 5 decode steps."""
    print("\n[Test 8a] KVCache multi-step concatenation")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashmla-deepgemm.cache")

    cache = cache_mod.KVCache(num_layers=2)
    B, H, D = 2, 4, 32

    all_keys = []
    all_vals = []
    for step in range(5):
        k = torch.randn(B, H, 1, D) + step  # offset so each step is distinguishable
        v = torch.randn(B, H, 1, D) + step
        all_keys.append(k)
        all_vals.append(v)
        full_k, full_v = cache.update(k, v, layer_idx=0)

    ok = True
    expected_k = torch.cat(all_keys, dim=2)
    expected_v = torch.cat(all_vals, dim=2)

    ok = assert_close("cached_keys", full_k, expected_k, atol=0) and ok
    ok = assert_close("cached_vals", full_v, expected_v, atol=0) and ok

    if cache.get_seq_length(0) != 5:
        print(f"  FAIL seq_length: {cache.get_seq_length(0)}, expected 5")
        ok = False
    else:
        print(f"  PASS seq_length = 5")

    # Layer 1 should still be empty
    if cache.get_seq_length(1) != 0:
        print(f"  FAIL layer_1 seq_length: {cache.get_seq_length(1)}, expected 0")
        ok = False

    return ok


def test_kvcache_reset():
    """Verify cache reset clears all state."""
    print("\n[Test 8b] KVCache reset")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashmla-deepgemm.cache")

    cache = cache_mod.KVCache(num_layers=2)
    k = torch.randn(1, 4, 3, 32)
    v = torch.randn(1, 4, 3, 32)
    cache.update(k, v, layer_idx=0)

    cache.reset()
    ok = cache.get_seq_length(0) == 0
    if ok:
        print("  PASS reset clears cache")
    else:
        print("  FAIL reset did not clear cache")
    return ok


def test_paged_kvcache_allocation():
    """Verify PagedKVCache page allocation and freeing."""
    print("\n[Test 8c] PagedKVCache allocation")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashmla-deepgemm.cache")

    cache = cache_mod.PagedKVCache(
        num_layers=1, num_pages=8, page_block_size=4,
        num_kv_heads=1, head_dim=64, dtype=torch.bfloat16, device="cpu",
    )

    ok = True
    # Allocate 5 pages
    pages = [cache.allocate_page() for _ in range(5)]
    if len(set(pages)) != 5:
        print(f"  FAIL duplicate pages: {pages}")
        ok = False

    # Free 2 and reallocate
    cache.free_page(pages[0])
    cache.free_page(pages[1])
    p_new = cache.allocate_page()
    if p_new not in (pages[0], pages[1]):
        print(f"  FAIL freed page not reused: got {p_new}, expected {pages[0]} or {pages[1]}")
        ok = False

    # Exhaust remaining pages
    remaining = 8 - 5 + 1  # 5 allocated, freed 2, reallocated 1 → 3 left + 1 freed = 4 free
    try:
        for _ in range(10):
            cache.allocate_page()
        print("  FAIL should have raised on exhaustion")
        ok = False
    except RuntimeError:
        print("  PASS allocation exhaustion raises correctly")

    if ok:
        print("  PASS page allocation")
    return ok


if __name__ == "__main__":
    results = [
        test_kvcache_multistep(),
        test_kvcache_reset(),
        test_paged_kvcache_allocation(),
    ]
    sys.exit(0 if all(results) else 1)
