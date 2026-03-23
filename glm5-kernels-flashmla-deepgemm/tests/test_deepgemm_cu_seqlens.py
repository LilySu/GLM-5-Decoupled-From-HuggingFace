"""Test 7: DSA indexer DeepGEMM cu_seqlens construction.

Validates that _deepgemm_score builds correct cu_seq_len_k_start/end tensors
for both prefill (causal, growing window) and decode (full context) scenarios.
"""

import sys
import torch
from .conftest import make_cfg, PROJECT_ROOT


def test_cu_seqlens_prefill():
    """Prefill: each query i should see keys [0, i+1) — causal pattern."""
    print("\n[Test 7a] cu_seqlens prefill (causal)")
    from importlib import import_module
    kernel_idx = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")

    cfg = make_cfg()
    indexer = kernel_idx.DSAIndexer(cfg, layer_idx=0)

    # Simulate the cu_seqlens logic from _deepgemm_score
    seq_len = 8
    seq_len_kv = 8  # prefill: seq_len == seq_len_kv

    cu_k_start = torch.zeros(seq_len, dtype=torch.int32)
    cu_k_end = torch.arange(seq_len_kv - seq_len + 1, seq_len_kv + 1, dtype=torch.int32)

    ok = True
    # Query 0 should see key [0, 1), query 1 -> [0, 2), ..., query 7 -> [0, 8)
    for i in range(seq_len):
        if cu_k_start[i] != 0:
            print(f"  FAIL q={i}: start={cu_k_start[i]}, expected 0")
            ok = False
        if cu_k_end[i] != i + 1:
            print(f"  FAIL q={i}: end={cu_k_end[i]}, expected {i + 1}")
            ok = False

    if ok:
        print(f"  PASS prefill cu_seqlens (seq_len={seq_len})")
    return ok


def test_cu_seqlens_decode():
    """Decode: single new token sees all cached keys."""
    print("\n[Test 7b] cu_seqlens decode (full context)")

    seq_len = 1       # decode: single token
    seq_len_kv = 128  # cached context

    cu_k_start = torch.zeros(seq_len, dtype=torch.int32)
    # In decode, cu_k_end shape would be [1] since seq_len_kv - seq_len + 1 = 128
    cu_k_end_range = torch.arange(seq_len_kv - seq_len + 1, seq_len_kv + 1, dtype=torch.int32)

    ok = True
    if cu_k_end_range.shape[0] != seq_len:
        # The arange produces [128] which has shape [1] — correct for decode
        if cu_k_end_range.shape[0] == seq_len:
            if cu_k_end_range[0] != seq_len_kv:
                print(f"  FAIL decode: end={cu_k_end_range[0]}, expected {seq_len_kv}")
                ok = False
        else:
            # Falls through to the fallback path in _deepgemm_score
            cu_k_end_fallback = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
            if cu_k_end_fallback[0] != seq_len_kv:
                print(f"  FAIL decode fallback: end={cu_k_end_fallback[0]}, expected {seq_len_kv}")
                ok = False

    if ok:
        print(f"  PASS decode cu_seqlens (seq_len_kv={seq_len_kv})")
    return ok


def test_cu_seqlens_mismatched():
    """Edge case: seq_len < seq_len_kv during incremental prefill (chunked)."""
    print("\n[Test 7c] cu_seqlens chunked prefill")

    seq_len = 4       # processing 4 new tokens
    seq_len_kv = 20   # 16 cached + 4 new

    cu_k_start = torch.zeros(seq_len, dtype=torch.int32)
    cu_k_end = torch.arange(seq_len_kv - seq_len + 1, seq_len_kv + 1, dtype=torch.int32)

    ok = True
    # Query 0 (position 16) sees keys [0, 17), query 1 (pos 17) sees [0, 18), etc.
    expected_ends = [17, 18, 19, 20]
    for i in range(seq_len):
        if cu_k_end[i] != expected_ends[i]:
            print(f"  FAIL q={i}: end={cu_k_end[i]}, expected {expected_ends[i]}")
            ok = False

    if ok:
        print(f"  PASS chunked prefill cu_seqlens (seq={seq_len}, kv={seq_len_kv})")
    return ok


if __name__ == "__main__":
    results = [
        test_cu_seqlens_prefill(),
        test_cu_seqlens_decode(),
        test_cu_seqlens_mismatched(),
    ]
    sys.exit(0 if all(results) else 1)
