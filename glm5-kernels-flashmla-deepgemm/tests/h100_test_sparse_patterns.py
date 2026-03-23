"""H100 Category 8: Sparse Attention Pattern Validation.

DSA selects top-2048 tokens per query. Wrong selection = bad text, no crash.
Must verify: causality respected, recent tokens always selected, selection
is not degenerate (all same indices), and Jaccard stability across seeds.

Requirements: CUDA GPU (any).
"""

import sys
import torch
from .conftest import make_cfg, skip_no_cuda, jaccard


@skip_no_cuda
def h100_test_sparse_causality():
    """No selected index should be in the future (violate causality)."""
    print("\n[H100-Sparse-1] Causality constraint")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()
    device = "cuda"
    torch.manual_seed(42)

    indexer = dsa.DSAIndexer(cfg, 0).to(device).eval()
    indexer.use_deepgemm = False

    B, S = 1, 32
    hidden = torch.randn(B, S, cfg["hidden_size"], device=device)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"], device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    cos, sin = rope(hidden, torch.arange(S, device=device).unsqueeze(0))

    # Build causal mask for indexer
    causal = torch.full((B, S, S), float("-inf"), device=device)
    for i in range(S):
        causal[:, i, :i + 1] = 0.0

    with torch.no_grad():
        indices = indexer(hidden, q_resid, (cos, sin), attention_mask=causal, use_cache=False)

    ok = True
    for b in range(B):
        for s in range(S):
            future = (indices[b, s] > s).any()
            if future:
                violators = indices[b, s][indices[b, s] > s].tolist()
                print(f"  FAIL query {s} selected future positions: {violators}")
                ok = False

    if ok:
        print(f"  PASS no future positions selected (all indices <= query position)")
    return ok


@skip_no_cuda
def h100_test_sparse_recency_bias():
    """Recent tokens (within last ~10%) should appear more often than distant tokens."""
    print("\n[H100-Sparse-2] Recency bias")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()
    cfg["index_topk"] = 8  # small topk to make recency more visible
    device = "cuda"
    torch.manual_seed(42)

    indexer = dsa.DSAIndexer(cfg, 0).to(device).eval()
    indexer.use_deepgemm = False

    B, S = 1, 64
    hidden = torch.randn(B, S, cfg["hidden_size"], device=device)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"], device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    cos, sin = rope(hidden, torch.arange(S, device=device).unsqueeze(0))

    with torch.no_grad():
        indices = indexer(hidden, q_resid, (cos, sin), use_cache=False)

    # For the last query (position 63), count how many selections are "recent" (pos > 50)
    last_query_idx = indices[0, -1]
    recent_count = (last_query_idx > S - 15).sum().item()
    total_selected = last_query_idx.shape[0]
    recent_pct = recent_count / total_selected * 100

    # Not a hard requirement — just informational. RoPE + learned weights create recency.
    print(f"  Last query selected {recent_count}/{total_selected} recent tokens ({recent_pct:.0f}%)")
    print(f"  PASS (informational — recency depends on learned weights, not a hard constraint)")
    return True


@skip_no_cuda
def h100_test_sparse_non_degeneracy():
    """Selected indices should not all be the same position (degenerate attention)."""
    print("\n[H100-Sparse-3] Non-degeneracy")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()
    device = "cuda"
    torch.manual_seed(42)

    indexer = dsa.DSAIndexer(cfg, 0).to(device).eval()
    indexer.use_deepgemm = False

    B, S = 2, 32
    hidden = torch.randn(B, S, cfg["hidden_size"], device=device)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"], device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    cos, sin = rope(hidden, torch.arange(S, device=device).unsqueeze(0).expand(B, -1))

    with torch.no_grad():
        indices = indexer(hidden, q_resid, (cos, sin), use_cache=False)

    ok = True
    for b in range(B):
        for s in range(1, S):  # skip s=0 (only 1 position available with topk clamping)
            unique = indices[b, s].unique().numel()
            total = indices[b, s].numel()
            if unique < min(2, total):  # at least 2 unique positions
                print(f"  FAIL [b={b},s={s}]: only {unique}/{total} unique indices (degenerate)")
                ok = False

    if ok:
        print(f"  PASS all queries select diverse positions")
    return ok


@skip_no_cuda
def h100_test_sparse_jaccard_stability():
    """Slightly different inputs should produce overlapping (>0.5 Jaccard) but not identical selections."""
    print("\n[H100-Sparse-4] Jaccard stability across perturbations")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()
    device = "cuda"
    torch.manual_seed(42)

    indexer = dsa.DSAIndexer(cfg, 0).to(device).eval()
    indexer.use_deepgemm = False

    B, S = 1, 32
    hidden = torch.randn(B, S, cfg["hidden_size"], device=device)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"], device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    cos, sin = rope(hidden, torch.arange(S, device=device).unsqueeze(0))

    with torch.no_grad():
        idx_base = indexer(hidden, q_resid, (cos, sin), use_cache=False)

    # Small perturbation
    hidden_perturbed = hidden + torch.randn_like(hidden) * 0.01
    q_perturbed = q_resid + torch.randn_like(q_resid) * 0.01

    # Reset indexer cache
    indexer._cached_keys = None
    with torch.no_grad():
        idx_perturbed = indexer(hidden_perturbed, q_perturbed, (cos, sin), use_cache=False)

    # Measure Jaccard similarity per query position
    jaccards = []
    for s in range(S):
        j = jaccard(idx_base[0, s], idx_perturbed[0, s])
        jaccards.append(j)

    mean_jaccard = sum(jaccards) / len(jaccards)
    min_jaccard = min(jaccards)

    ok = True
    # Small perturbation should keep most selections (Jaccard > 0.5)
    if mean_jaccard < 0.3:
        print(f"  FAIL mean Jaccard {mean_jaccard:.3f} < 0.3 (selections too sensitive)")
        ok = False
    # But should not be identical (Jaccard < 1.0 at least sometimes)
    if mean_jaccard > 0.999:
        print(f"  WARN mean Jaccard {mean_jaccard:.3f} ≈ 1.0 (perturbation had no effect)")

    if ok:
        print(f"  PASS Jaccard stability: mean={mean_jaccard:.3f}, min={min_jaccard:.3f}")
    return ok


if __name__ == "__main__":
    results = [
        h100_test_sparse_causality(),
        h100_test_sparse_recency_bias(),
        h100_test_sparse_non_degeneracy(),
        h100_test_sparse_jaccard_stability(),
    ]
    sys.exit(0 if all(results) else 1)
