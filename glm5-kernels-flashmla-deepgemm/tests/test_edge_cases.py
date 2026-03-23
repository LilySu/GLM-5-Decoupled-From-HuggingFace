"""Test 16: Edge cases — batch_size=1, seq_len=1, topk > seq_len, empty experts.

These boundary conditions are where bugs tend to hide.
"""

import sys
import torch
from .conftest import assert_close, make_cfg


def test_single_token_forward():
    """batch_size=1, seq_len=1 — minimum viable forward pass."""
    print("\n[Test 16a] Single token forward (B=1, S=1)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=1)
    torch.manual_seed(42)
    model = kernel_model.GlmMoeDsaForCausalLM(cfg)
    for layer in model.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False
    model.eval()

    input_ids = torch.tensor([[42]])
    with torch.no_grad():
        loss, logits, cache = model(input_ids=input_ids, use_cache=True)

    ok = True
    if logits.shape != (1, 1, cfg["vocab_size"]):
        print(f"  FAIL shape: {logits.shape}")
        ok = False
    if not torch.isfinite(logits).all():
        print(f"  FAIL non-finite logits")
        ok = False
    if ok:
        print(f"  PASS single token: logits {logits.shape}, all finite")
    return ok


def test_topk_exceeds_seq_len():
    """index_topk=16 but seq_len=4 — topk should be clamped to seq_len."""
    print("\n[Test 16b] topk > seq_len (topk=16, seq=4)")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()  # index_topk=16
    torch.manual_seed(42)
    indexer = dsa.DSAIndexer(cfg, layer_idx=0)
    indexer.use_deepgemm = False

    B, S, D = 1, 4, cfg["hidden_size"]  # S=4 < index_topk=16
    hidden = torch.randn(B, S, D)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"])
    rope = rope_mod.RotaryEmbedding(cfg)
    pos_ids = torch.arange(S).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)

    indices = indexer(hidden, q_resid, (cos, sin), use_cache=False)

    ok = True
    # topk should be min(16, 4) = 4
    if indices.shape != (1, 4, 4):
        print(f"  FAIL shape: {indices.shape}, expected (1, 4, 4)")
        ok = False
    else:
        print(f"  PASS topk clamped: indices shape {indices.shape}")

    # All indices should be in [0, 4)
    if (indices >= S).any() or (indices < 0).any():
        print(f"  FAIL out-of-range indices: min={indices.min()}, max={indices.max()}")
        ok = False
    else:
        print(f"  PASS all indices in valid range [0, {S})")
    return ok


def test_moe_with_shared_expert():
    """Verify shared expert output is added to routed expert output."""
    print("\n[Test 16c] MoE shared expert contribution")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    moe = kernel_model.MoE(cfg)
    moe.eval()

    B, S, D = 1, 4, cfg["hidden_size"]
    x = torch.randn(B, S, D)

    with torch.no_grad():
        out = moe(x)

    ok = True
    if out.shape != x.shape:
        print(f"  FAIL shape: {out.shape} vs {x.shape}")
        ok = False

    # Output should NOT be zero (shared expert always contributes)
    if out.abs().max() < 1e-6:
        print("  FAIL output is all zeros (shared expert not contributing?)")
        ok = False

    if ok:
        print(f"  PASS MoE output shape {out.shape}, non-zero")
    return ok


def test_empty_expert_assignment():
    """When routing selects experts that receive zero tokens from some batch elements."""
    print("\n[Test 16d] Empty expert assignment")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashmla-deepgemm.moe_grouped_gemm")

    torch.manual_seed(42)
    N, D, I, E, K = 2, 32, 16, 8, 2

    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2 * I, D)
    down = torch.randn(E, D, I)
    # Token 0 → experts [0,1], Token 1 → experts [6,7]
    # Experts 2-5 get zero tokens
    indices = torch.tensor([[0, 1], [6, 7]])
    weights = torch.ones(N, K) * 0.5

    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)

    ok = True
    if output.shape != (N, D):
        print(f"  FAIL shape: {output.shape}")
        ok = False
    if not torch.isfinite(output).all():
        print("  FAIL non-finite output with empty experts")
        ok = False
    if ok:
        print("  PASS empty experts handled correctly")
    return ok


if __name__ == "__main__":
    results = [
        test_single_token_forward(),
        test_topk_exceeds_seq_len(),
        test_moe_with_shared_expert(),
        test_empty_expert_assignment(),
    ]
    sys.exit(0 if all(results) else 1)
