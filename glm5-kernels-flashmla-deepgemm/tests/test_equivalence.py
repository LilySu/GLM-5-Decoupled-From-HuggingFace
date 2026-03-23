"""Test that glm5-kernels-flashmla-deepgemm produces identical outputs to glm5-triton.

Run: python3 -m glm5-kernels-flashmla-deepgemm.tests.test_equivalence
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg, PROJECT_ROOT


# ── Test 1: MoE Router ──────────────────────────────────────────────────

def test_moe_router():
    print("\n[Test 1] MoE Sigmoid Router")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    cfg = make_cfg()

    # Reference: glm5-triton router
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")

    # Kernels: our router
    from importlib import import_module as im2
    kernel_router = im2("glm5-kernels-flashmla-deepgemm.moe_router")

    torch.manual_seed(42)
    num_tokens, n_experts = 16, cfg["n_routed_experts"]
    router_logits = torch.randn(num_tokens, n_experts)
    bias = torch.randn(n_experts)

    # Reference path: instantiate MoE and call route_tokens_to_experts
    ref_moe = triton_model.MoE(cfg)
    ref_moe.gate.e_score_correction_bias.copy_(bias)
    # Override the gate forward to return our fixed logits
    ref_indices, ref_weights = ref_moe.route_tokens_to_experts(router_logits.clone())

    # Kernel path
    kern_indices, kern_weights = kernel_router.sigmoid_topk_route(
        router_logits.clone(), bias,
        top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"],
        topk_group=cfg["topk_group"],
        norm_topk_prob=cfg["norm_topk_prob"],
        routed_scaling_factor=cfg["routed_scaling_factor"],
    )

    ok = True
    # Indices may be in different order (topk sorted=False), so compare sets per token
    for i in range(num_tokens):
        ref_set = set(ref_indices[i].tolist())
        kern_set = set(kern_indices[i].tolist())
        if ref_set != kern_set:
            print(f"  FAIL token {i}: ref experts {ref_set} vs kernel experts {kern_set}")
            ok = False

    if ok:
        # Sort both by index for weight comparison
        ref_sorted_idx = ref_indices.sort(dim=-1)
        kern_sorted_idx = kern_indices.sort(dim=-1)
        ref_w_sorted = ref_weights.gather(1, ref_sorted_idx.indices)
        kern_w_sorted = kern_weights.gather(1, kern_sorted_idx.indices)
        ok = assert_close("routing_weights", ref_w_sorted, kern_w_sorted, atol=1e-5)
    else:
        print("  FAIL routing indices mismatch — skipping weight comparison")

    return ok


# ── Test 2: FP8 Quantization ────────────────────────────────────────────

def test_fp8_utils():
    print("\n[Test 2] FP8 Quantization Utilities")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashmla-deepgemm.fp8_utils")

    ok = True

    # Test roundtrip: quantize then dequantize, check error is small
    torch.manual_seed(42)
    x = torch.randn(4, 512, dtype=torch.float32)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)
    # FP8 E4M3 has 3 mantissa bits → ~6.25% worst-case relative error
    max_rel_err = ((x - x_rt).abs() / (x.abs() + 1e-6)).max().item()
    if max_rel_err < 0.07:  # 7% accounts for FP8 E4M3 precision limits
        print(f"  PASS DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
    else:
        print(f"  FAIL DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
        ok = False

    # Test FlashMLA KV cache format: check shapes and layout
    kv = torch.randn(2, 64, 1, 576, dtype=torch.bfloat16)
    try:
        quantized = fp8.quantize_kv_flashmla(kv)
        expected_bytes = 512 + 4 * 4 + 2 * 64  # 656
        if quantized.shape == (2, 64, 1, expected_bytes):
            print(f"  PASS FlashMLA KV shape {quantized.shape}")
        else:
            print(f"  FAIL FlashMLA KV shape: expected (2, 64, 1, {expected_bytes}), got {quantized.shape}")
            ok = False
    except Exception as e:
        print(f"  FAIL FlashMLA KV quantization: {e}")
        ok = False

    return ok


# ── Test 3: DSA Indexer ──────────────────────────────────────────────────

def test_dsa_indexer():
    print("\n[Test 3] DSA Indexer (PyTorch fallback path)")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    cfg = make_cfg()

    from importlib import import_module
    triton_idx = import_module("glm5-triton.dsa_indexer")
    kernel_idx = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")

    torch.manual_seed(42)

    ref_indexer = triton_idx.DSAIndexer(cfg, layer_idx=0)
    kern_indexer = kernel_idx.DSAIndexer(cfg, layer_idx=0)

    # Copy weights so they're identical
    kern_indexer.load_state_dict(ref_indexer.state_dict())

    B, S, D = 2, 8, cfg["hidden_size"]
    hidden = torch.randn(B, S, D)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"])

    # Build position embeddings
    triton_rope = import_module("glm5-triton.rope_partial")
    rope_emb = triton_rope.RotaryEmbedding(cfg)
    pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = rope_emb(hidden, pos_ids)

    # Force both to use PyTorch path (no DeepGEMM)
    kern_indexer.use_deepgemm = False

    ref_out = ref_indexer(hidden, q_resid, (cos, sin), use_cache=False)
    kern_out = kern_indexer(hidden, q_resid, (cos, sin), use_cache=False)

    ok = True
    if ref_out.shape != kern_out.shape:
        print(f"  FAIL shape: {ref_out.shape} vs {kern_out.shape}")
        ok = False
    else:
        # topk indices should be identical (same weights, same computation)
        match = (ref_out.sort(dim=-1).values == kern_out.sort(dim=-1).values).all().item()
        if match:
            print(f"  PASS DSA indices match (shape={ref_out.shape})")
        else:
            # Check how many match per token
            n_match = sum(
                set(ref_out[b, s].tolist()) == set(kern_out[b, s].tolist())
                for b in range(B) for s in range(S)
            )
            total = B * S
            print(f"  FAIL DSA indices: {n_match}/{total} tokens match")
            ok = False

    return ok


# ── Test 4: RMSNorm equivalence ─────────────────────────────────────────

def test_rmsnorm():
    print("\n[Test 4] RMSNorm (should be identical)")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    from importlib import import_module
    triton_mla = import_module("glm5-triton.mla_attention")
    kernel_mla = import_module("glm5-kernels-flashmla-deepgemm.mla_attention")

    torch.manual_seed(42)
    ref_norm = triton_mla.RMSNorm(128, eps=1e-5)
    kern_norm = kernel_mla.RMSNorm(128, eps=1e-5)
    kern_norm.load_state_dict(ref_norm.state_dict())

    x = torch.randn(2, 8, 128)
    return assert_close("RMSNorm", ref_norm(x), kern_norm(x), atol=1e-6)


# ── Test 5: Full MLA Attention ───────────────────────────────────────────

def test_mla_attention():
    print("\n[Test 5] MLA Attention (full forward, eager fallback)")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    cfg = make_cfg(num_layers=1)

    from importlib import import_module
    triton_mla = import_module("glm5-triton.mla_attention")
    kernel_mla = import_module("glm5-kernels-flashmla-deepgemm.mla_attention")
    triton_rope = import_module("glm5-triton.rope_partial")

    torch.manual_seed(42)
    ref_attn = triton_mla.MLAttention(cfg, layer_idx=0)
    kern_attn = kernel_mla.MLAttention(cfg, layer_idx=0)

    # Copy all weights
    kern_attn.load_state_dict(ref_attn.state_dict())
    # Force eager path
    kern_attn.use_flash_mla = False
    kern_attn.indexer.use_deepgemm = False

    B, S, D = 1, 8, cfg["hidden_size"]
    hidden = torch.randn(B, S, D)
    rope_emb = triton_rope.RotaryEmbedding(cfg)
    pos_ids = torch.arange(S).unsqueeze(0)
    cos, sin = rope_emb(hidden, pos_ids)

    # Build causal mask
    from importlib import import_module as im3
    triton_model = im3("glm5-triton.model")
    mask = triton_model.make_causal_mask(S, 0, hidden.dtype, hidden.device)

    ref_out, _ = ref_attn(hidden, (cos, sin), attention_mask=mask)
    kern_out, _ = kern_attn(hidden, (cos, sin), attention_mask=mask)

    return assert_close("MLA_output", ref_out, kern_out, atol=1e-4, rtol=1e-3)


# ── Test 6: Full Model Forward ──────────────────────────────────────────

def test_full_model():
    print("\n[Test 6] Full Model Forward (2 layers)")
    sys.path.insert(0, "/home/lily/wsl_git/glm5")
    cfg = make_cfg(num_layers=2)

    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    torch.manual_seed(42)
    ref_model = triton_model.GlmMoeDsaForCausalLM(cfg)

    torch.manual_seed(42)
    kern_model = kernel_model.GlmMoeDsaForCausalLM(cfg)

    # Force eager paths
    for layer in kern_model.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False

    # Copy weights to ensure identical initialization
    kern_model.load_state_dict(ref_model.state_dict())

    B, S = 1, 8
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S))

    ref_model.eval()
    kern_model.eval()
    with torch.no_grad():
        ref_loss, ref_logits, _ = ref_model(input_ids=input_ids, labels=input_ids)
        kern_loss, kern_logits, _ = kern_model(input_ids=input_ids, labels=input_ids)

    ok = assert_close("logits", ref_logits, kern_logits, atol=1e-3, rtol=1e-2)
    if ref_loss is not None and kern_loss is not None:
        ok = assert_close("loss", ref_loss, kern_loss, atol=1e-3) and ok
    return ok


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("moe_router", test_moe_router),
        ("fp8_utils", test_fp8_utils),
        ("dsa_indexer", test_dsa_indexer),
        ("rmsnorm", test_rmsnorm),
        ("mla_attention", test_mla_attention),
        ("full_model", test_full_model),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
    sys.exit(0 if all_pass else 1)
