"""Test that glm5-kernels-flashinfer produces identical outputs to glm5-triton.

Run: python3 -m glm5-kernels-flashinfer.tests.test_equivalence
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg


def test_moe_router():
    print("\n[Test 1] MoE Sigmoid Router")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_router = import_module("glm5-kernels-flashinfer.moe_router")

    cfg = make_cfg()
    torch.manual_seed(42)
    num_tokens, n_experts = 16, cfg["n_routed_experts"]
    router_logits = torch.randn(num_tokens, n_experts)
    bias = torch.randn(n_experts)

    ref_moe = triton_model.MoE(cfg)
    ref_moe.gate.e_score_correction_bias.copy_(bias)
    ref_indices, ref_weights = ref_moe.route_tokens_to_experts(router_logits.clone())

    kern_indices, kern_weights = kernel_router.sigmoid_topk_route(
        router_logits.clone(), bias, top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"], topk_group=cfg["topk_group"],
        norm_topk_prob=cfg["norm_topk_prob"], routed_scaling_factor=cfg["routed_scaling_factor"],
    )

    ok = True
    for i in range(num_tokens):
        if set(ref_indices[i].tolist()) != set(kern_indices[i].tolist()):
            print(f"  FAIL token {i}: ref {set(ref_indices[i].tolist())} vs kern {set(kern_indices[i].tolist())}")
            ok = False
    if ok:
        ref_s = ref_indices.sort(dim=-1)
        kern_s = kern_indices.sort(dim=-1)
        ok = assert_close("routing_weights", ref_weights.gather(1, ref_s.indices),
                          kern_weights.gather(1, kern_s.indices), atol=1e-5)
    return ok


def test_fp8_utils():
    print("\n[Test 2] FP8 Quantization Utilities")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    ok = True
    torch.manual_seed(42)
    x = torch.randn(4, 512, dtype=torch.float32)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)
    max_rel_err = ((x - x_rt).abs() / (x.abs() + 1e-6)).max().item()
    if max_rel_err < 0.07:
        print(f"  PASS DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
    else:
        print(f"  FAIL DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
        ok = False

    # FlashInfer KV format: contiguous [pages, page_size, 576]
    ckv = torch.randn(2, 64, 512, dtype=torch.bfloat16)
    kpe = torch.randn(2, 64, 64, dtype=torch.bfloat16)
    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)
    if kv_fp8.shape == (2, 64, 576) and kv_fp8.dtype == torch.float8_e4m3fn:
        print(f"  PASS FlashInfer KV shape {kv_fp8.shape}, scale={scale:.6f}")
    else:
        print(f"  FAIL FlashInfer KV shape {kv_fp8.shape} or dtype {kv_fp8.dtype}")
        ok = False

    # Roundtrip
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale, head_dim_ckv=512)
    if ckv_rt.shape == (2, 64, 512) and kpe_rt.shape == (2, 64, 64):
        print(f"  PASS FlashInfer KV dequant shapes correct")
    else:
        print(f"  FAIL FlashInfer KV dequant shapes: ckv={ckv_rt.shape}, kpe={kpe_rt.shape}")
        ok = False

    return ok


def test_dsa_indexer():
    print("\n[Test 3] DSA Indexer (PyTorch fallback path)")
    from importlib import import_module
    triton_idx = import_module("glm5-triton.dsa_indexer")
    kernel_idx = import_module("glm5-kernels-flashinfer.dsa_indexer")
    triton_rope = import_module("glm5-triton.rope_partial")

    cfg = make_cfg()
    torch.manual_seed(42)
    ref_indexer = triton_idx.DSAIndexer(cfg, layer_idx=0)
    kern_indexer = kernel_idx.DSAIndexer(cfg, layer_idx=0)
    kern_indexer.load_state_dict(ref_indexer.state_dict())
    kern_indexer.use_deepgemm = False

    B, S, D = 2, 8, cfg["hidden_size"]
    hidden = torch.randn(B, S, D)
    q_resid = torch.randn(B, S, cfg["q_lora_rank"])
    rope_emb = triton_rope.RotaryEmbedding(cfg)
    pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = rope_emb(hidden, pos_ids)

    ref_out = ref_indexer(hidden, q_resid, (cos, sin), use_cache=False)
    kern_out = kern_indexer(hidden, q_resid, (cos, sin), use_cache=False)

    match = (ref_out.sort(dim=-1).values == kern_out.sort(dim=-1).values).all().item()
    if match:
        print(f"  PASS DSA indices match (shape={ref_out.shape})")
    else:
        print(f"  FAIL DSA indices mismatch")
    return match


def test_rmsnorm():
    print("\n[Test 4] RMSNorm (should be identical)")
    from importlib import import_module
    triton_mla = import_module("glm5-triton.mla_attention")
    kernel_mla = import_module("glm5-kernels-flashinfer.mla_attention")

    torch.manual_seed(42)
    ref_norm = triton_mla.RMSNorm(128, eps=1e-5)
    kern_norm = kernel_mla.RMSNorm(128, eps=1e-5)
    kern_norm.load_state_dict(ref_norm.state_dict())
    x = torch.randn(2, 8, 128)
    return assert_close("RMSNorm", ref_norm(x), kern_norm(x), atol=1e-6)


def test_mla_attention():
    print("\n[Test 5] MLA Attention (full forward, eager fallback)")
    from importlib import import_module
    triton_mla = import_module("glm5-triton.mla_attention")
    kernel_mla = import_module("glm5-kernels-flashinfer.mla_attention")
    triton_rope = import_module("glm5-triton.rope_partial")
    triton_model = import_module("glm5-triton.model")

    cfg = make_cfg(num_layers=1)
    torch.manual_seed(42)
    ref_attn = triton_mla.MLAttention(cfg, layer_idx=0)
    kern_attn = kernel_mla.MLAttention(cfg, layer_idx=0)
    kern_attn.load_state_dict(ref_attn.state_dict())
    kern_attn.use_flashinfer = False
    kern_attn.indexer.use_deepgemm = False

    B, S, D = 1, 8, cfg["hidden_size"]
    hidden = torch.randn(B, S, D)
    rope_emb = triton_rope.RotaryEmbedding(cfg)
    cos, sin = rope_emb(hidden, torch.arange(S).unsqueeze(0))
    mask = triton_model.make_causal_mask(S, 0, hidden.dtype, hidden.device)

    ref_out, _ = ref_attn(hidden, (cos, sin), attention_mask=mask)
    kern_out, _ = kern_attn(hidden, (cos, sin), attention_mask=mask)
    return assert_close("MLA_output", ref_out, kern_out, atol=1e-4, rtol=1e-3)


def test_full_model():
    print("\n[Test 6] Full Model Forward (2 layers)")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)
    for layer in kern.model.layers:
        layer.self_attn.use_flashinfer = False
        layer.self_attn.indexer.use_deepgemm = False
    kern.load_state_dict(ref.state_dict())

    B, S = 1, 8
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S))
    ref.eval(); kern.eval()
    with torch.no_grad():
        ref_loss, ref_logits, _ = ref(input_ids=input_ids, labels=input_ids)
        kern_loss, kern_logits, _ = kern(input_ids=input_ids, labels=input_ids)

    ok = assert_close("logits", ref_logits, kern_logits, atol=1e-3, rtol=1e-2)
    if ref_loss is not None and kern_loss is not None:
        ok = assert_close("loss", ref_loss, kern_loss, atol=1e-3) and ok
    return ok


if __name__ == "__main__":
    results = {}
    for name, fn in [("moe_router", test_moe_router), ("fp8_utils", test_fp8_utils),
                     ("dsa_indexer", test_dsa_indexer), ("rmsnorm", test_rmsnorm),
                     ("mla_attention", test_mla_attention), ("full_model", test_full_model)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)
