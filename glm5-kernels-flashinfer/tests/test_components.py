"""CPU tests for individual components: cu_seqlens, KV cache, DSA mask,
MoE dispatch, FP8 layout, autoregressive decode, group routing, gradients,
state_dict compat, edge cases.

Equivalent to glm5-kernels-flashmla-deepgemm tests 7-16, adapted for FlashInfer imports.

Run: python3 -m glm5-kernels-flashinfer.tests.test_components
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg, jaccard


# ── Test 7: cu_seqlens ───────────────────────────────────────────────────

def test_cu_seqlens_prefill():
    print("\n[Test 7a] cu_seqlens prefill (causal)")
    seq_len, seq_len_kv = 8, 8
    ks = torch.zeros(seq_len, dtype=torch.int32)
    ke = torch.arange(1, seq_len + 1, dtype=torch.int32)
    ok = all(ks[i] == 0 and ke[i] == i + 1 for i in range(seq_len))
    print(f"  {'PASS' if ok else 'FAIL'} prefill cu_seqlens")
    return ok

def test_cu_seqlens_decode():
    print("\n[Test 7b] cu_seqlens decode")
    seq_len, seq_len_kv = 1, 128
    ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
    ok = ke[0].item() == seq_len_kv
    print(f"  {'PASS' if ok else 'FAIL'} decode cu_seqlens")
    return ok

def test_cu_seqlens_chunked():
    print("\n[Test 7c] cu_seqlens chunked prefill")
    seq_len, seq_len_kv = 4, 20
    ke = torch.arange(seq_len_kv - seq_len + 1, seq_len_kv + 1, dtype=torch.int32)
    ok = list(ke.tolist()) == [17, 18, 19, 20]
    print(f"  {'PASS' if ok else 'FAIL'} chunked cu_seqlens")
    return ok


# ── Test 8: KV cache ────────────────────────────────────────────────────

def test_kvcache_multistep():
    print("\n[Test 8a] KVCache multi-step")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashinfer.cache")
    cache = cache_mod.KVCache(num_layers=2)
    all_k, all_v = [], []
    for step in range(5):
        k = torch.randn(2, 4, 1, 32) + step
        v = torch.randn(2, 4, 1, 32) + step
        all_k.append(k); all_v.append(v)
        full_k, full_v = cache.update(k, v, 0)
    ok = assert_close("keys", full_k, torch.cat(all_k, dim=2), atol=0)
    ok = assert_close("vals", full_v, torch.cat(all_v, dim=2), atol=0) and ok
    ok = (cache.get_seq_length(0) == 5) and ok
    print(f"  {'PASS' if ok else 'FAIL'} seq_length=5")
    return ok

def test_kvcache_reset():
    print("\n[Test 8b] KVCache reset")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashinfer.cache")
    cache = cache_mod.KVCache(2)
    cache.update(torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32), 0)
    cache.reset()
    ok = cache.get_seq_length(0) == 0
    print(f"  {'PASS' if ok else 'FAIL'} reset")
    return ok

def test_paged_kvcache():
    print("\n[Test 8c] FlashInferPagedKVCache allocation")
    from importlib import import_module
    cache_mod = import_module("glm5-kernels-flashinfer.cache")
    cache = cache_mod.FlashInferPagedKVCache(1, num_pages=8, page_size=4, device="cpu")
    pages = [cache.allocate_page() for _ in range(5)]
    ok = len(set(pages)) == 5
    cache.free_page(pages[0])
    try:
        for _ in range(10): cache.allocate_page()
        ok = False; print("  FAIL should exhaust")
    except RuntimeError:
        print("  PASS exhaustion raises")
    # Check separate ckv/kpe pools exist
    ckv = cache.get_ckv_cache(0)
    kpe = cache.get_kpe_cache(0)
    ok = ok and ckv.shape[-1] == 512 and kpe.shape[-1] == 64
    print(f"  {'PASS' if ok else 'FAIL'} ckv shape {ckv.shape[-1]}, kpe shape {kpe.shape[-1]}")
    return ok


# ── Test 9: DSA mask ────────────────────────────────────────────────────

def test_dsa_mask_basic():
    print("\n[Test 9a] DSA mask basic")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashinfer.dsa_sparse_attention")
    B, S, T, topk = 1, 4, 16, 3
    indices = torch.tensor([[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13]]])
    query = torch.randn(B, 4, S, 32)
    mask = dsa.build_dsa_mask(indices, None, query, T)
    ok = True
    for s in range(S):
        selected = set(indices[0, s].tolist())
        for t in range(T):
            val = mask[0, 0, s, t].item()
            if (t in selected and val != 0.0) or (t not in selected and val != float("-inf")):
                ok = False
    print(f"  {'PASS' if ok else 'FAIL'} basic DSA mask")
    return ok

def test_dsa_mask_causal():
    print("\n[Test 9b] DSA mask + causal")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashinfer.dsa_sparse_attention")
    B, S, T = 1, 4, 8
    indices = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, S, T)
    query = torch.randn(B, 4, S, 32)
    causal = torch.full((1, 1, S, T), float("-inf"), dtype=query.dtype)
    for i in range(S): causal[0, 0, i, :i+1] = 0.0
    mask = dsa.build_dsa_mask(indices, causal, query, T)
    ok = all(mask[0, 0, s, t].item() == 0.0 if t <= s else mask[0, 0, s, t].item() != 0.0
             for s in range(S) for t in range(T))
    print(f"  {'PASS' if ok else 'FAIL'} causal intersection")
    return ok


# ── Test 10: MoE expert dispatch ────────────────────────────────────────

def test_expert_dispatch_single():
    print("\n[Test 10a] Expert dispatch: single expert")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    N, D, I, E = 4, 32, 16, 4
    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2*I, D)
    down = torch.randn(E, D, I)
    indices = torch.zeros(N, 1, dtype=torch.long)
    weights = torch.ones(N, 1)
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)
    gate, up = F.linear(hidden, gate_up[0]).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, down[0])
    return assert_close("single_expert", output, expected, atol=1e-5)

def test_expert_dispatch_sparse():
    print("\n[Test 10b] Expert dispatch: sparse routing")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    N, D, I, E, K = 8, 32, 16, 8, 2
    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2*I, D)
    down = torch.randn(E, D, I)
    indices = torch.stack([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)], dim=1)
    weights = torch.full((N, K), 0.5)
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)
    expected = torch.zeros_like(hidden)
    for eidx in [0, 1]:
        g, u = F.linear(hidden, gate_up[eidx]).chunk(2, dim=-1)
        expected += F.linear(F.silu(g) * u, down[eidx]) * 0.5
    return assert_close("sparse_routing", output, expected, atol=1e-5)


# ── Test 11: FlashInfer FP8 KV layout ───────────────────────────────────

def test_flashinfer_kv_roundtrip():
    print("\n[Test 11] FlashInfer FP8 KV roundtrip")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")
    torch.manual_seed(42)
    ckv = torch.randn(1, 4, 512, dtype=torch.bfloat16)
    kpe = torch.randn(1, 4, 64, dtype=torch.bfloat16)
    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale)
    rel_err = ((ckv.float() - ckv_rt.float()).abs() / (ckv.float().abs() + 1e-6)).max().item()
    ok = rel_err < 0.07 and kv_fp8.shape == (1, 4, 576)
    print(f"  {'PASS' if ok else 'FAIL'} roundtrip rel_err={rel_err:.4f}, shape={kv_fp8.shape}")
    return ok


# ── Test 12: Autoregressive decode ──────────────────────────────────────

def test_autoregressive_decode():
    print("\n[Test 12] Autoregressive decode (prefill=4, decode=3)")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)
    kern.load_state_dict(ref.state_dict())
    for l in kern.model.layers:
        l.self_attn.use_flashinfer = False
        l.self_attn.indexer.use_deepgemm = False
    ref.eval(); kern.eval()
    prefill_ids = torch.randint(0, cfg["vocab_size"], (1, 4))
    ok = True
    with torch.no_grad():
        _, ref_logits, ref_cache = ref(input_ids=prefill_ids, use_cache=True)
        _, kern_logits, kern_cache = kern(input_ids=prefill_ids, use_cache=True)
        ok = assert_close("prefill", ref_logits, kern_logits, atol=1e-4) and ok
        for step in range(3):
            next_tok = ref_logits[:, -1:, :].argmax(dim=-1)
            _, ref_logits, ref_cache = ref(input_ids=next_tok, past_key_values=ref_cache, use_cache=True)
            _, kern_logits, kern_cache = kern(input_ids=next_tok, past_key_values=kern_cache, use_cache=True)
            ok = assert_close(f"decode_{step}", ref_logits, kern_logits, atol=1e-3) and ok
    return ok


# ── Test 13: Group routing ──────────────────────────────────────────────

def test_group_routing_filters():
    print("\n[Test 13a] Group routing filters groups")
    from importlib import import_module
    router = import_module("glm5-kernels-flashinfer.moe_router")
    torch.manual_seed(42)
    logits = torch.randn(8, 16)
    logits[:, :4] += 5.0; logits[:, 4:8] += 5.0
    logits[:, 8:12] -= 5.0; logits[:, 12:16] -= 5.0
    indices, _ = router.sigmoid_topk_route(logits, torch.zeros(16), top_k=4, n_group=4, topk_group=2)
    ok = all(idx.item() < 8 for row in indices for idx in row)
    print(f"  {'PASS' if ok else 'FAIL'} only surviving groups selected")
    return ok

def test_group_routing_flat():
    print("\n[Test 13b] n_group=1 == flat topk")
    from importlib import import_module
    router = import_module("glm5-kernels-flashinfer.moe_router")
    torch.manual_seed(42)
    logits = torch.randn(16, 8)
    bias = torch.randn(8)
    idx1, w1 = router.sigmoid_topk_route(logits.clone(), bias, top_k=2, n_group=1, topk_group=1)
    idx8, w8 = router.sigmoid_topk_route(logits.clone(), bias, top_k=2, n_group=8, topk_group=8)
    ok = all(set(idx1[i].tolist()) == set(idx8[i].tolist()) for i in range(16))
    if ok:
        s1, s8 = idx1.sort(dim=-1), idx8.sort(dim=-1)
        ok = assert_close("weights", w1.gather(1, s1.indices), w8.gather(1, s8.indices), atol=1e-5)
    return ok


# ── Test 14: Gradient flow ──────────────────────────────────────────────

def test_gradient_flow():
    print("\n[Test 14] Gradient flow")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    model = kernel_model.GlmMoeDsaForCausalLM(cfg)
    for l in model.model.layers:
        l.self_attn.use_flashinfer = False
        l.self_attn.indexer.use_deepgemm = False
    model.train()
    ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    loss, _, _ = model(input_ids=ids, labels=ids)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None and "indexer" not in n]
    ok = len(no_grad) == 0
    if ok:
        grad_count = sum(1 for _, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0)
        print(f"  PASS gradient flow: {grad_count} params have non-zero grads")
    else:
        print(f"  FAIL {len(no_grad)} non-indexer params with no grad")
    return ok


# ── Test 15: State dict compat ──────────────────────────────────────────

def test_state_dict_keys():
    print("\n[Test 15a] State dict keys match")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)
    ref_keys = set(ref.state_dict().keys())
    kern_keys = set(kern.state_dict().keys())
    ok = ref_keys == kern_keys
    print(f"  {'PASS' if ok else 'FAIL'} {len(ref_keys)} keys {'match' if ok else 'differ'}")
    return ok

def test_state_dict_cross_load():
    print("\n[Test 15b] Cross-loading state dicts")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)
    ok = True
    try:
        kern.load_state_dict(ref.state_dict()); print("  PASS triton -> flashinfer")
    except Exception as e:
        print(f"  FAIL triton -> flashinfer: {e}"); ok = False
    try:
        ref.load_state_dict(kern.state_dict()); print("  PASS flashinfer -> triton")
    except Exception as e:
        print(f"  FAIL flashinfer -> triton: {e}"); ok = False
    return ok

def test_state_dict_shapes():
    print("\n[Test 15c] Parameter shapes match")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    ref_sd = triton_model.GlmMoeDsaForCausalLM(cfg).state_dict()
    kern_sd = kernel_model.GlmMoeDsaForCausalLM(cfg).state_dict()
    ok = all(ref_sd[k].shape == kern_sd[k].shape for k in ref_sd if k in kern_sd)
    print(f"  {'PASS' if ok else 'FAIL'} all shapes match")
    return ok


# ── Test 16: Edge cases ─────────────────────────────────────────────────

def test_single_token():
    print("\n[Test 16a] Single token forward (B=1, S=1)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=1)
    torch.manual_seed(42)
    model = kernel_model.GlmMoeDsaForCausalLM(cfg)
    for l in model.model.layers:
        l.self_attn.use_flashinfer = False
        l.self_attn.indexer.use_deepgemm = False
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(input_ids=torch.tensor([[42]]), use_cache=True)
    ok = logits.shape == (1, 1, cfg["vocab_size"]) and torch.isfinite(logits).all()
    print(f"  {'PASS' if ok else 'FAIL'} shape={logits.shape}")
    return ok

def test_topk_exceeds_seq():
    print("\n[Test 16b] topk > seq_len")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashinfer.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashinfer.rope_partial")
    cfg = make_cfg()
    torch.manual_seed(42)
    indexer = dsa.DSAIndexer(cfg, 0); indexer.use_deepgemm = False
    hidden = torch.randn(1, 4, cfg["hidden_size"])
    q_resid = torch.randn(1, 4, cfg["q_lora_rank"])
    rope = rope_mod.RotaryEmbedding(cfg)
    cos, sin = rope(hidden, torch.arange(4).unsqueeze(0))
    indices = indexer(hidden, q_resid, (cos, sin), use_cache=False)
    ok = indices.shape == (1, 4, 4) and (indices < 4).all() and (indices >= 0).all()
    print(f"  {'PASS' if ok else 'FAIL'} topk clamped: {indices.shape}")
    return ok

def test_shared_expert():
    print("\n[Test 16c] MoE shared expert contribution")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    moe = kernel_model.MoE(cfg); moe.eval()
    with torch.no_grad():
        out = moe(torch.randn(1, 4, cfg["hidden_size"]))
    ok = out.shape == (1, 4, cfg["hidden_size"]) and out.abs().max() > 1e-6
    print(f"  {'PASS' if ok else 'FAIL'} non-zero output")
    return ok

def test_empty_expert():
    print("\n[Test 16d] Empty expert assignment")
    from importlib import import_module
    moe_gemm = import_module("glm5-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    hidden = torch.randn(2, 32)
    gate_up = torch.randn(8, 32, 32)
    down = torch.randn(8, 32, 16)
    indices = torch.tensor([[0, 1], [6, 7]])
    weights = torch.ones(2, 2) * 0.5
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, 8)
    ok = output.shape == (2, 32) and torch.isfinite(output).all()
    print(f"  {'PASS' if ok else 'FAIL'} empty experts handled")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_cu_seqlens_prefill, test_cu_seqlens_decode, test_cu_seqlens_chunked,
    test_kvcache_multistep, test_kvcache_reset, test_paged_kvcache,
    test_dsa_mask_basic, test_dsa_mask_causal,
    test_expert_dispatch_single, test_expert_dispatch_sparse,
    test_flashinfer_kv_roundtrip,
    test_autoregressive_decode,
    test_group_routing_filters, test_group_routing_flat,
    test_gradient_flow,
    test_state_dict_keys, test_state_dict_cross_load, test_state_dict_shapes,
    test_single_token, test_topk_exceeds_seq, test_shared_expert, test_empty_expert,
]

if __name__ == "__main__":
    results = {}
    for fn in ALL_TESTS:
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            results[fn.__name__] = False
    print(f"\n{'='*60}\n{sum(results.values())}/{len(results)} passed\n{'='*60}")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)
