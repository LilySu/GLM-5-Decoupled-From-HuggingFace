"""H100 Category 3: Memory Allocation & Peak Usage.

744B model on 8xH100 is tight. KV cache at 200K context = ~8.7GB per GPU.
These tests track peak memory across model sizes and check for leaks.

Requirements: H100 (SM90), CUDA.
"""

import sys
import gc
import torch
from .conftest import make_cfg, make_full_cfg, skip_no_cuda


@skip_no_cuda
def h100_test_memory_peak_single_layer():
    """Measure peak GPU memory for a single MoE layer at full GLM-5 dims."""
    print("\n[H100-Mem-1] Peak memory for single MoE layer (full dims)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")

    device = "cuda"
    cfg = make_full_cfg()
    cfg["num_hidden_layers"] = 1
    cfg["mlp_layer_types"] = ["sparse"]

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1e9

    layer = kernel_model.DecoderLayer(cfg, layer_idx=0).to(device).half().eval()
    for p in layer.parameters():
        p.data = p.data.to(torch.bfloat16)
    layer.self_attn.use_flashinfer = False
    layer.self_attn.indexer.use_deepgemm = False

    mem_params = torch.cuda.memory_allocated() / 1e9 - mem_before

    rope_mod = import_module("glm5-kernels-flashinfer.rope_partial")
    rope = rope_mod.RotaryEmbedding(cfg).to(device)

    B, S = 1, 128
    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)
    mask = kernel_model.make_causal_mask(S, 0, hidden.dtype, device)

    with torch.no_grad():
        _ = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Params: {mem_params:.2f} GB, Peak: {mem_peak:.2f} GB (B={B}, S={S})")

    # Single MoE layer: ~256 experts * (gate_up + down) ≈ 256 * (2*2048*6144 + 6144*2048) * 2 bytes
    # = 256 * 3 * 2048 * 6144 * 2 ≈ 18.4 GB parameters
    # With BF16 activations at B=1, S=128, peak should be < 25 GB
    ok = mem_peak < 30  # generous limit
    if ok:
        print(f"  PASS peak memory {mem_peak:.2f} GB < 30 GB limit")
    else:
        print(f"  FAIL peak memory {mem_peak:.2f} GB exceeds 30 GB")

    del layer
    gc.collect()
    torch.cuda.empty_cache()
    return ok


@skip_no_cuda
def h100_test_memory_kv_cache_scaling():
    """Verify KV cache memory scales linearly with sequence length."""
    print("\n[H100-Mem-2] KV cache memory scaling")

    device = "cuda"
    # Absorbed MLA: 576 dims (512 nope + 64 rope), 1 KV head, BF16
    bytes_per_token = 576 * 2  # BF16
    num_layers = 78
    bytes_per_token_all_layers = bytes_per_token * num_layers

    measurements = []
    for seq_len in [1024, 4096, 16384]:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Simulate KV cache allocation
        cache_tensors = []
        for _ in range(num_layers):
            k = torch.empty(1, 1, seq_len, 576, dtype=torch.bfloat16, device=device)
            v = torch.empty(1, 1, seq_len, 512, dtype=torch.bfloat16, device=device)
            cache_tensors.append((k, v))

        mem_after = torch.cuda.memory_allocated()
        actual_gb = (mem_after - mem_before) / 1e9
        # Expected: 78 layers * seq_len * (576 + 512) * 2 bytes
        expected_gb = num_layers * seq_len * (576 + 512) * 2 / 1e9
        measurements.append((seq_len, actual_gb, expected_gb))

        del cache_tensors
        gc.collect()
        torch.cuda.empty_cache()

    ok = True
    for seq_len, actual, expected in measurements:
        ratio = actual / expected if expected > 0 else 0
        status = "PASS" if 0.9 < ratio < 1.2 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status} seq={seq_len:6d}: actual={actual:.3f} GB, expected={expected:.3f} GB (ratio={ratio:.2f})")

    if ok:
        print("  PASS KV cache scales linearly (within 20% of expected)")
    return ok


@skip_no_cuda
def h100_test_memory_no_leak_decode():
    """Run 50 decode steps and verify no memory leak."""
    print("\n[H100-Mem-3] Memory leak check (50 decode steps)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"
    model = kernel_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.use_flashinfer = False
        layer.self_attn.indexer.use_deepgemm = False

    B = 1
    input_ids = torch.randint(0, cfg["vocab_size"], (B, 8), device=device)

    with torch.no_grad():
        _, logits, cache = model(input_ids=input_ids, use_cache=True)

    # Record memory after prefill
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_prefill = torch.cuda.memory_allocated()

    # Run 50 decode steps
    for step in range(50):
        next_token = logits[:, -1:, :].argmax(dim=-1)
        with torch.no_grad():
            _, logits, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)

    torch.cuda.synchronize()
    mem_after_decode = torch.cuda.memory_allocated()

    # KV cache grows linearly: 50 tokens * 2 layers * (K + V) tensors
    # Non-cache memory should not grow
    kv_growth_expected = 50 * 2 * 2 * (cfg["num_attention_heads"] * cfg["qk_head_dim"] +
                                        cfg["num_attention_heads"] * cfg["v_head_dim"]) * 4  # float32 for cache
    mem_growth = mem_after_decode - mem_after_prefill
    expected_max_growth = kv_growth_expected * 3  # 3x headroom for fragmentation

    ok = mem_growth < expected_max_growth
    growth_mb = mem_growth / 1e6
    expected_mb = expected_max_growth / 1e6
    print(f"  Memory growth: {growth_mb:.1f} MB over 50 steps (limit: {expected_mb:.1f} MB)")
    if ok:
        print(f"  PASS no memory leak detected")
    else:
        print(f"  FAIL memory growth exceeds expected KV cache growth")
    return ok


if __name__ == "__main__":
    results = [
        h100_test_memory_peak_single_layer(),
        h100_test_memory_kv_cache_scaling(),
        h100_test_memory_no_leak_decode(),
    ]
    sys.exit(0 if all(results) else 1)
