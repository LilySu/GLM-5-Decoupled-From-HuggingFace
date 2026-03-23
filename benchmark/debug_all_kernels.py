"""Debug ALL kernel APIs on the actual H100 — find exact working signatures.

Three errors to diagnose:
1. FlashMLA: "Only head_size_k == 576 or 512 is supported"
2. DeepGEMM fp8_mqa_logits: "t.dim() == N" assertion
3. DeepGEMM m_grouped_fp8_gemm: "sf.dim()" scale assertion (already diagnosed)
"""

import torch
import inspect

print("=" * 70)
print("  Kernel API Diagnostic — H100")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ════════════════════════════════════════════════════════════════════════
# 1. FlashMLA — head_size_k must be 576 or 512
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  1. FlashMLA dense decode")
print("=" * 70)

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    sig = inspect.signature(flash_mla_with_kvcache)
    print(f"Signature: {sig}")
    print()

    # The benchmark was passing d_qk as head_size_k.
    # GLM-5 absorbed: d_qk = 576 (512 nope + 64 rope)
    # FlashMLA only supports head_size_k == 576 or 512
    # Let's test with the correct absorbed dims

    B, H = 4, 64
    d_qk = 576  # absorbed: kv_lora_rank(512) + qk_rope_head_dim(64)
    d_v = 512   # absorbed: kv_lora_rank
    page_size = 64
    seq_kv = 256
    num_pages = (B * seq_kv + page_size - 1) // page_size

    # Q: [B, 1, H, d_qk]
    q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device="cuda")

    # KV cache: [num_pages, page_size, 1, d_qk] — single KV head (absorbed MLA)
    k_cache = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device="cuda")

    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).view(B, -1)
    cache_seqlens = torch.full((B,), seq_kv, dtype=torch.int32, device="cuda")

    metadata, _ = get_mla_metadata(cache_seqlens, page_size * torch.ones(1, dtype=torch.int32, device="cuda"))

    # Try the call with head_dim_v and tile_scheduler_metadata
    print(f"Trying: q={q.shape}, k_cache={k_cache.shape}, d_v={d_v}")
    out, lse = flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens,
        head_dim_v=d_v,
        tile_scheduler_metadata=metadata,
        causal=False,
    )
    print(f"SUCCESS: output={out.shape}")

except ImportError:
    print("flash_mla not installed")
except Exception as e:
    print(f"FAILED: {e}")
    print()
    # Try to figure out what args are needed
    print("Trying alternative parameter combinations...")
    try:
        # Maybe the API changed — check all param names
        print(f"Full signature: {inspect.signature(flash_mla_with_kvcache)}")
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════
# 2. DeepGEMM fp8_mqa_logits — tensor dimension assertion
# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  2. DeepGEMM fp8_mqa_logits")
print("=" * 70)

try:
    import deep_gemm
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

    sig = inspect.signature(deep_gemm.fp8_mqa_logits)
    print(f"Signature: {sig}")
    print()

    # GLM-5 DSA indexer dims
    seq_len = 1       # decode: single query
    seq_len_kv = 256  # cached tokens
    num_heads = 32    # index_n_heads
    head_dim = 128    # index_head_dim

    # The h100_bench creates these tensors:
    # q: [seq_len, num_heads, head_dim] FP8
    # kv: [seq_len_kv, head_dim] FP8 with scales
    # weights: [seq_len, num_heads] float
    # cu_k_start, cu_k_end: [seq_len] int

    q_bf16 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    kv_bf16 = torch.randn(seq_len_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
    cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
    cu_k_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device="cuda")

    # Try different quantization approaches
    print("--- Approach A: per_block_cast_to_fp8 ---")
    try:
        q_fp8 = per_block_cast_to_fp8(q_bf16.view(seq_len * num_heads, head_dim), True)
        kv_fp8 = per_block_cast_to_fp8(kv_bf16, True)
        print(f"  q: tensor={q_fp8[0].shape} scales={q_fp8[1].shape}")
        print(f"  kv: tensor={kv_fp8[0].shape} scales={kv_fp8[1].shape}")
        q_fp8_3d = (q_fp8[0].view(seq_len, num_heads, head_dim), q_fp8[1])
        deep_gemm.fp8_mqa_logits(q_fp8_3d, kv_fp8, weights, cu_k_start, cu_k_end)
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")

    print()
    print("--- Approach B: per_token_cast_to_fp8 ---")
    try:
        q_fp8 = per_token_cast_to_fp8(q_bf16.view(seq_len * num_heads, head_dim), True)
        kv_fp8 = per_token_cast_to_fp8(kv_bf16, True)
        print(f"  q: tensor={q_fp8[0].shape} scales={q_fp8[1].shape}")
        print(f"  kv: tensor={kv_fp8[0].shape} scales={kv_fp8[1].shape}")
        q_fp8_3d = (q_fp8[0].view(seq_len, num_heads, head_dim), q_fp8[1].view(seq_len, num_heads))
        deep_gemm.fp8_mqa_logits(q_fp8_3d, kv_fp8, weights, cu_k_start, cu_k_end)
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")

    print()
    print("--- Approach C: raw FP8 cast (no scales) ---")
    try:
        q_fp8_raw = q_bf16.to(torch.float8_e4m3fn)
        kv_fp8_raw = kv_bf16.to(torch.float8_e4m3fn)
        kv_scales = torch.ones(seq_len_kv, dtype=torch.float32, device="cuda")
        print(f"  q: {q_fp8_raw.shape}")
        print(f"  kv: ({kv_fp8_raw.shape}, {kv_scales.shape})")
        deep_gemm.fp8_mqa_logits(q_fp8_raw, (kv_fp8_raw, kv_scales), weights, cu_k_start, cu_k_end)
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")

    print()
    print("--- Approach D: Check what the existing dsa_indexer.py does ---")
    try:
        from deep_gemm.utils import per_custom_dims_cast_to_fp8
        # This is what dsa_indexer.py line 120-122 does:
        q_2d = q_bf16.reshape(-1, head_dim)  # [seq*heads, head_dim]
        k_2d = kv_bf16  # [seq_kv, head_dim]
        q_fp8 = q_2d.to(torch.float8_e4m3fn)
        k_fp8_tuple = per_custom_dims_cast_to_fp8(k_2d, (0,), False)
        w_2d = weights.squeeze(0) if weights.dim() > 1 else weights
        print(f"  q: {q_fp8.shape} (raw FP8, no scales)")
        print(f"  kv: tensor={k_fp8_tuple[0].shape} scales={k_fp8_tuple[1].shape}")
        print(f"  weights: {w_2d.shape}")
        print(f"  cu_k: start={cu_k_start.shape} end={cu_k_end.shape}")
        deep_gemm.fp8_mqa_logits(q_fp8, k_fp8_tuple, w_2d, cu_k_start, cu_k_end)
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")

    print()
    print("--- Approach E: per_block q with per_token kv ---")
    try:
        q_flat = q_bf16.reshape(seq_len * num_heads, head_dim)
        q_fp8 = per_block_cast_to_fp8(q_flat, True)
        kv_fp8 = per_token_cast_to_fp8(kv_bf16, True)
        print(f"  q: tensor={q_fp8[0].shape} scales={q_fp8[1].shape}")
        print(f"  kv: tensor={kv_fp8[0].shape} scales={kv_fp8[1].shape}")
        deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights.view(-1), cu_k_start, cu_k_end)
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")

except ImportError:
    print("deep_gemm not installed")

# ════════════════════════════════════════════════════════════════════════
# 3. BF16 grouped GEMM (confirmed working — just benchmark it)
# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  3. BF16 grouped GEMM (working fallback)")
print("=" * 70)

try:
    E, I, D, M = 256, 2048, 6144, 8192
    a = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(E, I, D, dtype=torch.bfloat16, device="cuda")
    d = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    layout = (torch.arange(M, device="cuda", dtype=torch.int32) % E)

    # Warmup
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 20
    flops = 2 * M * I * D
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"  {ms:.3f} ms, {tflops:.1f} TFLOPS ({tflops/989*100:.1f}% MFU)")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 70)
print("  Done")
print("=" * 70)
