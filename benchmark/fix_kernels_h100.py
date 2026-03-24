"""
GLM-5 Custom Kernel Fixes — H100
Demonstrates the exact tensor shapes, alignments, and API parameters 
required to run FlashMLA and DeepGEMM without CUDA asserts.
"""

import torch

def test_flashmla_fix():
    print("=" * 60)
    print(" 1. FlashMLA Dense Decode Fix")
    print("=" * 60)
    try:
        import flash_mla
        
        # FIX: Hardcode absorbed dimensions. FlashMLA strictly requires d_qk=576 or 512.
        B, H = 4, 64
        d_qk = 576  # 512 (NOPE) + 64 (RoPE)
        d_v = 512   # 512 (KV LoRA rank)
        
        page_size = 64
        seq_kv = 256
        num_pages = (B * seq_kv + page_size - 1) // page_size

        q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device="cuda")
        # Note: single KV head for absorbed MLA
        k_cache = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device="cuda")
        
        block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(B, -1)
        cache_seqlens = torch.full((B,), seq_kv, dtype=torch.int32, device="cuda")

        # FIX: Generate tile metadata and explicitly pass head_dim_v
        metadata, _ = flash_mla.get_mla_metadata(
            cache_seqlens, 
            page_size * torch.ones(1, dtype=torch.int32, device="cuda")
        )

        out, lse = flash_mla.flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens,
            head_dim_v=d_v,                   # <-- REQUIRED
            tile_scheduler_metadata=metadata, # <-- REQUIRED
            causal=False
        )
        print(f"SUCCESS! FlashMLA output shape: {out.shape}")
        
    except ImportError:
        print("SKIPPED: flash_mla not installed.")
    except Exception as e:
        print(f"FAILED: {e}")

def test_deepgemm_indexer_fix():
    print("\n" + "=" * 60)
    print(" 2. DeepGEMM DSA Indexer (fp8_mqa_logits) Fix")
    print("=" * 60)
    try:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8

        seq_len, seq_len_kv = 1, 256
        num_heads, head_dim = 32, 128

        q_bf16 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        kv_bf16 = torch.randn(seq_len_kv, head_dim, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
        
        cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
        cu_k_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device="cuda")

        # FIX: Cast Q to raw FP8 natively, and use per_token for KV with ue8m0=True
        q_fp8_raw = q_bf16.to(torch.float8_e4m3fn)
        kv_tok_tuple = per_token_cast_to_fp8(kv_bf16, use_ue8m0=True)

        deep_gemm.fp8_mqa_logits(
            q_fp8_raw, kv_tok_tuple, weights, cu_k_start, cu_k_end
        )
        print("SUCCESS! DeepGEMM fp8_mqa_logits executed cleanly.")
        
    except ImportError:
        print("SKIPPED: deep_gemm not installed.")
    except Exception as e:
        print(f"FAILED: {e}")

def test_deepgemm_moe_forward_fix():
    print("\n" + "=" * 60)
    print(" 3. DeepGEMM MoE Forward (m_grouped_fp8_gemm) Fix")
    print("=" * 60)
    try:
        import deep_gemm
        from deep_gemm.utils import (
            per_block_cast_to_fp8, 
            get_mn_major_tma_aligned_packed_ue8m0_tensor
        )

        E, I, D, M = 4, 64, 128, 32  # Scaled down dims for quick test
        
        a_bf16 = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
        b_grouped_bf16 = torch.randn(E, I, D, dtype=torch.bfloat16, device="cuda")
        
        grouped_layout = torch.arange(M, dtype=torch.int32, device="cuda") % E
        d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")

        # FIX 1: Quantize with use_ue8m0=True
        a_fp8 = per_block_cast_to_fp8(a_bf16, use_ue8m0=True)
        
        b_flat = b_grouped_bf16.reshape(E * I, D)
        b_fp8_flat = per_block_cast_to_fp8(b_flat, use_ue8m0=True)
        
        # FIX 2: Re-align scales specifically for TMA
        a_scales_tma = get_mn_major_tma_aligned_packed_ue8m0_tensor(a_fp8[1])
        
        b_scales_3d = b_fp8_flat[1].view(E, I, -1)
        b_scales_tma = get_mn_major_tma_aligned_packed_ue8m0_tensor(b_scales_3d)

        # FIX 3: Execute passing the TMA-aligned scales in the tuple
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8[0], a_scales_tma),
            (b_fp8_flat[0].view(E, I, D), b_scales_tma),
            d_out, 
            grouped_layout
        )
        print(f"SUCCESS! DeepGEMM Grouped GEMM output shape: {d_out.shape}")

    except ImportError:
        print("SKIPPED: deep_gemm not installed.")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required to run this script.")
    else:
        test_flashmla_fix()
        test_deepgemm_indexer_fix()
        test_deepgemm_moe_forward_fix()
        print("\nAll targeted kernel checks completed.")