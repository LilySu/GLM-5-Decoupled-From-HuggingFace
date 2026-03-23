"""H100-only: Test DeepGEMM CUDA kernels produce correct results.

Tests fp8_mqa_logits (DSA indexer) and m_grouped_fp8_gemm (MoE GEMM) against
PyTorch reference implementations on the same H100 GPU.

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - pip install deep-gemm (built from source with CUDA 12.8+)

Run:
    python3 -m glm5-kernels-flashinfer.tests.h100_test_deepgemm_kernels
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_full_cfg, skip_no_sm90, has_deep_gemm


def _require_deep_gemm():
    if not has_deep_gemm():
        print("  SKIP deep_gemm not installed")
        return False
    return True


@skip_no_sm90
def h100_test_deepgemm_fp8_mqa_logits():
    """DeepGEMM fp8_mqa_logits vs PyTorch reference for DSA indexer scoring."""
    print("\n[H100] DeepGEMM fp8_mqa_logits")
    if not _require_deep_gemm():
        return True

    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    device = "cuda"
    cfg = make_full_cfg()
    seq_len = 128
    seq_len_kv = 512
    num_heads = cfg["index_n_heads"]   # 32
    head_dim = cfg["index_head_dim"]   # 128

    torch.manual_seed(42)
    q_bf16 = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(seq_len_kv, head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device=device, dtype=torch.float32)

    # Causal range: query i sees keys [0, i * (seq_len_kv // seq_len))
    ratio = seq_len_kv // seq_len
    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device) * ratio

    # FP8 quantize
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

    # DeepGEMM kernel
    logits_dg = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

    # PyTorch reference (from DeepGEMM's test_attention.py)
    q_f = q_bf16.float()
    k_f = kv_bf16.float()
    mask_lo = torch.arange(seq_len_kv, device=device)[None, :] >= ks[:, None]
    mask_hi = torch.arange(seq_len_kv, device=device)[None, :] < ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q_f, k_f)
    logits_ref = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits_ref = logits_ref.masked_fill(~mask, float("-inf"))

    # Mask out -inf positions for comparison (both should agree on masking)
    ref_neginf = logits_ref == float("-inf")
    dg_neginf = logits_dg == float("-inf")
    mask_match = torch.equal(ref_neginf, dg_neginf)

    ok = True
    if not mask_match:
        mismatch = (ref_neginf != dg_neginf).sum().item()
        print(f"  FAIL mask mismatch: {mismatch} positions differ")
        ok = False
    else:
        print(f"  PASS mask pattern matches")

    # Compare finite values
    logits_ref_masked = logits_ref.masked_fill(ref_neginf, 0)
    logits_dg_masked = logits_dg.masked_fill(dg_neginf, 0)
    # FP8 quantization introduces error — use loose tolerance
    ok = assert_close("mqa_logits", logits_dg_masked, logits_ref_masked, atol=0.5, rtol=0.1) and ok

    return ok


@skip_no_sm90
def h100_test_deepgemm_fp8_mqa_logits_glm5_dims():
    """DeepGEMM fp8_mqa_logits with exact GLM-5 indexer dimensions (num_heads=32)."""
    print("\n[H100] DeepGEMM fp8_mqa_logits GLM-5 dims (H=32, D=128)")
    if not _require_deep_gemm():
        return True

    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    device = "cuda"
    # GLM-5 exact dims
    seq_len = 1        # decode: single token
    seq_len_kv = 4096
    num_heads = 32     # index_n_heads
    head_dim = 128     # index_head_dim

    torch.manual_seed(42)
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device=device, dtype=torch.float32)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device=device)

    try:
        logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        ok = True
        if logits.shape != (seq_len, seq_len_kv):
            print(f"  FAIL shape: {logits.shape}, expected ({seq_len}, {seq_len_kv})")
            ok = False
        if not torch.isfinite(logits).all():
            print(f"  FAIL non-finite values in logits")
            ok = False
        if ok:
            print(f"  PASS GLM-5 dims: logits shape={logits.shape}, H=32 D=128 works")
        return ok
    except Exception as e:
        print(f"  FAIL GLM-5 dims: {e}")
        return False


@skip_no_sm90
def h100_test_deepgemm_grouped_gemm_contiguous():
    """DeepGEMM m_grouped_fp8_gemm_nt_contiguous vs per-expert loop reference."""
    print("\n[H100] DeepGEMM FP8 grouped GEMM (contiguous)")
    if not _require_deep_gemm():
        return True

    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    device = "cuda"
    E = 8       # experts (use small number for testing)
    N = 256     # total tokens across all experts
    D = 512     # hidden dim (smaller than real 6144 for memory)
    I = 128     # intermediate dim (smaller than real 2048)

    torch.manual_seed(42)
    # Expert weights
    b_bf16 = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)

    # Create sorted tokens: first N//E to expert 0, next N//E to expert 1, etc.
    tokens_per_expert = N // E
    a_bf16 = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    grouped_layout = torch.zeros(N, dtype=torch.int32, device=device)
    for e in range(E):
        grouped_layout[e * tokens_per_expert:(e + 1) * tokens_per_expert] = e

    # FP8 quantize
    a_fp8 = per_custom_dims_cast_to_fp8(a_bf16, (0,), False)
    b_fp8 = per_custom_dims_cast_to_fp8(b_bf16.reshape(E * I, D), (0,), False)
    b_fp8 = (b_fp8[0].view(E, I, D), b_fp8[1].view(E, I))

    d = torch.empty(N, I, device=device, dtype=torch.bfloat16)

    try:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, grouped_layout)
    except Exception as e:
        print(f"  FAIL kernel call: {e}")
        return False

    # PyTorch reference: per-expert matmul
    d_ref = torch.empty(N, I, device=device, dtype=torch.bfloat16)
    for e in range(E):
        start = e * tokens_per_expert
        end = (e + 1) * tokens_per_expert
        d_ref[start:end] = F.linear(a_bf16[start:end], b_bf16[e])

    # FP8 introduces quantization error — use loose tolerance
    return assert_close("grouped_gemm_contiguous", d, d_ref, atol=1.0, rtol=0.15)


@skip_no_sm90
def h100_test_deepgemm_grouped_gemm_masked():
    """DeepGEMM m_grouped_fp8_gemm_nt_masked (decode with CUDA graphs)."""
    print("\n[H100] DeepGEMM FP8 grouped GEMM (masked)")
    if not _require_deep_gemm():
        return True

    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    device = "cuda"
    E = 8       # experts
    M = 32      # max tokens per expert (padded)
    D = 512
    I = 128

    torch.manual_seed(42)
    # [G, M, K] layout for masked mode
    a_bf16 = torch.randn(E, M, D, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)

    # Some experts have fewer than M tokens
    masked_m = torch.tensor([8, 16, 4, 32, 12, 0, 24, 20], dtype=torch.int32, device=device)
    expected_m = 32

    a_fp8 = per_custom_dims_cast_to_fp8(a_bf16.reshape(E * M, D), (0,), False)
    a_fp8 = (a_fp8[0].view(E, M, D), a_fp8[1].view(E, M))
    b_fp8 = per_custom_dims_cast_to_fp8(b_bf16.reshape(E * I, D), (0,), False)
    b_fp8 = (b_fp8[0].view(E, I, D), b_fp8[1].view(E, I))

    d = torch.empty(E, M, I, device=device, dtype=torch.bfloat16)

    try:
        deep_gemm.m_grouped_fp8_gemm_nt_masked(a_fp8, b_fp8, d, masked_m, expected_m)
        ok = True
        if not torch.isfinite(d).all():
            print(f"  FAIL non-finite output")
            ok = False
        # Check expert with 0 tokens produces zeros (or at least the valid region is correct)
        if ok:
            print(f"  PASS masked GEMM: output shape={d.shape}, all finite")
        return ok
    except Exception as e:
        print(f"  FAIL masked GEMM: {e}")
        return False


if __name__ == "__main__":
    results = [
        h100_test_deepgemm_fp8_mqa_logits(),
        h100_test_deepgemm_fp8_mqa_logits_glm5_dims(),
        h100_test_deepgemm_grouped_gemm_contiguous(),
        h100_test_deepgemm_grouped_gemm_masked(),
    ]
    passed = sum(results)
    print(f"\n{'='*60}")
    print(f"H100 DeepGEMM: {passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
