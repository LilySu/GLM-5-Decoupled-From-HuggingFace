"""H100 Category 2: TMA (Tensor Memory Accelerator) Verification.

Verify TMA activity via bandwidth checks. If TMA is not active, kernels
fall back to standard loads with 20-30% perf loss and no visible error.

Requirements: H100 (SM90), flashinfer and/or deep-gemm installed.
"""

import sys
import torch
from .conftest import skip_no_sm90, has_flashinfer, has_deep_gemm


def _cuda_timer(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


@skip_no_sm90
def h100_test_tma_bandwidth_flashinfer():
    """Verify FlashInfer FA3 MLA decode achieves near-peak bandwidth (implies TMA active)."""
    print("\n[H100-TMA-1] FlashInfer FA3 bandwidth check (TMA proxy)")
    if not has_flashinfer():
        print("  SKIP flashinfer not installed")
        return True

    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    device = "cuda"
    B, H = 32, 64
    d_ckv, d_kpe = 512, 64
    seq_kv = 4096
    page_size = 1

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")

    q_nope = torch.randn(B, H, d_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_kpe, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(B * seq_kv, 1, d_ckv, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(B * seq_kv, 1, d_kpe, dtype=torch.bfloat16, device=device)

    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * seq_kv
    kv_indices = torch.arange(0, B * seq_kv, dtype=torch.int32, device=device)
    kv_lens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    sm_scale = 1.0 / ((d_ckv + d_kpe) ** 0.5)

    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens,
                 H, d_ckv, d_kpe, page_size, False, sm_scale,
                 torch.bfloat16, torch.bfloat16)

    def run():
        wrapper.run(q_nope, q_pe, ckv, kpe)

    median_ms = _cuda_timer(run, warmup=10, iters=30)

    bytes_q = B * H * (d_ckv + d_kpe) * 2
    bytes_kv = B * seq_kv * (d_ckv + d_kpe) * 2
    bytes_o = B * H * d_ckv * 2
    total_bytes = bytes_q + bytes_kv + bytes_o
    bandwidth = total_bytes / (median_ms * 1e-3) / 1e9

    threshold = 1000
    ok = bandwidth > threshold
    print(f"  Bandwidth: {bandwidth:.0f} GB/s (threshold: >{threshold} GB/s)")
    if ok:
        print(f"  PASS bandwidth suggests TMA is active")
    else:
        print(f"  FAIL bandwidth too low — TMA may not be active")
    return ok


@skip_no_sm90
def h100_test_tma_bandwidth_deepgemm():
    """Verify DeepGEMM grouped GEMM achieves high TFLOPS (implies TMA+WGMMA active)."""
    print("\n[H100-TMA-2] DeepGEMM TFLOPS check (TMA+WGMMA proxy)")
    if not has_deep_gemm():
        print("  SKIP deep_gemm not installed")
        return True

    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    device = "cuda"
    E, N, D, I = 8, 2048, 512, 128
    a = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)
    a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
    b_fp8 = per_custom_dims_cast_to_fp8(b.reshape(E * I, D), (0,), False)
    b_fp8 = (b_fp8[0].view(E, I, D), b_fp8[1].view(E, I))
    d = torch.empty(N, I, device=device, dtype=torch.bfloat16)
    layout = torch.arange(N, device=device, dtype=torch.int32) % E

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, layout)

    median_ms = _cuda_timer(run, warmup=5, iters=20)
    flops = 2 * N * D * I
    tflops = flops / (median_ms * 1e-3) / 1e12

    threshold = 50
    ok = tflops > threshold
    print(f"  TFLOPS: {tflops:.1f} (threshold: >{threshold})")
    if ok:
        print(f"  PASS TFLOPS suggests WGMMA+TMA active")
    else:
        print(f"  FAIL TFLOPS too low")
    return ok


if __name__ == "__main__":
    results = [h100_test_tma_bandwidth_flashinfer(), h100_test_tma_bandwidth_deepgemm()]
    sys.exit(0 if all(results) else 1)
