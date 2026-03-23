"""H100 Category 4: FP8 Numeric Edge Cases.

E4M3 max=448, min_subnormal≈0.001953. Outlier activations can overflow
silently. Per-group scaling is critical to prevent this. These tests
exercise pathological input distributions.
"""

import sys
import torch
from .conftest import assert_close, skip_no_cuda


@skip_no_cuda
def h100_test_fp8_overflow_detection():
    """Values > 448 should be clamped to 448 by per-block scaling, not silently overflow."""
    print("\n[H100-FP8-1] Overflow detection (outliers > 448)")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    device = "cuda"
    # Normal data with extreme outliers
    x = torch.randn(4, 512, device=device) * 0.1
    x[0, 0] = 1000.0    # massive outlier
    x[1, 100] = -500.0   # negative outlier
    x[2, 255] = 448.1    # just above E4M3 max

    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    ok = True
    # Roundtrip should preserve outliers approximately (scaling handles range)
    if not torch.isfinite(x_rt).all():
        print("  FAIL non-finite values after FP8 roundtrip with outliers")
        ok = False

    # The outlier at [0,0]=1000 should survive with ~7% error (per-block scaling)
    outlier_err = abs(x_rt[0, 0].item() - 1000.0) / 1000.0
    if outlier_err > 0.1:  # 10% tolerance for extreme outlier
        print(f"  FAIL outlier error {outlier_err:.3f} > 0.10 for x=1000")
        ok = False
    else:
        print(f"  PASS outlier x=1000 roundtrip error={outlier_err:.3f}")

    # Negative outlier
    neg_err = abs(x_rt[1, 100].item() - (-500.0)) / 500.0
    if neg_err > 0.1:
        print(f"  FAIL negative outlier error {neg_err:.3f}")
        ok = False
    else:
        print(f"  PASS negative outlier x=-500 roundtrip error={neg_err:.3f}")

    return ok


@skip_no_cuda
def h100_test_fp8_zero_handling():
    """All-zero blocks should not produce NaN/Inf scales."""
    print("\n[H100-FP8-2] Zero block handling")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    device = "cuda"
    x = torch.zeros(2, 256, device=device)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    ok = True
    if not torch.isfinite(scales).all():
        print(f"  FAIL scales contain non-finite values for zero input")
        ok = False
    if not torch.isfinite(x_rt).all():
        print(f"  FAIL roundtrip contains non-finite values for zero input")
        ok = False
    if x_rt.abs().max() > 0.01:
        print(f"  FAIL zero input roundtrip max={x_rt.abs().max():.6f}")
        ok = False

    if ok:
        print("  PASS zero blocks produce finite scales and near-zero roundtrip")
    return ok


@skip_no_cuda
def h100_test_fp8_subnormal_precision():
    """Very small values (near E4M3 subnormal range) should not vanish entirely."""
    print("\n[H100-FP8-3] Subnormal precision")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    device = "cuda"
    # All values near 1e-3 (close to E4M3 subnormal)
    x = torch.full((1, 128), 1e-3, device=device)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)

    # With per-block scaling, amax=1e-3 → scale=1e-3/448≈2.2e-6
    # After quantize: 1e-3 / 2.2e-6 ≈ 448 → representable in E4M3
    # So values should survive even though they're tiny
    rel_err = ((x - x_rt).abs() / (x.abs() + 1e-12)).max().item()
    ok = rel_err < 0.1  # 10% tolerance
    if ok:
        print(f"  PASS subnormal preservation: rel_err={rel_err:.4f}")
    else:
        print(f"  FAIL subnormal lost: rel_err={rel_err:.4f}")
    return ok


@skip_no_cuda
def h100_test_fp8_flashinfer_kv_scale_correctness():
    """FlashInfer KV global scale should correctly bracket the data range."""
    print("\n[H100-FP8-4] FlashInfer KV global scale correctness")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)
    # Create KV with mixed value ranges
    ckv = torch.zeros(1, 4, 512, dtype=torch.bfloat16, device=device)
    kpe = torch.zeros(1, 4, 64, dtype=torch.bfloat16, device=device)
    ckv[0, :, 0:128] = 100.0
    ckv[0, :, 128:256] = 0.01
    ckv[0, :, 256:384] = 1.0
    ckv[0, :, 384:512] = -50.0
    kpe[0, :, :] = 2.0

    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)

    ok = True
    # Scale should bracket the global max: amax / 448
    amax = max(ckv.abs().max().item(), kpe.abs().max().item())
    expected_scale = amax / 448.0
    if abs(scale - expected_scale) / expected_scale > 0.01:
        print(f"  FAIL scale={scale:.6f}, expected={expected_scale:.6f}")
        ok = False

    # All quantized values should be in E4M3 range [-448, 448]
    kv_float = kv_fp8.float()
    if kv_float.abs().max() > 448.5:
        print(f"  FAIL quantized max={kv_float.abs().max():.1f} > 448")
        ok = False

    # Shape should be contiguous [pages, page_size, 576]
    if kv_fp8.shape != (1, 4, 576):
        print(f"  FAIL shape={kv_fp8.shape}, expected (1, 4, 576)")
        ok = False

    # Roundtrip accuracy
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale)
    rel_err = ((ckv.float() - ckv_rt.float()).abs() / (ckv.float().abs() + 1e-6)).max().item()
    if rel_err > 0.07:
        print(f"  FAIL roundtrip rel_err={rel_err:.4f} > 0.07")
        ok = False

    if ok:
        print(f"  PASS scale={scale:.6f}, shape={kv_fp8.shape}, roundtrip rel_err={rel_err:.4f}")
    return ok


if __name__ == "__main__":
    results = [
        h100_test_fp8_overflow_detection(),
        h100_test_fp8_zero_handling(),
        h100_test_fp8_subnormal_precision(),
        h100_test_fp8_flashinfer_kv_scale_correctness(),
    ]
    sys.exit(0 if all(results) else 1)
