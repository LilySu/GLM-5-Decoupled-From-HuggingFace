"""Test 11: FlashMLA FP8 KV cache byte layout verification.

Verifies the interleaved 656-byte layout: nope[0:512], scales[512:528], rope[528:656].
Also tests that the rope portion survives quantization losslessly (stored as BF16).
"""

import sys
import torch
from .conftest import assert_close


def test_flashmla_kv_byte_layout():
    """Verify the 656-byte per-token layout matches FlashMLA spec."""
    print("\n[Test 11a] FlashMLA KV byte layout")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashmla-deepgemm.fp8_utils")

    torch.manual_seed(42)
    kv = torch.randn(1, 4, 1, 576, dtype=torch.bfloat16)  # 1 block, 4 tokens
    quantized = fp8.quantize_kv_flashmla(kv)

    ok = True
    # Shape check
    if quantized.shape != (1, 4, 1, 656):
        print(f"  FAIL shape: {quantized.shape}, expected (1, 4, 1, 656)")
        return False

    # Reinterpret as raw bytes for each token
    raw = quantized.squeeze(2)  # [1, 4, 656]

    # Region 1: nope portion [0:512] — FP8 values
    nope_fp8 = raw[..., :512]
    if nope_fp8.dtype != torch.float8_e4m3fn:
        print(f"  FAIL nope dtype: {nope_fp8.dtype}")
        ok = False
    else:
        print("  PASS nope region [0:512] is FP8")

    # Region 2: scales [512:528] — 4 x float32 = 16 bytes
    scales_raw = raw[..., 512:528].view(torch.float32)
    if scales_raw.shape[-1] != 4:
        print(f"  FAIL scales shape: {scales_raw.shape}")
        ok = False
    else:
        # Scales should be positive powers of 2
        for i in range(4):
            s = scales_raw[0, 0, i].item()
            if s <= 0:
                print(f"  FAIL scale[{i}] = {s} (not positive)")
                ok = False
            # Check power of 2: log2 should be integer
            import math
            log2_s = math.log2(s)
            if abs(log2_s - round(log2_s)) > 1e-6:
                print(f"  FAIL scale[{i}] = {s} (not power of 2, log2={log2_s})")
                ok = False
        if ok:
            print("  PASS scales region [512:528] are positive powers of 2")

    # Region 3: rope [528:656] — 64 x BF16 = 128 bytes, should match input exactly
    rope_raw = raw[..., 528:656].view(torch.bfloat16)
    rope_original = kv[0, :, 0, 512:]  # [4, 64]
    if rope_raw.shape[-1] != 64:
        print(f"  FAIL rope shape: {rope_raw.shape}")
        ok = False
    else:
        rope_match = torch.equal(rope_raw[0], rope_original)
        if rope_match:
            print("  PASS rope region [528:656] matches input exactly (lossless BF16)")
        else:
            max_diff = (rope_raw[0].float() - rope_original.float()).abs().max().item()
            print(f"  FAIL rope mismatch (max_diff={max_diff})")
            ok = False

    return ok


def test_fp8_nope_roundtrip():
    """Quantize nope portion and verify dequantized values are within FP8 tolerance."""
    print("\n[Test 11b] FP8 nope roundtrip accuracy")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashmla-deepgemm.fp8_utils")

    torch.manual_seed(42)
    nope_data = torch.randn(1, 4, 1, 576, dtype=torch.bfloat16)
    quantized = fp8.quantize_kv_flashmla(nope_data)
    raw = quantized.squeeze(2)  # [1, 4, 656]

    # Extract nope FP8 and scales
    nope_fp8 = raw[..., :512].float()
    scales = raw[..., 512:528].view(torch.float32)  # [1, 4, 4]

    # Dequantize: each tile of 128 values uses its scale
    dequantized = torch.zeros(1, 4, 512)
    for tile in range(4):
        tile_data = nope_fp8[..., tile * 128:(tile + 1) * 128]
        tile_scale = scales[..., tile:tile + 1]
        dequantized[..., tile * 128:(tile + 1) * 128] = tile_data * tile_scale

    original = nope_data[0, :, 0, :512].float()
    rel_err = ((dequantized[0] - original).abs() / (original.abs() + 1e-6)).max().item()

    if rel_err < 0.07:
        print(f"  PASS nope roundtrip (max_rel_err={rel_err:.4f})")
        return True
    else:
        print(f"  FAIL nope roundtrip (max_rel_err={rel_err:.4f})")
        return False


if __name__ == "__main__":
    results = [test_flashmla_kv_byte_layout(), test_fp8_nope_roundtrip()]
    sys.exit(0 if all(results) else 1)
