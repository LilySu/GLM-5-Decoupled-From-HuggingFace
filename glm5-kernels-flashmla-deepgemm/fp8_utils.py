# FP8 quantization utilities for FlashMLA and DeepGEMM.
#
# Two different FP8 formats are needed:
#
# 1. FlashMLA KV cache format (V32_FP8Sparse):
#    Per token: [512 FP8 nope][4×float32 scales][64 BF16 rope] = 656 bytes
#    Scales are per-128-dim-tile, power-of-2 (UE8M0).
#
# 2. DeepGEMM activation format:
#    (tensor, scales) pair where scales are per-token or per-block float32.
#
# Sources:
#   FlashMLA: tests/quant.py quantize_k_cache()
#   DeepGEMM: deep_gemm.utils.per_custom_dims_cast_to_fp8()

import torch


# ---------------------------------------------------------------------------
# FlashMLA FP8 KV cache format
# ---------------------------------------------------------------------------

def quantize_kv_flashmla(
    kv: torch.Tensor,  # [num_blocks, block_size, 1, d] where d=576 (512 nope + 64 rope)
) -> torch.Tensor:
    """Quantize KV cache into FlashMLA's V32 FP8 sparse format.

    Layout per token (656 bytes):
        [0:512]   — 512 × FP8 E4M3 (nope portion, quantized in 4 tiles of 128)
        [512:528] — 4 × float32 (scale factors, one per 128-dim tile)
        [528:656] — 64 × BF16 (rope portion, unquantized)

    Args:
        kv: [num_blocks, block_size, 1, 576] in BF16

    Returns:
        quantized: [num_blocks, block_size, 1, 656] as FP8 view (raw bytes)
    """
    d_nope, d_rope = 512, 64
    tile_size, num_tiles = 128, 4

    num_blocks, block_size, h_k, d = kv.shape
    assert d == d_nope + d_rope and h_k == 1
    kv = kv.squeeze(2)  # [num_blocks, block_size, 576]

    bytes_per_token = d_nope + num_tiles * 4 + kv.element_size() * d_rope  # 512 + 16 + 128 = 656
    result = torch.empty(
        (num_blocks, block_size + 1, bytes_per_token),
        dtype=torch.float8_e4m3fn, device=kv.device,
    )[:, :block_size, :]

    # Nope portion (FP8 quantized in 4 tiles of 128 dims)
    result_nope = result[..., :d_nope]
    result_scales = result[..., d_nope:d_nope + num_tiles * 4].view(torch.float32)
    result_rope = result[..., d_nope + num_tiles * 4:].view(kv.dtype)

    # Rope portion stays in BF16
    result_rope[:] = kv[..., d_nope:]

    # Quantize nope in 128-dim tiles with power-of-2 scales
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = tile_start + tile_size
        tile_data = kv[..., tile_start:tile_end]

        # Per-token amax, power-of-2 scale (UE8M0)
        amax = tile_data.abs().float().amax(dim=-1).clamp(min=1e-4)
        scale_inv = amax / 448.0
        scale_inv = torch.pow(2, scale_inv.log2().ceil())  # Round to power of 2
        result_scales[:, :, tile_idx] = scale_inv

        # Quantize
        quantized = (tile_data.float() / scale_inv.unsqueeze(-1).float()).to(torch.float8_e4m3fn)
        result_nope[..., tile_start:tile_end] = quantized

    return result.view(num_blocks, block_size, 1, -1)


# ---------------------------------------------------------------------------
# DeepGEMM FP8 activation format
# ---------------------------------------------------------------------------

def quantize_activations_deepgemm(
    x: torch.Tensor,  # [*, D]
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 E4M3 with per-block scaling for DeepGEMM.

    Args:
        x: Input tensor in BF16/FP32, last dim is the quantization axis
        block_size: Number of elements per scale factor (default 128)

    Returns:
        (x_fp8, scales): x_fp8 same shape as x in FP8, scales per block
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    flat = x.reshape(-1, d).float()
    m = flat.shape[0]

    num_blocks_per_row = (d + block_size - 1) // block_size
    # Pad to block boundary
    if d % block_size != 0:
        pad = block_size - (d % block_size)
        flat = torch.nn.functional.pad(flat, (0, pad))

    # Reshape into blocks: [m, num_blocks, block_size]
    flat_blocked = flat.reshape(m, num_blocks_per_row, block_size)
    amax = flat_blocked.abs().amax(dim=-1).clamp(min=1e-4)  # [m, num_blocks]
    scales = amax / 448.0  # [m, num_blocks]

    # Quantize
    x_scaled = flat_blocked / scales.unsqueeze(-1)
    x_fp8 = x_scaled.reshape(m, -1)[:, :d].to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(orig_shape)

    return x_fp8, scales


def dequantize_fp8(x_fp8: torch.Tensor, scales: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8 tensor back to BF16."""
    orig_shape = x_fp8.shape
    d = orig_shape[-1]
    flat = x_fp8.reshape(-1, d).float()
    m = flat.shape[0]

    num_blocks = scales.shape[-1]
    if d % block_size != 0:
        pad = block_size - (d % block_size)
        flat = torch.nn.functional.pad(flat, (0, pad))

    flat_blocked = flat.reshape(m, num_blocks, block_size)
    result = flat_blocked * scales.unsqueeze(-1)
    return result.reshape(m, -1)[:, :d].reshape(orig_shape).to(torch.bfloat16)
