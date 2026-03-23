# FP8 quantization utilities for FlashInfer and DeepGEMM.
#
# FlashInfer uses a SIMPLER FP8 format than FlashMLA:
#   [num_pages, page_size, 576] contiguous FP8 (ckv + kpe concatenated)
#   Scale factors are EXTERNAL (bmm1_scale, bmm2_scale), not inline.
#   576 bytes/token vs FlashMLA's 656 bytes/token (12% more memory-efficient).
#
# DeepGEMM uses the same (tensor, scales) pair format as in the FlashMLA path.

import torch


# ---------------------------------------------------------------------------
# FlashInfer FP8 KV cache format
# ---------------------------------------------------------------------------

def quantize_kv_flashinfer(
    ckv: torch.Tensor,  # [num_pages, page_size, head_dim_ckv] BF16 (512)
    kpe: torch.Tensor,  # [num_pages, page_size, head_dim_kpe] BF16 (64)
) -> tuple[torch.Tensor, float]:
    """Quantize KV cache into FlashInfer's contiguous FP8 format.

    Layout per token: [ckv (512) | kpe (64)] = 576 bytes as FP8 E4M3.
    Scale factor is computed globally and returned separately.

    Args:
        ckv: [num_pages, page_size, 512] compressed KV in BF16
        kpe: [num_pages, page_size, 64] RoPE keys in BF16

    Returns:
        kv_fp8: [num_pages, page_size, 576] in FP8 E4M3
        scale: float — global scale factor (amax / 448)
    """
    # Concatenate ckv and kpe
    kv = torch.cat([ckv, kpe], dim=-1)  # [num_pages, page_size, 576]

    # Global scale for the entire cache
    amax = kv.abs().float().max().clamp(min=1e-4)
    scale = (amax / 448.0).item()

    kv_fp8 = (kv.float() / scale).to(torch.float8_e4m3fn)
    return kv_fp8, scale


def dequantize_kv_flashinfer(
    kv_fp8: torch.Tensor,  # [num_pages, page_size, 576] FP8
    scale: float,
    head_dim_ckv: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FlashInfer FP8 KV cache back to BF16.

    Returns:
        ckv: [num_pages, page_size, head_dim_ckv] BF16
        kpe: [num_pages, page_size, head_dim_kpe] BF16
    """
    kv = kv_fp8.float() * scale
    ckv = kv[..., :head_dim_ckv].to(torch.bfloat16)
    kpe = kv[..., head_dim_ckv:].to(torch.bfloat16)
    return ckv, kpe


# ---------------------------------------------------------------------------
# DeepGEMM FP8 activation format (same as FlashMLA path)
# ---------------------------------------------------------------------------

def quantize_activations_deepgemm(
    x: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 E4M3 with per-block scaling for DeepGEMM."""
    orig_shape = x.shape
    d = orig_shape[-1]
    flat = x.reshape(-1, d).float()
    m = flat.shape[0]

    num_blocks_per_row = (d + block_size - 1) // block_size
    if d % block_size != 0:
        pad = block_size - (d % block_size)
        flat = torch.nn.functional.pad(flat, (0, pad))

    flat_blocked = flat.reshape(m, num_blocks_per_row, block_size)
    amax = flat_blocked.abs().amax(dim=-1).clamp(min=1e-4)
    scales = amax / 448.0

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
