# Fast RMS LayerNorm — Triton fused forward + backward.
# Extracted from unsloth/kernels/rms_layernorm.py for standalone use.
#
# Original copyright:
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import triton
import triton.language as tl
import torch
from .unsloth_utils import calculate_settings, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)  # .to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)


def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    # dW, dW_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel for the backward pass
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if GEMMA:
        dX += row_idx * dY_row_stride
    else:
        dX = dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = X.shape
        BLOCK_SIZE: int
        num_warps: int
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_gpu_device(device):
            fx[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                n_cols,
                eps,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim: int = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows: int
        n_cols: int
        n_rows, n_cols = dY.shape
        dX = torch.empty_like(dY) if ctx.GEMMA else dY

        with torch_gpu_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                n_cols,
                ctx.eps,
                GEMMA = ctx.GEMMA,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps = ctx.num_warps,
            )
        dX = dX.view(*shape)
        return dX, None, None, None


@torch.compiler.disable
def fast_rms_layernorm(layernorm, X: torch.Tensor, gemma: bool = False):
    """
    Drop-in replacement for RMSNorm forward.
    Accepts any module with a .weight and either .variance_epsilon or .eps.
    """
    W: torch.Tensor = layernorm.weight
    eps: float = (
        layernorm.variance_epsilon
        if hasattr(layernorm, "variance_epsilon")
        else layernorm.eps
    )
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
