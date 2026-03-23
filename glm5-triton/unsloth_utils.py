# Standalone utility functions for GLM-5 Triton kernels.
# Extracted from unsloth/kernels/utils.py — stripped of bitsandbytes,
# unsloth_zoo, and device_type dependencies so the kernels can be used
# independently.
#
# Original copyright:
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import triton
import triton.language as tl
import torch
import functools
import ctypes
from contextlib import nullcontext

# ---------------------------------------------------------------------------
# Triton version compatibility
# ---------------------------------------------------------------------------
MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2

def _parse_version(v):
    """Minimal semver parse so we don't need packaging/unsloth_zoo."""
    import re
    parts = re.split(r"[.\-+]", v)
    nums = []
    for p in parts[:3]:
        m = re.match(r"(\d+)", p)
        if m:
            nums.append(int(m.group(1)))
    return tuple(nums)

_triton_version = _parse_version(triton.__version__)
_torch_version = _parse_version(torch.__version__)

if _triton_version >= (3, 0, 0):
    try:
        from triton.language.extra import libdevice
        triton_tanh = libdevice.tanh
    except ImportError:
        triton_tanh = tl.math.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)


# ---------------------------------------------------------------------------
# Device detection (no unsloth device_type / bitsandbytes dependency)
# ---------------------------------------------------------------------------
@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))


@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "hip" if is_hip() else "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError(
        "glm5-triton requires a CUDA, ROCm, or XPU GPU."
    )


DEVICE_TYPE: str = get_device_type()
DEVICE_TYPE_TORCH: str = "cuda" if DEVICE_TYPE == "hip" else DEVICE_TYPE

@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    return 1

DEVICE_COUNT: int = get_device_count()


# ---------------------------------------------------------------------------
# GPU device context manager (for multi-GPU)
# ---------------------------------------------------------------------------
if DEVICE_COUNT > 1:
    if DEVICE_TYPE in ("cuda", "hip"):
        torch_gpu_device = torch.cuda.device
    elif DEVICE_TYPE == "xpu":
        torch_gpu_device = torch.xpu.device
else:
    def torch_gpu_device(device):
        return nullcontext()


# ---------------------------------------------------------------------------
# Stream helper
# ---------------------------------------------------------------------------
torch_device_stream = (
    torch.xpu.current_stream if DEVICE_TYPE == "xpu" else torch.cuda.current_stream
)


# ---------------------------------------------------------------------------
# CDNA / RDNA detection (for ROCm warp-count tuning)
# ---------------------------------------------------------------------------
@functools.lru_cache(1)
def is_cdna():
    if not is_hip():
        return False
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
        return arch in ("gfx940", "gfx941", "gfx942", "gfx950")
    except Exception:
        return False


@functools.lru_cache(1)
def is_rdna():
    if not is_hip():
        return False
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
        return arch in ("gfx1100", "gfx1101", "gfx1200", "gfx1201")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Triton block-size calculator
# ---------------------------------------------------------------------------
def calculate_settings(n: int) -> tuple[int, int]:
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# ---------------------------------------------------------------------------
# torch.cuda.amp compatibility
# ---------------------------------------------------------------------------
if _torch_version < (2, 4, 0):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

if DEVICE_TYPE == "xpu":
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="xpu")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="xpu")


# ---------------------------------------------------------------------------
# Weight dequantization stubs.
# Full bitsandbytes fast_dequantize / fast_gemv are NOT included here.
# If you need 4-bit LoRA, install bitsandbytes and use the original unsloth
# kernels.  For bf16/fp16 full-precision or fp8 weights these are not needed.
# ---------------------------------------------------------------------------
def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


def fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    """Passthrough when no quantisation is active."""
    if quant_state is None:
        return W
    raise NotImplementedError(
        "4-bit dequantization requires bitsandbytes. "
        "Use the full unsloth kernel utils for quantized models."
    )


def fast_gemv(X, W, quant_state, out=None):
    if quant_state is None:
        return torch.matmul(X, W, out=out)
    raise NotImplementedError(
        "4-bit GEMV requires bitsandbytes. "
        "Use the full unsloth kernel utils for quantized models."
    )


# ---------------------------------------------------------------------------
# LoRA parameter extraction (works with HF PEFT, no bnb dependency)
# ---------------------------------------------------------------------------
def get_lora_parameters(proj):
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    if hasattr(base_layer, "weight_fake_quantizer"):
        wfq = getattr(base_layer, "weight_fake_quantizer", None)
        if wfq is not None:
            W = wfq(W)

    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(base_layer, "quant_method", None) == "fp8":
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default",))
    adapter = adapter[0]

    lora_A_linear = proj.lora_A[adapter]
    lora_B_linear = proj.lora_B[adapter]
    A = lora_A_linear.weight
    B = lora_B_linear.weight
    if hasattr(lora_A_linear, "weight_fake_quantizer"):
        fq = getattr(lora_A_linear, "weight_fake_quantizer", None)
        if fq is not None:
            A = fq(A)
    if hasattr(lora_B_linear, "weight_fake_quantizer"):
        fq = getattr(lora_B_linear, "weight_fake_quantizer", None)
        if fq is not None:
            B = fq(B)

    return W, W_quant, A, B, proj.scaling[adapter]


def get_lora_parameters_bias(proj):
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None, base_layer.bias

    if getattr(base_layer, "quant_method", None) == "fp8":
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default",))
    adapter = adapter[0]

    return (
        W,
        W_quant,
        proj.lora_A[adapter].weight,
        proj.lora_B[adapter].weight,
        proj.scaling[adapter],
        base_layer.bias,
    )


def _maybe_fake_quantize_activations(X, proj):
    base_layer = getattr(proj, "base_layer", proj)
    activation_fake_quantizer = getattr(base_layer, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        X = activation_fake_quantizer(X)
    return X


# ---------------------------------------------------------------------------
# matmul_lora — handles base W @ X + LoRA delta, bf16/fp16 path only.
# For 4-bit quantized weights, use the original unsloth utils.
# ---------------------------------------------------------------------------
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if W_quant is None:
        out = torch.matmul(X, W.t(), out=out)
    else:
        W = fast_dequantize(W, W_quant)
        out = torch.matmul(X, W.t(), out=out)
        del W

    if A is not None:
        At, Bt = A.t(), B.t()
        XA = torch.matmul(X, At.to(dtype))
        out.addmm_(XA, Bt.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out


def fast_linear_forward(proj, X, temp_lora=None, out=None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch.matmul(X, W.t(), out=out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out=out)
    else:
        W = fast_dequantize(W, W_quant)
        out = torch.matmul(X, W, out=out)

    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype
        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out=temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha=lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(
                X.view(bsz, in_dim), lora_A._fast_lora.t(), out=temp_lora
            )
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha=lora_S)
        out = out.view(bsz, 1, out_dim)

    if bias is not None:
        out += bias

    return out
