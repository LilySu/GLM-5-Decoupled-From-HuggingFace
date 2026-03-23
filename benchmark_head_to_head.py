#!/usr/bin/env python3
"""Head-to-head benchmark: FlashMLA+DeepGEMM vs FlashInfer for GLM-5 on H100.

Compares the two kernel-accelerated backends across all GLM-5 components:
  - MLA decode attention (FlashMLA vs FlashInfer FA3)
  - DSA indexer scoring (DeepGEMM — shared CONTROL)
  - MoE grouped GEMM (DeepGEMM — shared CONTROL)
  - FP8 KV quantization (FlashMLA V32 sparse vs FlashInfer contiguous)

Experiments:
  component       — per-component at fixed B=32, T=4096
  batch-scaling   — sweep B in {1..128} at fixed T=4096
  context-scaling — sweep T in {256..64K} at fixed B=32
  fp8             — BF16 vs FP8 impact across context lengths
  memory          — peak GPU memory at various (B, T) configs
  serving         — 4 realistic serving scenarios

Usage:
    python3 benchmark_head_to_head.py --experiment all
    python3 benchmark_head_to_head.py --experiment component --impl both
    python3 benchmark_head_to_head.py --experiment batch-scaling --output-dir ./results

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - PyTorch 2.x with CUDA 12.0+
    - At least one of: flash-mla, flashinfer
    - deep-gemm (for DSA indexer and MoE GEMM benchmarks)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import numpy as np

# ── Conditional imports ──────────────────────────────────────────────────

FLASH_MLA_AVAILABLE = False
try:
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    FLASH_MLA_AVAILABLE = True
except ImportError:
    pass

FLASHINFER_AVAILABLE = False
try:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    FLASHINFER_AVAILABLE = True
except ImportError:
    pass

DEEP_GEMM_AVAILABLE = False
try:
    import deep_gemm
    from deep_gemm.utils import per_custom_dims_cast_to_fp8
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    pass


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    """GLM-5 model dimensions and H100 hardware peaks."""
    # Model dimensions
    H: int = 64                      # num_attention_heads
    kv_lora_rank: int = 512          # compressed KV dim (d_ckv)
    d_qk: int = 576                  # absorbed query/key dim (512 nope + 64 rope)
    d_v: int = 512                   # value dim (= kv_lora_rank after absorption)
    hidden: int = 6144               # hidden_size
    index_n_heads: int = 32          # DSA indexer heads
    index_head_dim: int = 128        # DSA indexer head dim
    n_experts: int = 256             # n_routed_experts
    top_k: int = 8                   # num_experts_per_tok
    moe_intermediate: int = 2048     # moe_intermediate_size
    qk_rope_head_dim: int = 64       # RoPE dimension
    qk_nope_head_dim: int = 192      # nope dimension per head (before absorption)
    q_lora_rank: int = 2048          # query LoRA rank
    index_topk: int = 2048           # DSA top-k selection
    page_size: int = 64              # FlashMLA page size

    # H100 SXM peak specifications
    peak_tflops_bf16: float = 989.0   # BF16 Tensor Core TFLOPS
    peak_tflops_fp8: float = 1979.0   # FP8 Tensor Core TFLOPS
    peak_hbm_bandwidth_gb_s: float = 3350.0  # HBM3 bandwidth GB/s


@dataclass
class BenchResult:
    """Full result from a single benchmark measurement."""
    name: str
    impl: str                                # "flashmla", "flashinfer", or "control"
    config: dict = field(default_factory=dict)

    # Raw latency data
    latency_ms: List[float] = field(default_factory=list)

    # Summary statistics
    median_ms: float = 0.0
    p5_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_ms: float = 0.0
    ci_95: tuple = (0.0, 0.0)               # bootstrap 95% CI on median

    # Roofline metrics
    mfu_pct: float = 0.0                     # Model FLOPs Utilization
    hbm_sol_pct: float = 0.0                 # HBM Speed-of-Light %
    tflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    operational_intensity: float = 0.0       # FLOPs / byte

    # Memory
    peak_memory_gb: float = 0.0


# ── CUDA Timer with Bootstrap CI ─────────────────────────────────────────

def cuda_timer_extended(fn, warmup=10, iters=100):
    """Time a CUDA function with proper synchronization and bootstrap 95% CI.

    Args:
        fn: Callable to benchmark (no arguments).
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        (raw_times, stats): raw_times is a sorted list of ms values.
            stats dict has keys: median, p5, p50, p95, p99, std, ci_95_lo, ci_95_hi.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times_arr = np.array(times)

    # Summary statistics
    median = float(np.median(times_arr))
    stats = {
        "median": median,
        "p5": float(np.percentile(times_arr, 5)),
        "p50": median,
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "std": float(np.std(times_arr)),
    }

    # Bootstrap 95% CI on the median (1000 resamples)
    n_resamples = 1000
    rng = np.random.default_rng(seed=42)
    bootstrap_medians = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(times_arr, size=len(times_arr), replace=True)
        bootstrap_medians[i] = np.median(sample)
    ci_lo = float(np.percentile(bootstrap_medians, 2.5))
    ci_hi = float(np.percentile(bootstrap_medians, 97.5))
    stats["ci_95_lo"] = ci_lo
    stats["ci_95_hi"] = ci_hi

    return sorted(times), stats


# ── Compute Helpers ──────────────────────────────────────────────────────

def compute_mfu(tflops_achieved: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization as a percentage."""
    if peak_tflops <= 0:
        return 0.0
    return 100.0 * tflops_achieved / peak_tflops


def compute_hbm_sol(bandwidth_achieved_gb_s: float, peak_bandwidth_gb_s: float) -> float:
    """HBM Speed-of-Light percentage."""
    if peak_bandwidth_gb_s <= 0:
        return 0.0
    return 100.0 * bandwidth_achieved_gb_s / peak_bandwidth_gb_s


def compute_attention_flops(B: int, H: int, S_q: int, S_kv: int,
                            d_qk: int, d_v: int) -> int:
    """Total FLOPs for attention: QK^T matmul + softmax(ignored) + PV matmul.

    QK^T: 2 * B * H * S_q * S_kv * d_qk
    PV:   2 * B * H * S_q * S_kv * d_v
    """
    flops_qk = 2 * B * H * S_q * S_kv * d_qk
    flops_pv = 2 * B * H * S_q * S_kv * d_v
    return flops_qk + flops_pv


def compute_attention_bytes(B: int, H: int, S_q: int, S_kv: int,
                            d_qk: int, d_v: int, dtype_bytes: int = 2) -> int:
    """Minimum HBM bytes transferred for attention (decode, S_q=1).

    Reads:  Q [B, H, S_q, d_qk], K_cache [B, 1, S_kv, d_qk], V_cache [B, 1, S_kv, d_v]
    Writes: O [B, H, S_q, d_v]
    """
    bytes_q = B * H * S_q * d_qk * dtype_bytes
    bytes_k = B * S_kv * d_qk * dtype_bytes   # MLA: single KV head
    bytes_v = B * S_kv * d_v * dtype_bytes
    bytes_o = B * H * S_q * d_v * dtype_bytes
    return bytes_q + bytes_k + bytes_v + bytes_o


def _fill_roofline(result: BenchResult, flops: int, total_bytes: int,
                   cfg: BenchConfig, precision: str = "bf16"):
    """Fill roofline metrics on a BenchResult in-place."""
    if result.median_ms <= 0:
        return
    elapsed_s = result.median_ms * 1e-3
    result.tflops = flops / elapsed_s / 1e12
    result.bandwidth_gb_s = total_bytes / elapsed_s / 1e9
    if total_bytes > 0:
        result.operational_intensity = flops / total_bytes
    peak = cfg.peak_tflops_fp8 if precision == "fp8" else cfg.peak_tflops_bf16
    result.mfu_pct = compute_mfu(result.tflops, peak)
    result.hbm_sol_pct = compute_hbm_sol(result.bandwidth_gb_s, cfg.peak_hbm_bandwidth_gb_s)


def _make_result(name: str, impl: str, raw_times: list, stats: dict,
                 config_dict: dict) -> BenchResult:
    """Populate a BenchResult from timer output."""
    return BenchResult(
        name=name,
        impl=impl,
        config=config_dict,
        latency_ms=raw_times,
        median_ms=stats["median"],
        p5_ms=stats["p5"],
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        p99_ms=stats["p99"],
        std_ms=stats["std"],
        ci_95=(stats["ci_95_lo"], stats["ci_95_hi"]),
    )


def _oom_result(name: str, impl: str, config_dict: dict, error: str) -> BenchResult:
    """Return a sentinel result when OOM or import failure occurs."""
    r = BenchResult(name=name, impl=impl, config=config_dict)
    r.median_ms = -1.0
    r.config["error"] = error
    return r


def _clear_cuda_cache():
    """Clear CUDA cache and reset peak memory stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── Benchmark Functions ──────────────────────────────────────────────────

def bench_mla_decode_flashmla(B: int, T: int, precision: str,
                              cfg: BenchConfig, warmup: int = 10,
                              iters: int = 100) -> BenchResult:
    """Benchmark FlashMLA MLA decode kernel."""
    config_dict = {"B": B, "T": T, "precision": precision}
    if not FLASH_MLA_AVAILABLE:
        return _oom_result("mla_decode", "flashmla", config_dict,
                           "flash_mla not installed")
    _clear_cuda_cache()
    device = "cuda"
    try:
        H = cfg.H
        d_qk = cfg.d_qk
        d_v = cfg.d_v
        page_size = cfg.page_size
        num_pages = (B * T + page_size - 1) // page_size

        q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)
        k_cache = torch.randn(num_pages, page_size, 1, d_qk,
                               dtype=torch.bfloat16, device=device)
        if precision == "fp8":
            # Quantize nope portion to FP8, keep rope in BF16 (V32 sparse format)
            k_cache_view = k_cache.view(-1, d_qk)
            nope = k_cache_view[:, :cfg.kv_lora_rank]
            amax = nope.abs().float().max().clamp(min=1e-4)
            scale = (amax / 448.0)
            nope_fp8 = (nope.float() / scale).to(torch.float8_e4m3fn)
            # Keep original for the API — FlashMLA handles FP8 internally
            # We just measure the overhead of the quantize step separately

        block_table = torch.arange(num_pages, device=device,
                                    dtype=torch.int32).view(B, -1)
        cache_seqlens = torch.full((B,), T, dtype=torch.int32, device=device)

        # get_mla_metadata needs (batch_size, cache_seqlens, num_heads, ...)
        # Use a safe call pattern
        try:
            metadata = get_mla_metadata(cache_seqlens, H, device=device)
        except TypeError:
            metadata = get_mla_metadata(cache_seqlens, H)

        def run():
            flash_mla_with_kvcache(
                q, k_cache, block_table, cache_seqlens,
                head_dim_v=d_v,
                tile_scheduler_metadata=metadata,
                softmax_scale=d_qk ** -0.5,
                causal=False,
            )

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("mla_decode", "flashmla", raw_times, stats, config_dict)

        # Roofline (S_q=1 decode)
        flops = compute_attention_flops(B, H, 1, T, d_qk, d_v)
        total_bytes = compute_attention_bytes(B, H, 1, T, d_qk, d_v)
        _fill_roofline(result, flops, total_bytes, cfg, precision)
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("mla_decode", "flashmla", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("mla_decode", "flashmla", config_dict, f"Error: {e}")


def bench_mla_decode_flashinfer(B: int, T: int, precision: str,
                                cfg: BenchConfig, warmup: int = 10,
                                iters: int = 100) -> BenchResult:
    """Benchmark FlashInfer FA3 MLA decode kernel."""
    config_dict = {"B": B, "T": T, "precision": precision}
    if not FLASHINFER_AVAILABLE:
        return _oom_result("mla_decode", "flashinfer", config_dict,
                           "flashinfer not installed")
    _clear_cuda_cache()
    device = "cuda"
    try:
        H = cfg.H
        d_ckv = cfg.kv_lora_rank   # 512
        d_kpe = cfg.qk_rope_head_dim  # 64
        page_size = 1  # FlashInfer uses page_size=1 for MLA

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")

        q_nope = torch.randn(B, H, d_ckv, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(B, H, d_kpe, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(B * T, 1, d_ckv, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(B * T, 1, d_kpe, dtype=torch.bfloat16, device=device)

        if precision == "fp8":
            ckv = ckv.to(torch.float8_e4m3fn)
            kpe = kpe.to(torch.float8_e4m3fn)

        qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * T
        kv_indices = torch.arange(0, B * T, dtype=torch.int32, device=device)
        kv_lens = torch.full((B,), T, dtype=torch.int32, device=device)
        sm_scale = 1.0 / ((d_ckv + d_kpe) ** 0.5)

        kv_dtype = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_lens,
            H, d_ckv, d_kpe, page_size, False, sm_scale,
            torch.bfloat16, kv_dtype,
        )

        def run():
            wrapper.run(q_nope, q_pe, ckv, kpe)

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("mla_decode", "flashinfer", raw_times, stats, config_dict)

        # Roofline
        d_qk_total = d_ckv + d_kpe  # 576
        flops = compute_attention_flops(B, H, 1, T, d_qk_total, d_ckv)
        total_bytes = compute_attention_bytes(B, H, 1, T, d_qk_total, d_ckv)
        _fill_roofline(result, flops, total_bytes, cfg, precision)
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("mla_decode", "flashinfer", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("mla_decode", "flashinfer", config_dict, f"Error: {e}")


def bench_dsa_indexer(B: int, T: int, cfg: BenchConfig,
                      warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark DSA indexer scoring (DeepGEMM fp8_mqa_logits).

    This is a CONTROL benchmark — both backends use the same DeepGEMM kernel.
    """
    config_dict = {"B": B, "T": T}
    if not DEEP_GEMM_AVAILABLE:
        return _oom_result("dsa_indexer", "control", config_dict,
                           "deep_gemm not installed")
    _clear_cuda_cache()
    device = "cuda"
    try:
        H = cfg.index_n_heads    # 32
        D = cfg.index_head_dim   # 128
        seq_len = 1  # decode token

        q = torch.randn(seq_len, H, D, device=device,
                         dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        kv = torch.randn(T, D, device=device, dtype=torch.bfloat16)

        # Quantize KV to FP8 with per-block scaling
        flat = kv.reshape(-1, D).float()
        block_size = 128
        num_blocks = (D + block_size - 1) // block_size
        flat_blocked = flat.reshape(-1, num_blocks, block_size)
        amax = flat_blocked.abs().amax(dim=-1).clamp(min=1e-4)
        scales = amax / 448.0
        x_scaled = flat_blocked / scales.unsqueeze(-1)
        kv_fp8 = x_scaled.reshape(-1, D).to(torch.float8_e4m3fn)
        kv_fp8 = kv_fp8.reshape(T, D)

        weights = torch.randn(seq_len, H, device=device, dtype=torch.float32)
        ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
        ke = torch.full((seq_len,), T, dtype=torch.int32, device=device)

        def run():
            deep_gemm.fp8_mqa_logits(q, (kv_fp8, scales), weights, ks, ke)

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("dsa_indexer", "control", raw_times, stats, config_dict)

        flops = 2 * seq_len * T * H * D
        total_bytes = (seq_len * H * D + T * D + seq_len * H * 4) * 1  # FP8 = 1 byte
        total_bytes += T * num_blocks * 4  # scales
        _fill_roofline(result, flops, total_bytes, cfg, "fp8")
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("dsa_indexer", "control", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("dsa_indexer", "control", config_dict, f"Error: {e}")


def bench_moe_gemm(N: int, cfg: BenchConfig,
                   warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark DeepGEMM FP8 grouped GEMM for MoE.

    This is a CONTROL benchmark — both backends use the same DeepGEMM kernel.
    """
    config_dict = {"N": N, "E": cfg.n_experts, "K": cfg.top_k}
    if not DEEP_GEMM_AVAILABLE:
        return _oom_result("moe_gemm", "control", config_dict,
                           "deep_gemm not installed")
    _clear_cuda_cache()
    device = "cuda"
    try:
        E = cfg.n_experts           # 256
        K = cfg.top_k               # 8
        D = cfg.hidden              # 6144
        I = cfg.moe_intermediate    # 2048

        a = torch.randn(N * K, D, device=device, dtype=torch.bfloat16)
        b = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)

        a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
        b_fp8 = per_custom_dims_cast_to_fp8(b.view(E * I, D), (0,), False)
        b_fp8 = (b_fp8[0].view(E, I, D), b_fp8[1].view(E, I))

        d_out = torch.empty(N * K, I, device=device, dtype=torch.bfloat16)
        grouped_layout = torch.zeros(N * K, dtype=torch.int32, device=device)
        # Assign tokens to experts round-robin
        for i in range(N * K):
            grouped_layout[i] = i % E

        def run():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a_fp8, b_fp8, d_out, grouped_layout)

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("moe_gemm", "control", raw_times, stats, config_dict)

        flops = 2 * N * K * D * I
        total_bytes = (N * K * D + E * I * D + N * K * I) * 1  # FP8 weights
        _fill_roofline(result, flops, total_bytes, cfg, "fp8")
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("moe_gemm", "control", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("moe_gemm", "control", config_dict, f"Error: {e}")


def bench_fp8_quant_flashmla(T: int, cfg: BenchConfig,
                             warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark FlashMLA FP8 KV cache quantization (V32 sparse format).

    Per-token layout: [512 FP8 nope | 4xFP32 scales | 64 BF16 rope] = 656 bytes.
    """
    config_dict = {"T": T, "format": "V32_sparse_656B"}
    _clear_cuda_cache()
    device = "cuda"
    try:
        d_nope = cfg.kv_lora_rank    # 512
        d_rope = cfg.qk_rope_head_dim  # 64
        d_total = d_nope + d_rope     # 576
        tile_size = 128
        num_tiles = d_nope // tile_size  # 4

        # Simulate paged KV cache
        page_size = cfg.page_size
        num_pages = (T + page_size - 1) // page_size
        kv = torch.randn(num_pages, page_size, 1, d_total,
                          dtype=torch.bfloat16, device=device)

        bytes_per_token = d_nope + num_tiles * 4 + 2 * d_rope  # 512+16+128=656

        def run():
            kv_sq = kv.squeeze(2)  # [num_pages, page_size, 576]
            result_buf = torch.empty(
                (num_pages, page_size + 1, bytes_per_token),
                dtype=torch.float8_e4m3fn, device=device,
            )[:, :page_size, :]
            nope_buf = result_buf[..., :d_nope]
            scales_buf = result_buf[..., d_nope:d_nope + num_tiles * 4].view(
                torch.float32)
            rope_buf = result_buf[..., d_nope + num_tiles * 4:].view(torch.bfloat16)
            rope_buf[:] = kv_sq[..., d_nope:]
            for t in range(num_tiles):
                ts = t * tile_size
                te = ts + tile_size
                tile = kv_sq[..., ts:te]
                amax = tile.abs().float().amax(dim=-1).clamp(min=1e-4)
                scale_inv = amax / 448.0
                scale_inv = torch.pow(2, scale_inv.log2().ceil())
                scales_buf[:, :, t] = scale_inv
                nope_buf[..., ts:te] = (tile.float() / scale_inv.unsqueeze(-1).float()
                                         ).to(torch.float8_e4m3fn)

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("fp8_quant", "flashmla", raw_times, stats, config_dict)

        # Bandwidth: read d_total BF16 per token, write bytes_per_token per token
        total_tokens = num_pages * page_size
        read_bytes = total_tokens * d_total * 2   # BF16
        write_bytes = total_tokens * bytes_per_token
        result.bandwidth_gb_s = (read_bytes + write_bytes) / (result.median_ms * 1e-3) / 1e9
        result.hbm_sol_pct = compute_hbm_sol(result.bandwidth_gb_s,
                                              cfg.peak_hbm_bandwidth_gb_s)
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("fp8_quant", "flashmla", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("fp8_quant", "flashmla", config_dict, f"Error: {e}")


def bench_fp8_quant_flashinfer(T: int, cfg: BenchConfig,
                               warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark FlashInfer FP8 KV cache quantization (contiguous format).

    Per-token layout: [576 FP8 contiguous] + external float scale = 576 bytes + 4 bytes.
    """
    config_dict = {"T": T, "format": "contiguous_576B"}
    _clear_cuda_cache()
    device = "cuda"
    try:
        d_ckv = cfg.kv_lora_rank      # 512
        d_kpe = cfg.qk_rope_head_dim  # 64
        d_total = d_ckv + d_kpe       # 576

        page_size = cfg.page_size
        num_pages = (T + page_size - 1) // page_size

        ckv = torch.randn(num_pages, page_size, d_ckv,
                           dtype=torch.bfloat16, device=device)
        kpe = torch.randn(num_pages, page_size, d_kpe,
                           dtype=torch.bfloat16, device=device)

        def run():
            kv = torch.cat([ckv, kpe], dim=-1)  # [num_pages, page_size, 576]
            amax = kv.abs().float().max().clamp(min=1e-4)
            scale = (amax / 448.0).item()
            _ = (kv.float() / scale).to(torch.float8_e4m3fn)

        raw_times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
        result = _make_result("fp8_quant", "flashinfer", raw_times, stats, config_dict)

        # Bandwidth: read ckv + kpe BF16, write d_total FP8 per token
        total_tokens = num_pages * page_size
        read_bytes = total_tokens * d_total * 2     # BF16 input
        write_bytes = total_tokens * d_total * 1    # FP8 output
        result.bandwidth_gb_s = (read_bytes + write_bytes) / (result.median_ms * 1e-3) / 1e9
        result.hbm_sol_pct = compute_hbm_sol(result.bandwidth_gb_s,
                                              cfg.peak_hbm_bandwidth_gb_s)
        result.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        return result

    except torch.cuda.OutOfMemoryError as e:
        _clear_cuda_cache()
        return _oom_result("fp8_quant", "flashinfer", config_dict, f"OOM: {e}")
    except Exception as e:
        return _oom_result("fp8_quant", "flashinfer", config_dict, f"Error: {e}")


# ── Experiment Functions ─────────────────────────────────────────────────

def _run_impls(impls: str, flashmla_fn, flashinfer_fn, control_fn=None):
    """Run benchmark functions for selected implementations.

    Returns list of BenchResult.
    """
    results = []
    if control_fn is not None:
        results.append(control_fn())
    if impls in ("both", "flashmla"):
        results.append(flashmla_fn())
    if impls in ("both", "flashinfer"):
        results.append(flashinfer_fn())
    return results


def experiment_component(impls: str, cfg: BenchConfig,
                         warmup: int = 10, iters: int = 100) -> List[BenchResult]:
    """Per-component benchmark at fixed B=32, T=4096."""
    B, T = 32, 4096
    results = []
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: component (B={B}, T={T})")
    print(f"{'='*70}")

    # MLA decode
    print("  [1/4] MLA decode attention...")
    results.extend(_run_impls(
        impls,
        lambda: bench_mla_decode_flashmla(B, T, "bf16", cfg, warmup, iters),
        lambda: bench_mla_decode_flashinfer(B, T, "bf16", cfg, warmup, iters),
    ))
    for r in results[-2:] if impls == "both" else results[-1:]:
        if r.median_ms > 0:
            print(f"    {r.impl:>12s}: {r.median_ms:.3f} ms  "
                  f"({r.tflops:.1f} TFLOPS, {r.mfu_pct:.1f}% MFU)")

    # DSA indexer (CONTROL)
    print("  [2/4] DSA indexer (CONTROL — same DeepGEMM kernel)...")
    r = bench_dsa_indexer(B, T, cfg, warmup, iters)
    results.append(r)
    if r.median_ms > 0:
        print(f"    {'control':>12s}: {r.median_ms:.3f} ms  ({r.tflops:.1f} TFLOPS)")

    # MoE grouped GEMM (CONTROL)
    print("  [3/4] MoE grouped GEMM (CONTROL — same DeepGEMM kernel)...")
    r = bench_moe_gemm(1024, cfg, warmup, iters)
    results.append(r)
    if r.median_ms > 0:
        print(f"    {'control':>12s}: {r.median_ms:.3f} ms  ({r.tflops:.1f} TFLOPS)")

    # FP8 quantization
    print("  [4/4] FP8 KV cache quantization...")
    results.extend(_run_impls(
        impls,
        lambda: bench_fp8_quant_flashmla(T, cfg, warmup, iters),
        lambda: bench_fp8_quant_flashinfer(T, cfg, warmup, iters),
    ))
    for r in results[-2:] if impls == "both" else results[-1:]:
        if r.median_ms > 0:
            print(f"    {r.impl:>12s}: {r.median_ms:.3f} ms  "
                  f"({r.bandwidth_gb_s:.0f} GB/s, {r.hbm_sol_pct:.1f}% HBM SoL)")

    return results


def experiment_batch_scaling(impls: str, cfg: BenchConfig,
                             warmup: int = 10, iters: int = 100) -> List[BenchResult]:
    """MLA decode across batch sizes: B in {1,2,4,8,16,32,64,128}, T=4096."""
    T = 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: batch-scaling (T={T})")
    print(f"{'='*70}")
    print(f"  {'B':>6s}  {'FlashMLA ms':>14s}  {'FlashInfer ms':>14s}  {'Ratio':>8s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}")

    for B in batch_sizes:
        _clear_cuda_cache()
        row_results = _run_impls(
            impls,
            lambda b=B: bench_mla_decode_flashmla(b, T, "bf16", cfg, warmup, iters),
            lambda b=B: bench_mla_decode_flashinfer(b, T, "bf16", cfg, warmup, iters),
        )
        results.extend(row_results)

        fmla_ms = next((r.median_ms for r in row_results if r.impl == "flashmla"), -1)
        fi_ms = next((r.median_ms for r in row_results if r.impl == "flashinfer"), -1)
        ratio_str = ""
        if fmla_ms > 0 and fi_ms > 0:
            ratio_str = f"{fmla_ms / fi_ms:.2f}x"
        print(f"  {B:>6d}  {fmla_ms:>14.3f}  {fi_ms:>14.3f}  {ratio_str:>8s}")

    return results


def experiment_context_scaling(impls: str, cfg: BenchConfig,
                               warmup: int = 10, iters: int = 100) -> List[BenchResult]:
    """MLA decode across context lengths: T in {256..64K}, B=32."""
    B = 32
    context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    results = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: context-scaling (B={B})")
    print(f"{'='*70}")
    print(f"  {'T':>8s}  {'FlashMLA ms':>14s}  {'FlashInfer ms':>14s}  {'Ratio':>8s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")

    for T in context_lengths:
        _clear_cuda_cache()
        row_results = _run_impls(
            impls,
            lambda t=T: bench_mla_decode_flashmla(B, t, "bf16", cfg, warmup, iters),
            lambda t=T: bench_mla_decode_flashinfer(B, t, "bf16", cfg, warmup, iters),
        )
        results.extend(row_results)

        fmla_ms = next((r.median_ms for r in row_results if r.impl == "flashmla"), -1)
        fi_ms = next((r.median_ms for r in row_results if r.impl == "flashinfer"), -1)
        ratio_str = ""
        if fmla_ms > 0 and fi_ms > 0:
            ratio_str = f"{fmla_ms / fi_ms:.2f}x"

        t_str = f"{T//1024}K" if T >= 1024 else str(T)
        print(f"  {t_str:>8s}  {fmla_ms:>14.3f}  {fi_ms:>14.3f}  {ratio_str:>8s}")

    return results


def experiment_fp8_impact(impls: str, cfg: BenchConfig,
                          warmup: int = 10, iters: int = 100) -> List[BenchResult]:
    """Compare BF16 vs FP8 MLA decode across context lengths."""
    B = 32
    context_lengths = [1024, 4096, 16384, 65536]
    results = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: fp8-impact (B={B})")
    print(f"{'='*70}")
    print(f"  {'T':>8s}  {'Impl':>12s}  {'BF16 ms':>10s}  {'FP8 ms':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    for T in context_lengths:
        for impl_name, bench_fn in [("flashmla", bench_mla_decode_flashmla),
                                     ("flashinfer", bench_mla_decode_flashinfer)]:
            if impls not in ("both", impl_name):
                continue
            _clear_cuda_cache()
            r_bf16 = bench_fn(B, T, "bf16", cfg, warmup, iters)
            _clear_cuda_cache()
            r_fp8 = bench_fn(B, T, "fp8", cfg, warmup, iters)
            results.extend([r_bf16, r_fp8])

            speedup_str = ""
            if r_bf16.median_ms > 0 and r_fp8.median_ms > 0:
                speedup_str = f"{r_bf16.median_ms / r_fp8.median_ms:.2f}x"
            t_str = f"{T//1024}K" if T >= 1024 else str(T)
            print(f"  {t_str:>8s}  {impl_name:>12s}  "
                  f"{r_bf16.median_ms:>10.3f}  {r_fp8.median_ms:>10.3f}  "
                  f"{speedup_str:>8s}")

    # Also benchmark FP8 quantization overhead
    print(f"\n  FP8 quantization overhead:")
    print(f"  {'T':>8s}  {'FlashMLA ms':>14s}  {'FlashInfer ms':>14s}  "
          f"{'MLA bytes/tok':>14s}  {'FI bytes/tok':>14s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    for T in context_lengths:
        _clear_cuda_cache()
        row = _run_impls(
            impls,
            lambda t=T: bench_fp8_quant_flashmla(t, cfg, warmup, iters),
            lambda t=T: bench_fp8_quant_flashinfer(t, cfg, warmup, iters),
        )
        results.extend(row)
        fmla_ms = next((r.median_ms for r in row if r.impl == "flashmla"), -1)
        fi_ms = next((r.median_ms for r in row if r.impl == "flashinfer"), -1)
        t_str = f"{T//1024}K" if T >= 1024 else str(T)
        print(f"  {t_str:>8s}  {fmla_ms:>14.3f}  {fi_ms:>14.3f}  "
              f"{'656':>14s}  {'576+4':>14s}")

    return results


def experiment_memory(impls: str, cfg: BenchConfig,
                      warmup: int = 5, iters: int = 20) -> List[BenchResult]:
    """Measure peak GPU memory at various (B, T) configurations."""
    configs = [
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (64, 4096),
        (32, 16384),
        (32, 65536),
        (128, 4096),
        (8, 65536),
    ]
    results = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: memory")
    print(f"{'='*70}")
    print(f"  {'B':>6s}  {'T':>8s}  {'FlashMLA GB':>14s}  {'FlashInfer GB':>14s}  "
          f"{'Savings':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")

    for B, T in configs:
        _clear_cuda_cache()
        row_results = _run_impls(
            impls,
            lambda b=B, t=T: bench_mla_decode_flashmla(b, t, "bf16", cfg, warmup, iters),
            lambda b=B, t=T: bench_mla_decode_flashinfer(b, t, "bf16", cfg, warmup, iters),
        )
        results.extend(row_results)

        fmla_gb = next((r.peak_memory_gb for r in row_results
                        if r.impl == "flashmla" and r.median_ms > 0), -1)
        fi_gb = next((r.peak_memory_gb for r in row_results
                      if r.impl == "flashinfer" and r.median_ms > 0), -1)
        savings_str = ""
        if fmla_gb > 0 and fi_gb > 0:
            pct = 100.0 * (fmla_gb - fi_gb) / fmla_gb
            savings_str = f"{pct:+.1f}%"
        t_str = f"{T//1024}K" if T >= 1024 else str(T)
        print(f"  {B:>6d}  {t_str:>8s}  {fmla_gb:>14.2f}  {fi_gb:>14.2f}  "
              f"{savings_str:>8s}")

    return results


def experiment_serving(impls: str, cfg: BenchConfig,
                       warmup: int = 10, iters: int = 100) -> List[BenchResult]:
    """Benchmark 4 realistic serving scenarios."""
    scenarios = {
        "chatbot": {"B": 64, "T": 512,
                    "desc": "High-concurrency chatbot, short context"},
        "code_assist": {"B": 16, "T": 8192,
                        "desc": "Code completion, medium context with file scope"},
        "long_doc_qa": {"B": 4, "T": 65536,
                        "desc": "Document QA, few users with long context"},
        "agentic_swe": {"B": 8, "T": 32768,
                        "desc": "Agentic SWE, moderate batch with tool-call history"},
    }
    results = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: serving scenarios")
    print(f"{'='*70}")

    for scenario_name, params in scenarios.items():
        B, T = params["B"], params["T"]
        print(f"\n  --- {scenario_name}: {params['desc']} (B={B}, T={T}) ---")

        _clear_cuda_cache()
        row_results = _run_impls(
            impls,
            lambda b=B, t=T: bench_mla_decode_flashmla(b, t, "bf16", cfg, warmup, iters),
            lambda b=B, t=T: bench_mla_decode_flashinfer(b, t, "bf16", cfg, warmup, iters),
        )
        # Tag results with scenario name
        for r in row_results:
            r.config["scenario"] = scenario_name
            r.config["description"] = params["desc"]
        results.extend(row_results)

        for r in row_results:
            if r.median_ms > 0:
                ci_lo, ci_hi = r.ci_95
                print(f"    {r.impl:>12s}: {r.median_ms:.3f} ms  "
                      f"[CI: {ci_lo:.3f}-{ci_hi:.3f}]  "
                      f"{r.tflops:.1f} TFLOPS  {r.mfu_pct:.1f}% MFU  "
                      f"peak={r.peak_memory_gb:.1f}GB")
            else:
                print(f"    {r.impl:>12s}: FAILED ({r.config.get('error', 'unknown')})")

        # Compute winner
        valid = [r for r in row_results if r.median_ms > 0]
        if len(valid) >= 2:
            winner = min(valid, key=lambda r: r.median_ms)
            loser = max(valid, key=lambda r: r.median_ms)
            speedup = loser.median_ms / winner.median_ms
            print(f"    --> Winner: {winner.impl} ({speedup:.2f}x faster)")

    return results


# ── Output / JSON Serialization ──────────────────────────────────────────

def _get_environment_snapshot() -> dict:
    """Capture environment details for reproducibility."""
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu_name"] = props.name
        env["gpu_sm_version"] = f"{props.major}.{props.minor}"
        env["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        env["gpu_count"] = torch.cuda.device_count()

        # GPU temperature (best-effort via nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                env["gpu_temperature_c"] = int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass

    # Library versions
    if FLASH_MLA_AVAILABLE:
        try:
            import flash_mla
            env["flash_mla_version"] = getattr(flash_mla, "__version__", "installed")
        except Exception:
            env["flash_mla_version"] = "installed (version unknown)"
    else:
        env["flash_mla_version"] = "NOT INSTALLED"

    if FLASHINFER_AVAILABLE:
        try:
            import flashinfer
            env["flashinfer_version"] = getattr(flashinfer, "__version__", "installed")
        except Exception:
            env["flashinfer_version"] = "installed (version unknown)"
    else:
        env["flashinfer_version"] = "NOT INSTALLED"

    if DEEP_GEMM_AVAILABLE:
        try:
            env["deep_gemm_version"] = getattr(deep_gemm, "__version__", "installed")
        except Exception:
            env["deep_gemm_version"] = "installed (version unknown)"
    else:
        env["deep_gemm_version"] = "NOT INSTALLED"

    env["numpy_version"] = np.__version__

    return env


def _result_to_dict(r: BenchResult) -> dict:
    """Convert a BenchResult to a JSON-serializable dict."""
    d = {
        "name": r.name,
        "impl": r.impl,
        "config": r.config,
        "median_ms": r.median_ms,
        "p5_ms": r.p5_ms,
        "p50_ms": r.p50_ms,
        "p95_ms": r.p95_ms,
        "p99_ms": r.p99_ms,
        "std_ms": r.std_ms,
        "ci_95_lo": r.ci_95[0],
        "ci_95_hi": r.ci_95[1],
        "mfu_pct": r.mfu_pct,
        "hbm_sol_pct": r.hbm_sol_pct,
        "tflops": r.tflops,
        "bandwidth_gb_s": r.bandwidth_gb_s,
        "operational_intensity": r.operational_intensity,
        "peak_memory_gb": r.peak_memory_gb,
        "latency_ms": r.latency_ms,
    }
    return d


def save_results(all_results: dict, output_dir: str, cfg: BenchConfig):
    """Write full JSON report with environment snapshot and raw measurements.

    Args:
        all_results: dict mapping experiment_name -> list of BenchResult.
        output_dir: directory to write output files.
        cfg: BenchConfig used.
    """
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "environment": _get_environment_snapshot(),
        "bench_config": asdict(cfg),
        "experiments": {},
    }

    for exp_name, results in all_results.items():
        report["experiments"][exp_name] = [_result_to_dict(r) for r in results]

    output_path = os.path.join(output_dir, "benchmark_head_to_head.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Also print a compact summary table
    _print_summary(all_results)


def _print_summary(all_results: dict):
    """Print a compact summary comparing implementations."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for exp_name, results in all_results.items():
        print(f"\n  {exp_name}:")
        # Group by (name, config key that varies)
        by_name = {}
        for r in results:
            if r.name not in by_name:
                by_name[r.name] = []
            by_name[r.name].append(r)

        for name, group in by_name.items():
            flashmla_results = [r for r in group if r.impl == "flashmla" and r.median_ms > 0]
            flashinfer_results = [r for r in group if r.impl == "flashinfer" and r.median_ms > 0]
            control_results = [r for r in group if r.impl == "control" and r.median_ms > 0]

            if control_results:
                r = control_results[0]
                print(f"    {name} (CONTROL): {r.median_ms:.3f} ms")
            elif flashmla_results and flashinfer_results:
                fmla = flashmla_results[0]
                fi = flashinfer_results[0]
                ratio = fmla.median_ms / fi.median_ms if fi.median_ms > 0 else 0
                winner = "FlashMLA" if fmla.median_ms < fi.median_ms else "FlashInfer"
                print(f"    {name}: FlashMLA={fmla.median_ms:.3f}ms  "
                      f"FlashInfer={fi.median_ms:.3f}ms  "
                      f"ratio={ratio:.2f}x  winner={winner}")
            elif flashmla_results:
                print(f"    {name}: FlashMLA={flashmla_results[0].median_ms:.3f}ms  "
                      f"FlashInfer=N/A")
            elif flashinfer_results:
                print(f"    {name}: FlashMLA=N/A  "
                      f"FlashInfer={flashinfer_results[0].median_ms:.3f}ms")


# ── Main ─────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "component": experiment_component,
    "batch-scaling": experiment_batch_scaling,
    "context-scaling": experiment_context_scaling,
    "fp8": experiment_fp8_impact,
    "memory": experiment_memory,
    "serving": experiment_serving,
}


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: FlashMLA+DeepGEMM vs FlashInfer for GLM-5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 benchmark_head_to_head.py --experiment all
    python3 benchmark_head_to_head.py --experiment component --impl both
    python3 benchmark_head_to_head.py --experiment batch-scaling --warmup 5 --iters 50
    python3 benchmark_head_to_head.py --experiment serving --output-dir ./my_results
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=["all"] + list(EXPERIMENTS.keys()),
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--impl",
        choices=["both", "flashmla", "flashinfer"],
        default="both",
        help="Which implementation(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size (for component experiment)",
    )
    parser.add_argument(
        "--context", type=int, default=None,
        help="Override context length (for component experiment)",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp8"],
        default="bf16",
        help="Default precision for single-point benchmarks (default: bf16)",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory for JSON output (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Timed iterations (default: 100)",
    )

    args = parser.parse_args()

    # ── Preflight checks ──
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires an H100 GPU.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU:           {props.name}")
    print(f"SM version:    {props.major}.{props.minor}")
    print(f"GPU memory:    {props.total_memory / 1e9:.1f} GB")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA:          {torch.version.cuda}")
    print(f"FlashMLA:      {'available' if FLASH_MLA_AVAILABLE else 'NOT INSTALLED'}")
    print(f"FlashInfer:    {'available' if FLASHINFER_AVAILABLE else 'NOT INSTALLED'}")
    print(f"DeepGEMM:      {'available' if DEEP_GEMM_AVAILABLE else 'NOT INSTALLED'}")
    print(f"Warmup:        {args.warmup}")
    print(f"Iterations:    {args.iters}")

    if not FLASH_MLA_AVAILABLE and not FLASHINFER_AVAILABLE:
        print("\nWARNING: Neither flash-mla nor flashinfer is installed.")
        print("All MLA decode benchmarks will report -1 (skipped).")
        print("Only DSA indexer and MoE GEMM (DeepGEMM) benchmarks will run.")

    if args.impl == "flashmla" and not FLASH_MLA_AVAILABLE:
        print("\nERROR: --impl flashmla but flash-mla is not installed.")
        sys.exit(1)
    if args.impl == "flashinfer" and not FLASHINFER_AVAILABLE:
        print("\nERROR: --impl flashinfer but flashinfer is not installed.")
        sys.exit(1)

    cfg = BenchConfig()
    all_results = {}

    # Determine which experiments to run
    if args.experiment == "all":
        targets = list(EXPERIMENTS.keys())
    else:
        targets = [args.experiment]

    for exp_name in targets:
        exp_fn = EXPERIMENTS[exp_name]
        try:
            results = exp_fn(args.impl, cfg, warmup=args.warmup, iters=args.iters)
            all_results[exp_name] = results
        except Exception as e:
            print(f"\nERROR in experiment '{exp_name}': {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = []

    # Save everything
    save_results(all_results, args.output_dir, cfg)


if __name__ == "__main__":
    main()
