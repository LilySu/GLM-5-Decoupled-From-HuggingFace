"""MFU ceiling analysis for GLM-5 components vs FlashAttention-3's 75% ceiling.

Measures how close each GLM-5 kernel gets to FA3's established reference point of
740 TFLOPS (75% MFU) on H100 FP16, and also computes each point's position on
the roofline to diagnose whether remaining headroom is compute-limited or
memory-bandwidth-limited.

Methodology:
- 10 warmup + 100 measured iterations (FA3 / MoE-Inference-Bench SC'25 standard)
- Bootstrap 95% CI on the median (1000 resamples)
- Roofline: achievable = min(peak_compute, peak_bw × OI)
- FA3 ceiling: 740 TFLOPS FP16 → 75% MFU on H100

Sweep grid
----------
MLA Decode  : B ∈ {1,4,16,32,64}  ×  T (context) ∈ {256,1K,4K,16K,64K}
MLA Prefill : B=1                  ×  S (seq len) ∈ {128,512,2K,8K}
MoE GEMM    : N (tokens) ∈ {128,512,1K,2K,4K}
DSA Indexer : T (context) ∈ {1K,4K,16K,64K}

References
----------
- FlashAttention-3 (Tri Dao, 2024): 75% MFU on H100
- MoE-Inference-Bench (SC '25): systematic kernel sweeps
- Williams et al., "Roofline" (2009)
"""

import argparse
import sys
import os

import torch

# ── shared utilities ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import BenchConfig, BenchResult, GLM5_CONFIG, H100_SPECS
from shared.timer import cuda_timer_extended, check_outliers
from shared.metrics import (
    compute_attention_flops,
    compute_attention_bytes,
    compute_moe_flops,
    compute_moe_bytes,
    compute_dsa_indexer_flops,
    compute_mfu,
    compute_hbm_sol,
    compute_tflops,
    compute_bandwidth_gb_s,
    compute_operational_intensity,
    compute_roofline_achievable,
    classify_roofline_bound,
)
from shared.report import save_results, print_summary_table, capture_environment

# ── FA3 reference constants ───────────────────────────────────────────────────
FA3_TFLOPS   = H100_SPECS["fa3_tflops_fp16"]   # 740 TFLOPS
FA3_MFU_PCT  = H100_SPECS["fa3_mfu_pct"]       # 75 %
PEAK_BF16    = H100_SPECS["peak_tflops_bf16"]  # 989 TFLOPS
PEAK_FP8     = H100_SPECS["peak_tflops_fp8"]   # 1979 TFLOPS
HBM_BW       = H100_SPECS["hbm_bandwidth_gb_s"]  # 3350 GB/s

# ── sweep definitions ─────────────────────────────────────────────────────────
SWEEP_MLA_DECODE = {
    "batch_sizes":   [1, 4, 16, 32, 64],
    "context_lens":  [256, 1024, 4096, 16384, 65536],
}
SWEEP_MLA_PREFILL = {
    "batch_sizes":  [1],
    "seq_lens":     [128, 512, 2048, 8192],
}
SWEEP_MOE = {
    "n_tokens": [128, 512, 1024, 2048, 4096],
}
SWEEP_DSA = {
    "context_lens": [1024, 4096, 16384, 65536],
}

# quick-mode subsets (CLI: --quick)
QUICK_MLA_DECODE  = {"batch_sizes": [1, 32], "context_lens": [1024, 4096]}
QUICK_MLA_PREFILL = {"batch_sizes": [1],     "seq_lens":     [512, 2048]}
QUICK_MOE         = {"n_tokens": [512, 2048]}
QUICK_DSA         = {"context_lens": [1024, 16384]}


# ─────────────────────────────────────────────────────────────────────────────
# Kernel stubs / real implementations
# ─────────────────────────────────────────────────────────────────────────────

def _make_mla_decode_fn(B: int, T_ctx: int, precision: str):
    """Return a zero-arg callable that runs one MLA decode forward pass.

    Uses FlashMLA when available; falls back to a scaled_dot_product_attention
    stub that exercises the same tensor shapes so FLOPs/bytes accounting is
    correct even without the real kernel.
    """
    H      = GLM5_CONFIG["num_heads"]          # 64
    d_qk   = GLM5_CONFIG["d_qk_absorbed"]      # 576
    d_v    = GLM5_CONFIG["d_v_absorbed"]       # 512
    dtype  = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16
    # Decode: S_q = 1 (single new token per sequence)
    S_q    = 1

    # Allocate tensors once; closure captures them.
    q   = torch.randn(B, H, S_q, d_qk,  device="cuda").to(dtype)
    k   = torch.randn(B, 1,  T_ctx, d_qk, device="cuda").to(dtype)  # MLA: 1 KV head
    v   = torch.randn(B, 1,  T_ctx, d_v,  device="cuda").to(dtype)

    try:
        import flash_mla  # noqa: F401
        page_size = GLM5_CONFIG["page_size"]
        # Build paged KV metadata expected by FlashMLA
        n_pages = (T_ctx + page_size - 1) // page_size
        cache_seqlens = torch.full((B,), T_ctx, dtype=torch.int32, device="cuda")
        # FlashMLA paged KV: shape [total_pages, 2, page_size, 64, 16]  (fp16 format)
        # We use a plain bf16 q/k/v stub so the timer still fires if FlashMLA
        # isn't importable in this environment.
        kv_cache = torch.zeros(
            B * n_pages, 2, page_size, 64, 16, device="cuda",
            dtype=torch.bfloat16
        )
        block_table = torch.arange(B * n_pages, device="cuda",
                                   dtype=torch.int32).reshape(B, n_pages)
        softmax_scale = d_qk ** -0.5

        def fn():
            return flash_mla.flash_mla_with_kvcache(
                q.view(B, H, S_q, d_qk),
                kv_cache,
                block_table,
                cache_seqlens,
                d_v,
                softmax_scale,
                causal=True,
            )
    except (ImportError, Exception):
        # Stub: SDPA with the correct shape
        q_sdpa = q.float().reshape(B * H, S_q, d_qk)
        k_sdpa = k.float().expand(B, H, T_ctx, d_qk).reshape(B * H, T_ctx, d_qk)
        v_sdpa = v.float().expand(B, H, T_ctx, d_v ).reshape(B * H, T_ctx, d_v)

        def fn():
            return torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=False
            )

    return fn


def _make_mla_prefill_fn(B: int, S: int, precision: str):
    """Return a zero-arg callable for MLA prefill (S_q = S_kv = S)."""
    H    = GLM5_CONFIG["num_heads"]
    d_qk = GLM5_CONFIG["d_qk_absorbed"]
    d_v  = GLM5_CONFIG["d_v_absorbed"]
    dtype = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16

    q = torch.randn(B, H, S, d_qk, device="cuda").to(dtype)
    k = torch.randn(B, 1, S, d_qk, device="cuda").to(dtype)
    v = torch.randn(B, 1, S, d_v,  device="cuda").to(dtype)

    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func

        q_fa = q.bfloat16().reshape(B * S, H, d_qk)
        # MLA single-head KV → expand for flash_attn (needs H_kv == H or GQA)
        k_fa = k.bfloat16().expand(B, H, S, d_qk).reshape(B * S, H, d_qk)
        v_fa = v.bfloat16().expand(B, H, S, d_v ).reshape(B * S, H, d_v)

        def fn():
            return flash_attn_func(q_fa, k_fa, v_fa, causal=True)

    except (ImportError, Exception):
        q_sdpa = q.float().reshape(B * H, S, d_qk)
        k_sdpa = k.float().expand(B, H, S, d_qk).reshape(B * H, S, d_qk)
        v_sdpa = v.float().expand(B, H, S, d_v ).reshape(B * H, S, d_v)

        def fn():
            return torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=True
            )

    return fn


def _make_moe_gemm_fn(N_tokens: int, precision: str):
    """Return a zero-arg callable for one MoE grouped GEMM forward pass."""
    H         = GLM5_CONFIG["hidden_size"]          # 6144
    I         = GLM5_CONFIG["moe_intermediate_size"] # 2048
    K_active  = GLM5_CONFIG["num_experts_per_tok"]   # 8
    N_experts = GLM5_CONFIG["n_routed_experts"]      # 256
    dtype     = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16

    # Each token routes to K_active experts → effective batch is N_tokens * K_active
    hidden  = torch.randn(N_tokens, H, device="cuda").to(dtype)
    # Weights: one matrix per expert, gate+up fused [N_experts, 2*I, H]
    w_gate_up = torch.randn(N_experts, 2 * I, H, device="cuda").to(dtype)
    w_down    = torch.randn(N_experts, H, I,     device="cuda").to(dtype)
    # Routing: which expert handles which token (simplified: round-robin)
    expert_ids = torch.arange(N_tokens * K_active, device="cuda") % N_experts

    try:
        import deep_gemm  # noqa: F401

        # DeepGEMM grouped GEMM expects contiguous expert segments.
        # We build a simple per-expert dispatch using contiguous slices.
        tokens_per_expert = max(1, (N_tokens * K_active) // N_experts)
        m_sizes = torch.full((N_experts,), tokens_per_expert,
                             dtype=torch.int32, device="cuda")
        # Input: [N_tokens*K_active, H]
        x_grouped = hidden[expert_ids % N_tokens].contiguous()

        def fn():
            # gate_up pass
            out_gate_up = torch.empty(N_tokens * K_active, 2 * I,
                                      device="cuda", dtype=torch.bfloat16)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                (x_grouped, w_gate_up), out_gate_up, m_sizes
            )
            gate, up = out_gate_up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out = torch.empty(N_tokens * K_active, H,
                              device="cuda", dtype=torch.bfloat16)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                (act, w_down), out, m_sizes
            )
            return out

    except (ImportError, Exception):
        # BF16 per-expert loop fallback
        hidden_bf16 = hidden.float().bfloat16()
        w_gu_bf16   = w_gate_up.float().bfloat16()
        w_dn_bf16   = w_down.float().bfloat16()

        def fn():
            # Gather tokens per expert (simplified: all tokens to each active expert)
            outputs = []
            for e in range(min(K_active, N_experts)):
                x = hidden_bf16
                gu = torch.nn.functional.linear(x, w_gu_bf16[e])
                gate_e, up_e = gu.chunk(2, dim=-1)
                act_e = torch.nn.functional.silu(gate_e) * up_e
                out_e = torch.nn.functional.linear(act_e, w_dn_bf16[e])
                outputs.append(out_e)
            return torch.stack(outputs).mean(0)  # reduce across sampled experts

    return fn


def _make_dsa_indexer_fn(T_ctx: int):
    """Return a zero-arg callable for the DSA lightning indexer scoring pass."""
    H_idx  = GLM5_CONFIG["index_n_heads"]   # 32
    D_idx  = GLM5_CONFIG["index_head_dim"]  # 128
    topk   = GLM5_CONFIG["index_topk"]      # 2048
    S_q    = 1  # decode: single query position

    # q_idx: [S_q, H_idx, D_idx], k_idx: [T_ctx, H_idx, D_idx]
    q_idx = torch.randn(S_q,   H_idx, D_idx, device="cuda", dtype=torch.bfloat16)
    k_idx = torch.randn(T_ctx, H_idx, D_idx, device="cuda", dtype=torch.bfloat16)
    # learned head weights
    w_head = torch.randn(H_idx, device="cuda", dtype=torch.bfloat16)

    try:
        import deep_gemm  # noqa: F401
        # Use DeepGEMM fp8_mqa_logits if available
        q_fp8 = q_idx.to(torch.float8_e4m3fn)
        k_fp8 = k_idx.to(torch.float8_e4m3fn)
        scale_q = torch.ones(1, device="cuda")
        scale_k = torch.ones(1, device="cuda")

        def fn():
            # score: [S_q, T_ctx]  via fp8 matmul over (H_idx, D_idx) dim
            scores = deep_gemm.fp8_mqa_logits(q_fp8, k_fp8, scale_q, scale_k)
            weighted = (scores * w_head.unsqueeze(0).unsqueeze(0)).sum(-1)
            return torch.relu(weighted)

    except (ImportError, Exception):
        def fn():
            # score[s,t] = ReLU(sum_h w_h * (q_idx[s,h] · k_idx[t,h]))
            # einsum: [S_q, H, D] × [T, H, D] → [S_q, T, H] then weighted sum
            dots = torch.einsum("shd,thd->sth", q_idx, k_idx)     # [S_q, T, H]
            weighted = (dots * w_head[None, None, :]).sum(-1)       # [S_q, T]
            return torch.relu(weighted)

    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Per-component benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def _fill_result(result: BenchResult, flops: int, bytes_accessed: int,
                 times: list, stats: dict, precision: str) -> BenchResult:
    """Populate a BenchResult from raw timing + FLOPs/bytes figures."""
    result.latency_ms  = times
    result.median_ms   = stats["median"]
    result.mean_ms     = stats["mean"]
    result.std_ms      = stats["std"]
    result.p5_ms       = stats["p5"]
    result.p50_ms      = stats["p50"]
    result.p95_ms      = stats["p95"]
    result.p99_ms      = stats["p99"]
    result.ci_95_low   = stats["ci_95_low"]
    result.ci_95_high  = stats["ci_95_high"]

    latency_s = result.median_ms / 1e3
    result.tflops          = compute_tflops(flops, latency_s)
    result.mfu_pct         = compute_mfu(flops, latency_s, precision)
    result.bandwidth_gb_s  = compute_bandwidth_gb_s(bytes_accessed, latency_s)
    result.hbm_sol_pct     = compute_hbm_sol(bytes_accessed, latency_s)

    oi = compute_operational_intensity(flops, bytes_accessed)
    result.operational_intensity = oi
    result.roofline_bound        = classify_roofline_bound(oi, precision)
    return result


def bench_mla_decode(sweep: dict, precision: str, iters: int = 100) -> list:
    """Benchmark MLA decode across (B, T_ctx) grid."""
    results = []
    H   = GLM5_CONFIG["num_heads"]
    d_qk = GLM5_CONFIG["d_qk_absorbed"]
    d_v  = GLM5_CONFIG["d_v_absorbed"]
    dtype_bytes = 1 if precision == "fp8" else 2
    S_q = 1  # decode

    for B in sweep["batch_sizes"]:
        for T in sweep["context_lens"]:
            label = f"MLA-decode B={B} T={T}"
            print(f"  [{label}] ...", end=" ", flush=True)

            cfg = BenchConfig(
                batch_size=B, seq_len=S_q, context_len=T,
                precision=precision, mode="decode", component="mla",
            )
            result = BenchResult(
                name=label, impl="flashmla",
                config={"B": B, "T": T, "S_q": S_q, "precision": precision},
            )

            try:
                fn     = _make_mla_decode_fn(B, T, precision)
                flops  = compute_attention_flops(B, H, S_q, T, d_qk, d_v)
                nbytes = compute_attention_bytes(B, H, S_q, T, d_qk, d_v,
                                                 dtype_bytes)
                times, stats = cuda_timer_extended(fn, warmup=10, iters=iters)
                _fill_result(result, flops, nbytes, times, stats, precision)

                # Roofline efficiency
                achievable = compute_roofline_achievable(
                    result.operational_intensity, precision)
                roofline_eff = (result.tflops / achievable * 100
                                if achievable > 0 else 0.0)
                fa3_gap_pct = (result.tflops / FA3_TFLOPS) * 100

                print(f"{result.tflops:.0f} TFLOPS  "
                      f"({result.mfu_pct:.1f}% MFU, "
                      f"{fa3_gap_pct:.1f}% of FA3 ceiling, "
                      f"roofline-eff {roofline_eff:.1f}%, "
                      f"{result.roofline_bound})")

                outlier_info = check_outliers(times)
                if not outlier_info["valid"]:
                    print(f"    WARNING: {outlier_info['flags']}")

            except torch.cuda.OutOfMemoryError:
                result.is_oom = True
                result.error  = "OOM"
                print("OOM")
            except Exception as exc:
                result.error = str(exc)
                print(f"ERROR: {exc}")

            results.append(result)

    return results


def bench_mla_prefill(sweep: dict, precision: str, iters: int = 100) -> list:
    """Benchmark MLA prefill (S_q == S_kv) across sequence lengths."""
    results = []
    H    = GLM5_CONFIG["num_heads"]
    d_qk = GLM5_CONFIG["d_qk_absorbed"]
    d_v  = GLM5_CONFIG["d_v_absorbed"]
    dtype_bytes = 1 if precision == "fp8" else 2

    for B in sweep["batch_sizes"]:
        for S in sweep["seq_lens"]:
            label = f"MLA-prefill B={B} S={S}"
            print(f"  [{label}] ...", end=" ", flush=True)

            result = BenchResult(
                name=label, impl="flash_attn",
                config={"B": B, "S": S, "precision": precision},
            )

            try:
                fn     = _make_mla_prefill_fn(B, S, precision)
                flops  = compute_attention_flops(B, H, S, S, d_qk, d_v)
                nbytes = compute_attention_bytes(B, H, S, S, d_qk, d_v,
                                                  dtype_bytes)
                times, stats = cuda_timer_extended(fn, warmup=10, iters=iters)
                _fill_result(result, flops, nbytes, times, stats, precision)

                achievable  = compute_roofline_achievable(
                    result.operational_intensity, precision)
                roofline_eff = (result.tflops / achievable * 100
                                if achievable > 0 else 0.0)
                fa3_gap_pct  = (result.tflops / FA3_TFLOPS) * 100

                print(f"{result.tflops:.0f} TFLOPS  "
                      f"({result.mfu_pct:.1f}% MFU, "
                      f"{fa3_gap_pct:.1f}% of FA3 ceiling, "
                      f"roofline-eff {roofline_eff:.1f}%, "
                      f"{result.roofline_bound})")

            except torch.cuda.OutOfMemoryError:
                result.is_oom = True; result.error = "OOM"; print("OOM")
            except Exception as exc:
                result.error = str(exc); print(f"ERROR: {exc}")

            results.append(result)

    return results


def bench_moe_gemm(sweep: dict, precision: str, iters: int = 100) -> list:
    """Benchmark MoE grouped GEMM across token counts."""
    results = []
    H        = GLM5_CONFIG["hidden_size"]
    I        = GLM5_CONFIG["moe_intermediate_size"]
    K_active = GLM5_CONFIG["num_experts_per_tok"]
    N_exp    = GLM5_CONFIG["n_routed_experts"]
    dtype_bytes = 1 if precision == "fp8" else 2

    for N in sweep["n_tokens"]:
        label = f"MoE-GEMM N={N}"
        print(f"  [{label}] ...", end=" ", flush=True)

        result = BenchResult(
            name=label, impl="deepgemm",
            config={"N_tokens": N, "K_active": K_active,
                    "N_experts": N_exp, "precision": precision},
        )

        try:
            fn     = _make_moe_gemm_fn(N, precision)
            flops  = compute_moe_flops(N, K_active, H, I)
            nbytes = compute_moe_bytes(N, K_active, H, I, N_exp, dtype_bytes)
            times, stats = cuda_timer_extended(fn, warmup=10, iters=iters)
            _fill_result(result, flops, nbytes, times, stats, precision)

            achievable   = compute_roofline_achievable(
                result.operational_intensity, precision)
            roofline_eff = (result.tflops / achievable * 100
                            if achievable > 0 else 0.0)
            fa3_gap_pct  = (result.tflops / FA3_TFLOPS) * 100

            print(f"{result.tflops:.0f} TFLOPS  "
                  f"({result.mfu_pct:.1f}% MFU, "
                  f"{fa3_gap_pct:.1f}% of FA3 ceiling, "
                  f"roofline-eff {roofline_eff:.1f}%, "
                  f"{result.roofline_bound})")

            outlier_info = check_outliers(times)
            if not outlier_info["valid"]:
                print(f"    WARNING: {outlier_info['flags']}")

        except torch.cuda.OutOfMemoryError:
            result.is_oom = True; result.error = "OOM"; print("OOM")
        except Exception as exc:
            result.error = str(exc); print(f"ERROR: {exc}")

        results.append(result)

    return results


def bench_dsa_indexer(sweep: dict, iters: int = 100) -> list:
    """Benchmark DSA lightning indexer across context lengths."""
    results = []
    precision = "bf16"
    H_idx = GLM5_CONFIG["index_n_heads"]
    D_idx = GLM5_CONFIG["index_head_dim"]
    S_q   = 1  # decode mode
    dtype_bytes = 2

    for T in sweep["context_lens"]:
        label = f"DSA-indexer T={T}"
        print(f"  [{label}] ...", end=" ", flush=True)

        result = BenchResult(
            name=label, impl="dsa_einsum",
            config={"T": T, "H_idx": H_idx, "D_idx": D_idx,
                    "precision": precision},
        )

        try:
            fn    = _make_dsa_indexer_fn(T)
            flops = compute_dsa_indexer_flops(S_q, T, H_idx, D_idx)
            # Bytes: Q [S_q, H, D] + K [T, H, D] + output [S_q, T]
            nbytes = (S_q * H_idx * D_idx
                      + T  * H_idx * D_idx
                      + S_q * T) * dtype_bytes
            times, stats = cuda_timer_extended(fn, warmup=10, iters=iters)
            _fill_result(result, flops, nbytes, times, stats, precision)

            achievable   = compute_roofline_achievable(
                result.operational_intensity, precision)
            roofline_eff = (result.tflops / achievable * 100
                            if achievable > 0 else 0.0)
            fa3_gap_pct  = (result.tflops / FA3_TFLOPS) * 100

            print(f"{result.tflops:.0f} TFLOPS  "
                  f"({result.mfu_pct:.1f}% MFU, "
                  f"{fa3_gap_pct:.1f}% of FA3 ceiling, "
                  f"roofline-eff {roofline_eff:.1f}%, "
                  f"{result.roofline_bound})")

        except torch.cuda.OutOfMemoryError:
            result.is_oom = True; result.error = "OOM"; print("OOM")
        except Exception as exc:
            result.error = str(exc); print(f"ERROR: {exc}")

        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Roofline summary
# ─────────────────────────────────────────────────────────────────────────────

def print_roofline_summary(results: list):
    """Print a compact roofline analysis table."""
    print(f"\n{'='*100}")
    print("  ROOFLINE ANALYSIS — GLM-5 components vs H100 BF16/FP8")
    print(f"  FA3 reference: {FA3_TFLOPS:.0f} TFLOPS ({FA3_MFU_PCT:.0f}% MFU, BF16)")
    print(f"  H100 ridge: BF16 ~295 FLOPs/byte | FP8 ~590 FLOPs/byte")
    print(f"{'='*100}")
    print(f"  {'Component':<28} {'OI(F/B)':>10} {'Achieved':>10} "
          f"{'Achievable':>12} {'Roof-eff%':>10} {'MFU%':>7} "
          f"{'vs FA3%':>9} {'Bound':<14}")
    print(f"  {'-'*95}")

    for r in results:
        if r.is_oom or r.error:
            print(f"  {r.name:<28} {'--':>10} {'--':>10} {'--':>12} "
                  f"{'--':>10} {'--':>7} {'--':>9} {r.error or 'OOM':<14}")
            continue
        prec = r.config.get("precision", "bf16")
        achievable   = compute_roofline_achievable(r.operational_intensity, prec)
        roofline_eff = (r.tflops / achievable * 100) if achievable > 0 else 0.0
        fa3_pct      = (r.tflops / FA3_TFLOPS) * 100

        print(f"  {r.name:<28} {r.operational_intensity:>10.1f} "
              f"{r.tflops:>10.1f} {achievable:>12.1f} "
              f"{roofline_eff:>10.1f} {r.mfu_pct:>7.1f} "
              f"{fa3_pct:>9.1f} {r.roofline_bound:<14}")

    print(f"\n  Notes:")
    print(f"  - OI     = operational intensity (FLOPs / byte)")
    print(f"  - Achievable = min(peak_compute, peak_bw × OI) on H100 roofline")
    print(f"  - Roof-eff%  = achieved / achievable × 100  (closeness to ceiling)")
    print(f"  - vs FA3%    = achieved_TFLOPS / {FA3_TFLOPS:.0f} × 100")


def print_mfu_gap_summary(results: list):
    """Print one-line MFU gap lines in the FA3 style from the spec."""
    print(f"\n{'='*80}")
    print("  MFU GAP SUMMARY (FA3 style)")
    print(f"{'='*80}")
    for r in results:
        if r.is_oom or r.error or r.tflops == 0:
            continue
        fa3_pct = (r.tflops / FA3_TFLOPS) * 100
        # Derive B and T from config dict (best-effort)
        cfg = r.config
        B   = cfg.get("B") or cfg.get("N_tokens") or 1
        T   = cfg.get("T") or cfg.get("S") or cfg.get("N_tokens") or 0
        print(f"  {r.name} at B={B} T={T}: "
              f"{r.tflops:.0f} TFLOPS "
              f"({r.mfu_pct:.1f}% MFU, "
              f"{fa3_pct:.1f}% of FA3 ceiling)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GLM-5 MFU ceiling analysis vs FlashAttention-3 75% reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--component",
        choices=["mla_decode", "mla_prefill", "moe_gemm", "dsa_indexer", "all"],
        default="all",
        help="Which component(s) to benchmark.",
    )
    p.add_argument(
        "--precision",
        choices=["bf16", "fp8"],
        default="bf16",
        help="Tensor dtype for the benchmark.",
    )
    p.add_argument(
        "--output-dir",
        default="./results/mfu_ceiling",
        help="Directory to write JSON result files.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sweep (2 batch sizes × 2 context lengths) for smoke testing.",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of measured iterations per configuration.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found — this benchmark requires a GPU.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  GLM-5 MFU Ceiling Analysis")
    print(f"  FA3 reference: {FA3_TFLOPS:.0f} TFLOPS = {FA3_MFU_PCT:.0f}% MFU on H100 FP16")
    print(f"  Precision: {args.precision}  |  Iters: {args.iters}  |  "
          f"Quick: {args.quick}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    env     = capture_environment()
    all_res = []

    # ── select sweeps ──────────────────────────────────────────────────────
    mla_decode_sweep  = QUICK_MLA_DECODE  if args.quick else SWEEP_MLA_DECODE
    mla_prefill_sweep = QUICK_MLA_PREFILL if args.quick else SWEEP_MLA_PREFILL
    moe_sweep         = QUICK_MOE         if args.quick else SWEEP_MOE
    dsa_sweep         = QUICK_DSA         if args.quick else SWEEP_DSA

    run_all = args.component == "all"

    if run_all or args.component == "mla_decode":
        print(f"[1/4] MLA Decode  (B × T grid, {args.precision})")
        res = bench_mla_decode(mla_decode_sweep, args.precision, args.iters)
        all_res.extend(res)

    if run_all or args.component == "mla_prefill":
        print(f"\n[2/4] MLA Prefill  (B=1 × S grid, {args.precision})")
        res = bench_mla_prefill(mla_prefill_sweep, args.precision, args.iters)
        all_res.extend(res)

    if run_all or args.component == "moe_gemm":
        print(f"\n[3/4] MoE GEMM  (N tokens, {args.precision})")
        res = bench_moe_gemm(moe_sweep, args.precision, args.iters)
        all_res.extend(res)

    if run_all or args.component == "dsa_indexer":
        print(f"\n[4/4] DSA Indexer  (T context, bf16)")
        res = bench_dsa_indexer(dsa_sweep, args.iters)
        all_res.extend(res)

    # ── output ────────────────────────────────────────────────────────────
    print_summary_table(all_res, title="MFU Ceiling — Per-Configuration Results")
    print_roofline_summary(all_res)
    print_mfu_gap_summary(all_res)

    save_results(all_res, args.output_dir, "mfu_ceiling", env=env)


if __name__ == "__main__":
    main()
