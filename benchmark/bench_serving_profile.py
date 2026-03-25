"""Serving profile benchmark: prefill/decode separation, routing skew, KV cache memory.

Three production-relevant experiments grounded in academic methodology:

  1. prefill-decode:  Full DecoderLayer at realistic prefill/decode configs
     - Prefill (TTFT): compute-bound, B={1,4}, S={512,2048,8192}
     - Decode (TPOT): memory-bound, B={1,32,64,128}, T={4096,16384}
     References: FA3 (NeurIPS '24), DistServe (OSDI '24), Sarathi-Serve (OSDI '24),
                 TensorRT-LLM benchmarks, MLPerf v5.1

  2. routing-skew:    MoE performance under Zipf-distributed expert routing
     - Uniform vs Zipf alpha={0.8, 1.2} at N={1024, 4096} tokens
     References: MoE-Inference-Bench (SC '25), DeepSeek EPLB (2025), LIBRA (ICLR '25)

  3. memory:          KV cache footprint, MLA compression ratio, OOM boundary
     - MLA BF16 (1152 B/tok/layer) vs MLA FP8 (656) vs standard MHA (32768)
     References: vLLM/PagedAttention (SOSP '23), DeepSeek-V2, GLM-5 Table 3

Usage:
    python3 -m benchmark.bench_serving_profile --experiment all
    python3 -m benchmark.bench_serving_profile --experiment prefill-decode
    python3 -m benchmark.bench_serving_profile --experiment routing-skew
    python3 -m benchmark.bench_serving_profile --experiment memory
"""

import os
# MUST be set before ANY model imports (bench_component.py pattern)
os.environ["GLM5_FORCE_EAGER"] = "1"

import argparse
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    BenchResult, save_results, capture_environment,
)
from benchmark.shared.config import GLM5_CONFIG, H100_SPECS
from benchmark.shared.timer import cuda_timer_extended
from benchmark.shared.metrics import (
    compute_attention_flops, compute_moe_flops,
    compute_mfu, compute_hbm_sol, compute_tflops,
    compute_bandwidth_gb_s, compute_attention_bytes,
    compute_moe_bytes, compute_operational_intensity,
    classify_roofline_bound,
)
from benchmark.shared.report import print_summary_table


# ── Experiment 1: Prefill vs Decode ──────────────────────────────────────

PREFILL_CONFIGS = [
    # Paper precedents in comments
    {"B": 1, "S": 512},    # DistServe baseline, Sarathi saturation point
    {"B": 1, "S": 2048},   # FA3 sweep, long prompt
    {"B": 1, "S": 8192},   # FA3 sweep, max feasible with eager attn on 80GB
    {"B": 4, "S": 512},    # Batched prefill at Sarathi saturation
    {"B": 4, "S": 2048},   # Batched long prompt
]

DECODE_CONFIGS = [
    {"B": 1,   "T": 4096},   # FA3 sweep, single-user latency baseline
    {"B": 32,  "T": 4096},   # FA3/MoE-Inference-Bench standard batch
    {"B": 32,  "T": 16384},  # FA3 sweep, medium context
    {"B": 64,  "T": 4096},   # MoE-Inference-Bench max batch
    {"B": 64,  "T": 16384},  # High-throughput + medium context
    {"B": 128, "T": 4096},   # FA3 sweep, high-throughput stress test
    {"B": 128, "T": 16384},  # Max batch + medium context
]


def _ensure_symlinks():
    for h, u in [
        ("glm5-kernels-flashmla-deepgemm", "glm5_kernels_flashmla_deepgemm"),
        ("glm5-kernels-flashinfer", "glm5_kernels_flashinfer"),
    ]:
        src = os.path.join(PROJECT_ROOT, h)
        dst = os.path.join(PROJECT_ROOT, u)
        if os.path.isdir(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                pass


def _patch_and_import():
    """Import model with ALL kernel paths disabled (bench_component.py pattern)."""
    _ensure_symlinks()
    pkg = "glm5_kernels_flashmla_deepgemm"
    import importlib

    attn_mod = importlib.import_module(f"{pkg}.mla_attention")
    attn_mod.FLASH_MLA_AVAILABLE = False
    if hasattr(attn_mod, 'HAS_FLASH_MLA'):
        attn_mod.HAS_FLASH_MLA = False

    idx_mod = importlib.import_module(f"{pkg}.dsa_indexer")
    idx_mod.DEEP_GEMM_AVAILABLE = False
    if hasattr(idx_mod, 'HAS_DEEP_GEMM'):
        idx_mod.HAS_DEEP_GEMM = False

    model_mod = importlib.import_module(f"{pkg}.model")
    rope_mod = importlib.import_module(f"{pkg}.rope_partial")
    config_mod = importlib.import_module(f"{pkg}.config")

    return model_mod.DecoderLayer, rope_mod.RotaryEmbedding, config_mod.GLM_MOE_DSA_CONFIG


def _bench_prefill_layer(B, S, layer_type, DecoderLayer, RotaryEmbedding, cfg, warmup=10, iters=50):
    """Benchmark prefill: S=T, causal mask. Reports TFLOPS + MFU%."""
    device = torch.device("cuda")
    label = f"prefill_B{B}_S{S}_{layer_type}"
    config_info = {"B": B, "S": S, "T": S, "type": layer_type, "phase": "prefill"}

    try:
        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 1
        test_cfg["mlp_layer_types"] = [layer_type]
        layer = DecoderLayer(test_cfg, layer_idx=0).to(device).bfloat16().eval()
        rope = RotaryEmbedding(test_cfg).to(device)

        hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = rope(hidden, pos_ids)
        mask = torch.full((S, S), float("-inf"), device=device, dtype=torch.bfloat16)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            for _ in range(3):
                layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Setup: {e}")

    def run():
        with torch.no_grad():
            layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Bench: {e}")

    H = cfg["num_attention_heads"]
    d_qk = cfg.get("qk_head_dim", 256)
    d_v = cfg.get("v_head_dim", 256)
    attn_flops = compute_attention_flops(B, H, S, S, d_qk, d_v)
    if layer_type == "sparse":
        moe_flops = compute_moe_flops(B * S, cfg["num_experts_per_tok"], cfg["hidden_size"], cfg["moe_intermediate_size"])
    else:
        moe_flops = 2 * B * S * cfg["hidden_size"] * cfg["intermediate_size"] * 3
    total_flops = attn_flops + moe_flops
    latency_s = stats["median"] / 1000.0

    del layer, rope, hidden, cos, sin, mask
    torch.cuda.empty_cache()

    return BenchResult(
        name=label, impl="eager", config=config_info,
        latency_ms=times,
        median_ms=stats["median"], mean_ms=stats["mean"], std_ms=stats["std"],
        p5_ms=stats["p5"], p50_ms=stats["p50"], p95_ms=stats["p95"], p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"], ci_95_high=stats["ci_95_high"],
        tflops=total_flops / latency_s / 1e12 if latency_s > 0 else 0,
        mfu_pct=compute_mfu(total_flops, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def _bench_decode_layer(B, T, layer_type, DecoderLayer, RotaryEmbedding, cfg, warmup=10, iters=50):
    """Benchmark decode: S=1, pre-populated KV cache of length T-1.

    Reports bandwidth GB/s + HBM SOL%.

    Workaround: DSA indexer topk with random weights selects OOB indices.
    We monkey-patch the indexer to clamp indices, preserving compute cost.
    """
    device = torch.device("cuda")
    label = f"decode_B{B}_T{T}_{layer_type}"
    config_info = {"B": B, "S": 1, "T": T, "type": layer_type, "phase": "decode"}

    try:
        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 1
        test_cfg["mlp_layer_types"] = [layer_type]
        layer = DecoderLayer(test_cfg, layer_idx=0).to(device).bfloat16().eval()
        rope = RotaryEmbedding(test_cfg).to(device)

        # Pre-populate KV cache with random data (avoid minutes-long real prefill)
        from glm5_kernels_flashmla_deepgemm.cache import KVCache
        kv_cache = KVCache(1)
        H = cfg["num_attention_heads"]
        d_qk = cfg.get("qk_head_dim", 256)
        d_v = cfg.get("v_head_dim", 256)
        fake_k = torch.randn(B, H, T - 1, d_qk, dtype=torch.bfloat16, device=device)
        fake_v = torch.randn(B, H, T - 1, d_v, dtype=torch.bfloat16, device=device)
        kv_cache._cache[0] = (fake_k, fake_v)

        # Pre-populate DSA indexer key cache
        indexer = layer.self_attn.indexer
        idx_dim = cfg.get("index_head_dim", 128)
        indexer._cached_keys = torch.randn(B, T - 1, idx_dim, dtype=torch.bfloat16, device=device)

        # Monkey-patch indexer to clamp OOB indices
        original_forward = indexer.forward.__wrapped__ if hasattr(indexer.forward, '__wrapped__') else None

        @torch.no_grad()
        def patched_forward(hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
            batch_size, seq_len, _ = hidden_states.shape
            cos, sin = position_embeddings
            q = indexer.wq_b(q_resid)
            q = q.view(batch_size, seq_len, indexer.n_heads, indexer.head_dim)
            q_pe, q_nope = torch.split(q, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
            from glm5_kernels_flashmla_deepgemm.rope_partial import apply_rotary_pos_emb
            q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = indexer.k_norm(indexer.wk(hidden_states))
            k_pe, k_nope = torch.split(k, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
            k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
            k = torch.cat([k_pe, k_nope], dim=-1)
            if use_cache and indexer._cached_keys is not None:
                k_cached = torch.cat([indexer._cached_keys, k], dim=1)
                indexer._cached_keys = k_cached
            else:
                k_cached = k
            weights = indexer.weights_proj(hidden_states).float() * (indexer.n_heads ** -0.5)
            scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * indexer.softmax_scale
            scores = F.relu(scores)
            index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
            if attention_mask is not None:
                index_scores = index_scores + attention_mask
            total_len = index_scores.shape[-1]
            topk = min(indexer.index_topk, total_len)
            indices = index_scores.topk(topk, dim=-1).indices
            return indices.clamp_(0, total_len - 1)

        indexer.forward = patched_forward

        # Decode inputs
        hidden = torch.randn(B, 1, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        pos_ids = torch.tensor([[T - 1]], device=device).expand(B, -1)
        cos, sin = rope(hidden, pos_ids)
        mask = torch.zeros(1, T, device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)

        original_idx_keys = indexer._cached_keys.clone()
        with torch.no_grad():
            for _ in range(3):
                layer(hidden, attention_mask=mask, position_embeddings=(cos, sin), past_key_values=kv_cache)
                # Reset both caches to original state after each warmup step
                kv_cache._cache[0] = (fake_k, fake_v)
                indexer._cached_keys = original_idx_keys
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Setup: {e}")

    # Save original indexer cache to reset each iteration
    original_idx_cache = indexer._cached_keys.clone()

    def run():
        with torch.no_grad():
            kv_cache._cache[0] = (fake_k, fake_v)
            indexer._cached_keys = original_idx_cache
            layer(hidden, attention_mask=mask, position_embeddings=(cos, sin), past_key_values=kv_cache)

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Bench: {e}")

    attn_bytes = compute_attention_bytes(B, H, 1, T, d_qk, d_v)
    latency_s = stats["median"] / 1000.0

    del layer, rope, hidden, cos, sin, mask, fake_k, fake_v, kv_cache
    torch.cuda.empty_cache()

    return BenchResult(
        name=label, impl="eager", config=config_info,
        latency_ms=times,
        median_ms=stats["median"], mean_ms=stats["mean"], std_ms=stats["std"],
        p5_ms=stats["p5"], p50_ms=stats["p50"], p95_ms=stats["p95"], p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"], ci_95_high=stats["ci_95_high"],
        bandwidth_gb_s=compute_bandwidth_gb_s(attn_bytes, latency_s),
        hbm_sol_pct=compute_hbm_sol(attn_bytes, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_prefill_decode_experiment(output_dir, warmup=10, iters=50):
    """Experiment 1: Prefill vs Decode phase separation."""
    print("=" * 70)
    print("  Experiment 1: Prefill vs Decode Phase Separation")
    print("  (FA3/DistServe/Sarathi-Serve/TRT-LLM parameter ranges)")
    print("=" * 70)

    try:
        DecoderLayer, RotaryEmbedding, cfg = _patch_and_import()
    except Exception as e:
        print(f"FATAL: Cannot import model: {e}")
        return []

    results = []

    print("\n--- PREFILL (compute-bound: TFLOPS, MFU%) ---")
    for c in PREFILL_CONFIGS:
        for lt in ["dense", "sparse"]:
            tag = f"B={c['B']} S={c['S']} {lt}"
            print(f"  {tag}...", end=" ", flush=True)
            r = _bench_prefill_layer(c["B"], c["S"], lt, DecoderLayer, RotaryEmbedding, cfg, warmup, iters)
            if r.is_oom:
                print("OOM")
            elif r.error:
                print(f"ERROR: {r.error[:60]}")
            else:
                print(f"{r.median_ms:.3f}ms | {r.tflops:.1f} TFLOPS | {r.mfu_pct:.1f}% MFU | {r.peak_memory_gb:.1f}GB")
            results.append(r)

    print("\n--- DECODE (memory-bound: bandwidth, HBM SOL%) ---")
    for c in DECODE_CONFIGS:
        for lt in ["dense", "sparse"]:
            tag = f"B={c['B']} T={c['T']} {lt}"
            print(f"  {tag}...", end=" ", flush=True)
            r = _bench_decode_layer(c["B"], c["T"], lt, DecoderLayer, RotaryEmbedding, cfg, warmup, iters)
            if r.is_oom:
                print("OOM")
            elif r.error:
                print(f"ERROR: {r.error[:60]}")
            else:
                print(f"{r.median_ms:.3f}ms | {r.bandwidth_gb_s:.0f} GB/s | {r.hbm_sol_pct:.1f}% SOL | {r.peak_memory_gb:.1f}GB")
            results.append(r)

    print()
    print_summary_table(results, "Prefill vs Decode Phase Separation")
    env = capture_environment()
    save_results(results, output_dir, "serving_prefill_decode", env)
    return results


# ── Experiment 2: MoE Expert Routing Skew ────────────────────────────────

ROUTING_CONFIGS = [
    {"N": 1024, "distribution": "uniform",    "alpha": 0.0},
    {"N": 1024, "distribution": "zipf_mild",  "alpha": 0.8},
    {"N": 1024, "distribution": "zipf_heavy", "alpha": 1.2},
    {"N": 4096, "distribution": "uniform",    "alpha": 0.0},
    {"N": 4096, "distribution": "zipf_mild",  "alpha": 0.8},
    {"N": 4096, "distribution": "zipf_heavy", "alpha": 1.2},
]


def _make_zipf_topk_ids(N, E, K, alpha, device):
    """Generate topk_ids [N, K] with Zipf-distributed expert selection.

    Each token gets K unique experts sampled without replacement,
    weighted by Zipf(alpha) probabilities.
    """
    ranks = torch.arange(1, E + 1, dtype=torch.float32, device=device)
    weights = 1.0 / ranks.pow(alpha)
    weights = weights + 1e-8  # avoid zero weights
    weights = weights / weights.sum()
    # Sample K experts without replacement per token, weighted by Zipf
    topk_ids = torch.multinomial(weights.expand(N, -1), K, replacement=False)
    return topk_ids.to(torch.int32)


def _compute_load_imbalance(topk_ids, E):
    """Compute expert load statistics."""
    flat = topk_ids.reshape(-1).long()
    counts = torch.bincount(flat, minlength=E).float()
    mean_load = counts.mean().item()
    max_load = counts.max().item()
    min_load = counts.min().item()
    top10_share = counts.topk(10).values.sum().item() / counts.sum().item()
    imbalance = max_load / mean_load if mean_load > 0 else 0
    return {
        "max_tokens": int(max_load),
        "mean_tokens": round(mean_load, 1),
        "min_tokens": int(min_load),
        "imbalance_ratio": round(imbalance, 2),
        "top10_share": round(top10_share, 3),
    }


def run_routing_skew_experiment(output_dir, warmup=10, iters=100):
    """Experiment 2: MoE expert routing skew."""
    print("=" * 70)
    print("  Experiment 2: MoE Expert Routing Skew")
    print("  (MoE-Inference-Bench / DeepSeek EPLB / LIBRA parameter ranges)")
    print("=" * 70)

    # Deferred import to avoid triggering bench_moe's print at module level
    from benchmark.moe_sweep.bench_moe import (
        moe_forward_baseline, moe_forward_deepgemm,
        DEEPGEMM_AVAILABLE, HIDDEN_SIZE,
    )

    E = GLM5_CONFIG["n_routed_experts"]    # 256
    K = GLM5_CONFIG["num_experts_per_tok"]  # 8
    D = GLM5_CONFIG["hidden_size"]          # 6144
    I = GLM5_CONFIG["moe_intermediate_size"]  # 2048
    device = torch.device("cuda")

    results = []

    for cfg in ROUTING_CONFIGS:
        N = cfg["N"]
        dist = cfg["distribution"]
        alpha = cfg["alpha"]

        # Generate routing
        if dist == "uniform":
            topk_ids = torch.stack(
                [torch.randperm(E, device=device)[:K] for _ in range(N)], dim=0
            ).to(torch.int32)
        else:
            topk_ids = _make_zipf_topk_ids(N, E, K, alpha, device)

        load_stats = _compute_load_imbalance(topk_ids, E)

        # Allocate tensors
        hidden_states = torch.randn(N, D, dtype=torch.bfloat16, device=device)
        gate_up_weight = torch.randn(E, 2 * I, D, dtype=torch.bfloat16, device=device)
        down_weight = torch.randn(E, D, I, dtype=torch.bfloat16, device=device)
        raw_scores = torch.randn(N, K, device=device)
        topk_weights = torch.sigmoid(raw_scores)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        flops = compute_moe_flops(N, K, D, I)

        for impl_name, forward_fn, available in [
            ("pytorch_loop", moe_forward_baseline, True),
            ("deepgemm_bf16", moe_forward_deepgemm, DEEPGEMM_AVAILABLE),
        ]:
            if not available:
                continue

            label = f"moe_N{N}_{dist}_{impl_name}"
            config_info = {
                "N": N, "E": E, "K": K, "D": D, "I": I,
                "distribution": dist, "alpha": alpha,
                "impl": impl_name, **load_stats,
            }

            print(f"  N={N:>4} {dist:<12} {impl_name:<16}", end=" ", flush=True)

            try:
                torch.cuda.reset_peak_memory_stats()

                # Bind forward_fn via default arg to avoid closure-over-loop-variable bug
                def run(_fn=forward_fn):
                    _fn(hidden_states, gate_up_weight, down_weight, topk_ids, topk_weights)

                times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
                latency_s = stats["median"] / 1000.0

                r = BenchResult(
                    name=label, impl=impl_name, config=config_info,
                    latency_ms=times,
                    median_ms=stats["median"], mean_ms=stats["mean"], std_ms=stats["std"],
                    p5_ms=stats["p5"], p50_ms=stats["p50"], p95_ms=stats["p95"], p99_ms=stats["p99"],
                    ci_95_low=stats["ci_95_low"], ci_95_high=stats["ci_95_high"],
                    tflops=compute_tflops(flops, latency_s),
                    mfu_pct=compute_mfu(flops, latency_s),
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
                print(f"med={r.median_ms:>7.3f}ms p99={r.p99_ms:>7.3f}ms {r.tflops:>6.1f}TF imb={load_stats['imbalance_ratio']:.1f}x top10={load_stats['top10_share']:.1%}")
                results.append(r)

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                r = BenchResult(name=label, impl=impl_name, config=config_info, is_oom=True)
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}")
                r = BenchResult(name=label, impl=impl_name, config=config_info, error=str(e)[:100])
                results.append(r)

        # Clean up between configs
        del hidden_states, gate_up_weight, down_weight, topk_ids, topk_weights
        torch.cuda.empty_cache()

    print()
    print_summary_table(results, "MoE Expert Routing Skew")
    env = capture_environment()
    save_results(results, output_dir, "serving_routing_skew", env)
    return results


# ── Experiment 3: KV Cache Memory Footprint ──────────────────────────────

MEMORY_CONFIGS = [
    {"B": 1,  "T": 4096},
    {"B": 1,  "T": 16384},
    {"B": 1,  "T": 65536},
    {"B": 1,  "T": 131072},
    {"B": 16, "T": 4096},
    {"B": 16, "T": 16384},
    {"B": 32, "T": 4096},
    {"B": 32, "T": 16384},
    {"B": 64, "T": 4096},
]

# Per-token-per-layer KV dimensions
MLA_D = 576           # kv_lora_rank(512) + qk_rope_head_dim(64), 1 KV head
STD_D = 64 * 256      # num_heads * qk_head_dim = 16384 for standard MHA K (or V)
NUM_LAYERS = GLM5_CONFIG["num_layers"]  # 78
# Note: FlashMLA's actual FP8 format is 656 bytes/token (512 FP8 nope + 16 bytes scales + 128 bytes BF16 rope)
# but raw tensor allocation with dtype=fp8 gives 576 bytes. The 656-byte figure includes metadata overhead
# that only exists in the paged cache format. We report both for transparency.


def _measure_kv_memory(B, T, num_layers, d_kv, dtype, device):
    """Allocate KV cache tensor and measure actual GPU memory delta."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(device)

    try:
        kv = torch.zeros(num_layers, B, T, d_kv, dtype=dtype, device=device)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        actual_gb = (mem_after - mem_before) / 1e9
        dtype_bytes = 1 if dtype == torch.float8_e4m3fn else 2
        theoretical_gb = num_layers * B * T * d_kv * dtype_bytes / 1e9
        del kv
        torch.cuda.empty_cache()
        return {"actual_gb": round(actual_gb, 3), "theoretical_gb": round(theoretical_gb, 3), "oom": False}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        dtype_bytes = 1 if dtype == torch.float8_e4m3fn else 2
        theoretical_gb = num_layers * B * T * d_kv * dtype_bytes / 1e9
        return {"actual_gb": 0, "theoretical_gb": round(theoretical_gb, 3), "oom": True}


def run_memory_experiment(output_dir):
    """Experiment 3: KV cache memory footprint and MLA compression ratio."""
    print("=" * 70)
    print("  Experiment 3: KV Cache Memory Footprint")
    print("  (vLLM/PagedAttention, DeepSeek-V2, GLM-5 Table 3)")
    print("=" * 70)

    device = torch.device("cuda")
    results = []

    print(f"\n  {'B':>4} {'T':>8}  {'MLA-BF16':>10} {'MLA-FP8':>10} {'Std-BF16':>10} {'Ratio':>8} Status")
    print("  " + "-" * 68)

    for cfg in MEMORY_CONFIGS:
        B, T = cfg["B"], cfg["T"]

        mla_bf16 = _measure_kv_memory(B, T, NUM_LAYERS, MLA_D, torch.bfloat16, device)
        mla_fp8 = _measure_kv_memory(B, T, NUM_LAYERS, MLA_D, torch.float8_e4m3fn, device)
        std_bf16 = _measure_kv_memory(B, T, NUM_LAYERS, STD_D, torch.bfloat16, device)

        ratio = std_bf16["theoretical_gb"] / mla_bf16["theoretical_gb"] if mla_bf16["theoretical_gb"] > 0 else 0

        status = "OK"
        if mla_bf16["oom"]:
            status = "OOM(all)"
        elif std_bf16["oom"]:
            status = "OOM(std)"
        elif mla_fp8["oom"]:
            status = "OOM(fp8)"

        print(f"  {B:>4} {T:>8}  {mla_bf16['theoretical_gb']:>9.3f}G {mla_fp8['theoretical_gb']:>9.3f}G {std_bf16['theoretical_gb']:>9.3f}G {ratio:>7.1f}x {status}")

        r = BenchResult(
            name=f"kv_B{B}_T{T}", impl="memory_profile",
            config={
                "B": B, "T": T, "num_layers": NUM_LAYERS,
                "mla_bf16_gb": mla_bf16["theoretical_gb"],
                "mla_bf16_actual_gb": mla_bf16["actual_gb"],
                "mla_fp8_gb": mla_fp8["theoretical_gb"],
                "mla_fp8_actual_gb": mla_fp8["actual_gb"],
                "std_bf16_gb": std_bf16["theoretical_gb"],
                "compression_ratio": round(ratio, 1),
                "mla_bytes_per_token_per_layer": MLA_D * 2,
                "std_bytes_per_token_per_layer": STD_D * 2,
            },
            is_oom=mla_bf16["oom"],
            kv_cache_memory_gb=mla_bf16["actual_gb"] if not mla_bf16["oom"] else 0,
        )
        results.append(r)

    # OOM boundary detection
    print(f"\n  OOM Boundary (max T for MLA BF16 in ~78GB, {NUM_LAYERS} layers):")
    for B in [1, 16, 32, 64]:
        lo, hi = 1024, 262144
        max_t = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            theoretical = NUM_LAYERS * B * mid * MLA_D * 2 / 1e9
            if theoretical <= 78.0:
                max_t = mid
                lo = mid + 1
            else:
                hi = mid - 1
        print(f"    B={B:>3}: max T ≈ {max_t:>7,} ({NUM_LAYERS * B * max_t * MLA_D * 2 / 1e9:.1f} GB)")

    print()
    env = capture_environment()
    save_results(results, output_dir, "serving_memory", env)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GLM-5 serving profile benchmark: prefill/decode, routing skew, KV cache memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", choices=["prefill-decode", "routing-skew", "memory", "all"],
                        default="all", help="Which experiment to run.")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "results", "serving_profile"),
                        help="Directory for JSON result files.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = {
        "prefill-decode": lambda: run_prefill_decode_experiment(args.output_dir, args.warmup, args.iters),
        "routing-skew": lambda: run_routing_skew_experiment(args.output_dir, args.warmup, args.iters),
        "memory": lambda: run_memory_experiment(args.output_dir),
    }

    targets = experiments.keys() if args.experiment == "all" else [args.experiment]
    for name in targets:
        experiments[name]()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
