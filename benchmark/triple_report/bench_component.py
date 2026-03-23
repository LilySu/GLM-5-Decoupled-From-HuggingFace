"""Triple Report Level 2: Component Integration Benchmark.

Tests how kernels perform within a FULL DECODER LAYER — not in isolation.
This captures inter-kernel overhead: quantization boundaries, memory allocation,
Python dispatch between components, and data format conversions.

Methodology: Run a single decoder layer forward pass (attention + MoE/dense FFN)
and measure total time vs sum of component times. The difference is "glue overhead."

References:
- MegaBlocks (MLSys '23): MoE kernels must be evaluated on full pipeline, not just GEMM
- DistServe (OSDI '24): Disaggregated prefill/decode evaluation
"""

import argparse
import sys
import os
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    cuda_timer_extended, BenchConfig, BenchResult,
    save_results, capture_environment, compute_mfu,
    compute_attention_flops, compute_moe_flops,
)
from benchmark.shared.config import GLM5_CONFIG, H100_SPECS
from benchmark.shared.report import print_summary_table


def bench_single_layer(layer_type: str, B: int, S: int, T: int,
                       impl: str, cfg: dict, warmup: int = 10, iters: int = 50):
    """Benchmark a single decoder layer (attention + MLP/MoE + norms + residuals).

    Args:
        layer_type: "dense" (layers 0-2) or "sparse" (layers 3-77, MoE)
        B: batch size
        S: sequence length (prefill) or 1 (decode)
        T: context length (KV cache length for decode)
        impl: "flashmla" | "flashinfer" | "eager"
        cfg: model config dict
    """
    device = torch.device("cuda")

    # Import model layer based on implementation
    try:
        if impl in ("flashmla", "eager"):
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "glm5-kernels-flashmla-deepgemm"))
            from model import DecoderLayer
        elif impl == "flashinfer":
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "glm5-kernels-flashinfer"))
            from model import DecoderLayer
        else:
            raise ValueError(f"Unknown impl: {impl}")
    except ImportError as e:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            is_oom=False, error=f"Import failed: {e}",
        )

    # Build a single layer
    try:
        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 1
        test_cfg["mlp_layer_types"] = [layer_type]
        if impl == "eager":
            test_cfg["use_flash_mla"] = False
            test_cfg["use_deepgemm"] = False

        layer = DecoderLayer(test_cfg, layer_idx=0).to(device).bfloat16().eval()
        hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        position_ids = torch.arange(T - S, T, device=device).unsqueeze(0).expand(B, -1)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = layer(hidden, position_ids=position_ids)
            torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            is_oom=True,
        )
    except Exception as e:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            error=str(e),
        )

    # Benchmark
    def run():
        with torch.no_grad():
            _ = layer(hidden, position_ids=position_ids)

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            is_oom=True,
        )

    # Compute FLOPs for the layer
    H = cfg["num_attention_heads"]
    d_qk = cfg.get("d_qk_absorbed", 576)
    d_v = cfg.get("d_v_absorbed", 512)
    attn_flops = compute_attention_flops(B, H, S, T, d_qk, d_v)

    if layer_type == "sparse":
        moe_flops = compute_moe_flops(
            B * S, cfg["num_experts_per_tok"],
            cfg["hidden_size"], cfg["moe_intermediate_size"]
        )
    else:
        # Dense FFN: 2 matmuls (gate+up fused, down)
        moe_flops = 2 * B * S * cfg["hidden_size"] * cfg["intermediate_size"] * 3

    total_flops = attn_flops + moe_flops
    latency_s = stats["median"] / 1000.0

    return BenchResult(
        name=f"layer_{layer_type}",
        impl=impl,
        config={"B": B, "S": S, "T": T, "type": layer_type},
        latency_ms=times,
        median_ms=stats["median"],
        mean_ms=stats["mean"],
        std_ms=stats["std"],
        p5_ms=stats["p5"],
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"],
        ci_95_high=stats["ci_95_high"],
        tflops=total_flops / latency_s / 1e12 if latency_s > 0 else 0,
        mfu_pct=compute_mfu(total_flops, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_component_benchmark(output_dir: str = "results/triple_report"):
    """Run the component integration benchmark.

    Tests both dense and MoE layers across decode and prefill modes.
    """
    results = []
    cfg = GLM5_CONFIG

    configs = [
        # Decode: B=32, S=1, T=4096 (memory-bound)
        {"B": 32, "S": 1, "T": 4096, "label": "decode_B32_T4K"},
        # Prefill: B=1, S=128, T=128 (compute-bound)
        {"B": 1, "S": 128, "T": 128, "label": "prefill_B1_S128"},
        # Long context decode: B=4, S=1, T=16384
        {"B": 4, "S": 1, "T": 16384, "label": "decode_B4_T16K"},
    ]

    for c in configs:
        for layer_type in ["dense", "sparse"]:
            for impl in ["eager", "flashmla", "flashinfer"]:
                print(f"  {c['label']} | {layer_type} | {impl}...", end=" ", flush=True)
                result = bench_single_layer(
                    layer_type, c["B"], c["S"], c["T"], impl, cfg,
                )
                if result.is_oom:
                    print("OOM")
                elif result.error:
                    print(f"ERROR: {result.error}")
                else:
                    print(f"{result.median_ms:.3f} ms | {result.mfu_pct:.1f}% MFU")
                results.append(result)

                torch.cuda.empty_cache()

    # Report
    print_summary_table(results, "Triple Report Level 2: Component Integration")
    env = capture_environment()
    save_results(results, output_dir, "triple_report_component", env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Report Level 2: Component Integration")
    parser.add_argument("--output-dir", default="results/triple_report")
    args = parser.parse_args()
    run_component_benchmark(args.output_dir)
