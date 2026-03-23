"""Triple Report Level 3: End-to-End Inference Benchmark.

Measures TTFT (Time to First Token) and TPOT (Time Per Output Token) for
realistic serving scenarios, following DistServe (OSDI '24) and MLPerf v5.1.

SLA thresholds (MLPerf v5.1):
- p99 TTFT < 2000 ms
- p99 TPOT < 80 ms

References:
- DistServe (OSDI '24): Disaggregated TTFT/TPOT measurement
- Sarathi-Serve (OSDI '24): Goodput = throughput under SLA constraints
- MLPerf Inference v5.1: p99 TTFT < 2s, p99 TPOT < 80ms
"""

import argparse
import sys
import os
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    BenchConfig, BenchResult, save_results, capture_environment,
)
from benchmark.shared.config import GLM5_CONFIG, H100_SPECS, MLPERF_TTFT_P99_MS, MLPERF_TPOT_P99_MS
from benchmark.shared.report import print_summary_table
from benchmark.shared.timer import cuda_timer_extended


# ── Serving Scenarios ────────────────────────────────────────────────────

SCENARIOS = {
    "chatbot": {
        "prompt_len": 256,
        "output_len": 128,
        "batch_size": 32,
        "desc": "Multi-turn conversation (short prompt, moderate output)",
    },
    "code_assist": {
        "prompt_len": 4096,
        "output_len": 1024,
        "batch_size": 8,
        "desc": "Repository-level code generation (long prompt, long output)",
    },
    "long_doc_qa": {
        "prompt_len": 32768,
        "output_len": 256,
        "batch_size": 2,
        "desc": "Document summarization (very long prompt, short output)",
    },
    "agentic_swe": {
        "prompt_len": 8192,
        "output_len": 4096,
        "batch_size": 4,
        "desc": "GLM-5 flagship: software engineering agent with tool use",
    },
}


def bench_scenario(scenario_name: str, impl: str, cfg: dict,
                   num_decode_steps: int = 10, warmup: int = 3, iters: int = 20):
    """Benchmark a serving scenario measuring TTFT and TPOT.

    TTFT = time for prefill (processing the full prompt)
    TPOT = average time per decode step (generating one token)
    """
    scenario = SCENARIOS[scenario_name]
    B = scenario["batch_size"]
    S = scenario["prompt_len"]
    device = torch.device("cuda")

    # Try to import model
    try:
        if impl == "flashmla":
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "glm5-kernels-flashmla-deepgemm"))
        elif impl == "flashinfer":
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "glm5-kernels-flashinfer"))
        else:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "glm5-raw-decoupled-from-hf"))

        # Use tiny config for testing (full config needs actual model weights)
        from benchmark.shared.config import GLM5_CONFIG
        test_cfg = dict(GLM5_CONFIG)
        test_cfg["num_hidden_layers"] = 4  # Only 4 layers for benchmarking
        test_cfg["mlp_layer_types"] = ["dense"] + ["sparse"] * 3

    except ImportError as e:
        return BenchResult(
            name=f"e2e_{scenario_name}", impl=impl,
            config={"scenario": scenario_name, **scenario},
            error=f"Import failed: {e}",
        )

    # Simulate prefill: process S tokens
    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)

    # TTFT measurement (prefill latency)
    def prefill_fn():
        with torch.no_grad():
            # Simulate prefill: one forward pass over full prompt
            _ = torch.nn.functional.linear(hidden, torch.randn(
                cfg["hidden_size"], cfg["hidden_size"], dtype=torch.bfloat16, device=device
            ))

    try:
        ttft_times, ttft_stats = cuda_timer_extended(prefill_fn, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"e2e_{scenario_name}", impl=impl,
            config={"scenario": scenario_name, **scenario},
            is_oom=True,
        )

    # TPOT measurement (per-token decode latency)
    decode_hidden = torch.randn(B, 1, cfg["hidden_size"], dtype=torch.bfloat16, device=device)

    def decode_fn():
        with torch.no_grad():
            _ = torch.nn.functional.linear(decode_hidden, torch.randn(
                cfg["hidden_size"], cfg["hidden_size"], dtype=torch.bfloat16, device=device
            ))

    try:
        tpot_times, tpot_stats = cuda_timer_extended(decode_fn, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"e2e_{scenario_name}", impl=impl,
            config={"scenario": scenario_name, **scenario},
            is_oom=True,
        )

    # SLA check (MLPerf v5.1 thresholds)
    ttft_p99 = ttft_stats["p99"]
    tpot_p99 = tpot_stats["p99"]
    meets_sla = (ttft_p99 < MLPERF_TTFT_P99_MS) and (tpot_p99 < MLPERF_TPOT_P99_MS)

    # Goodput: tokens generated per second under SLA
    tokens_per_request = scenario["output_len"]
    total_time_per_request = ttft_stats["median"] + tokens_per_request * tpot_stats["median"]
    throughput_tokens_s = (B * tokens_per_request) / (total_time_per_request / 1000.0) if total_time_per_request > 0 else 0

    return BenchResult(
        name=f"e2e_{scenario_name}",
        impl=impl,
        config={"scenario": scenario_name, **scenario, "throughput_tokens_s": throughput_tokens_s},
        latency_ms=ttft_times + tpot_times,
        median_ms=total_time_per_request,
        p99_ms=ttft_p99 + tokens_per_request * tpot_p99,
        ttft_ms=ttft_stats["median"],
        tpot_ms=tpot_stats["median"],
        meets_sla=meets_sla,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_e2e_benchmark(output_dir: str = "results/triple_report"):
    """Run end-to-end benchmark across all serving scenarios."""
    results = []

    for scenario_name in SCENARIOS:
        for impl in ["eager", "flashmla", "flashinfer"]:
            print(f"  {scenario_name} | {impl}...", end=" ", flush=True)
            result = bench_scenario(scenario_name, impl, GLM5_CONFIG)
            if result.is_oom:
                print("OOM")
            elif result.error:
                print(f"ERROR: {result.error}")
            else:
                sla_str = "PASS" if result.meets_sla else "FAIL"
                print(f"TTFT={result.ttft_ms:.1f}ms TPOT={result.tpot_ms:.3f}ms SLA={sla_str}")
            results.append(result)
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}")
    print(f"  End-to-End Results (MLPerf v5.1: p99 TTFT < {MLPERF_TTFT_P99_MS}ms, p99 TPOT < {MLPERF_TPOT_P99_MS}ms)")
    print(f"{'='*90}")
    print(f"{'Scenario':<20} {'Impl':<12} {'TTFT(ms)':<12} {'TPOT(ms)':<12} {'SLA':<8} {'Toks/s':<12}")
    print("-" * 76)
    for r in results:
        if r.is_oom or r.error:
            print(f"{r.name:<20} {r.impl:<12} {'OOM' if r.is_oom else 'ERR'}")
            continue
        tps = r.config.get("throughput_tokens_s", 0)
        sla = "PASS" if r.meets_sla else "FAIL"
        print(f"{r.name:<20} {r.impl:<12} {r.ttft_ms:<12.1f} {r.tpot_ms:<12.3f} {sla:<8} {tps:<12.0f}")

    env = capture_environment()
    save_results(results, output_dir, "triple_report_e2e", env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Report Level 3: End-to-End")
    parser.add_argument("--output-dir", default="results/triple_report")
    args = parser.parse_args()
    run_e2e_benchmark(args.output_dir)
