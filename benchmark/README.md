# GLM-5 Benchmark Suite

Academic-grade benchmark harness for H100 GPUs, aligned with MLSys/OSDI/NeurIPS 2024-2025 evaluation standards.

## Structure

```
benchmark/
  shared/                    # Shared utilities and metrics
    __init__.py
    timer.py                 # Extended CUDA timer (100 iters, bootstrap CI, p99)
    metrics.py               # MFU, HBM SOL%, roofline, FLOPs computation
    config.py                # GLM-5 dims + H100 hardware constants
    report.py                # JSON output + environment snapshot
  moe_sweep/                 # MoE-Inference-Bench style sweeps (SC '25 standard)
    __init__.py
    bench_moe.py             # Batch × tokens × experts × FFN dim sweeps
  triple_report/             # Triple Report: micro → component → end-to-end
    __init__.py
    bench_micro.py           # Kernel-level TFLOPS per component
    bench_component.py       # Full decoder layer integration
    bench_e2e.py             # End-to-end inference (TTFT + TPOT)
  mfu_ceiling/               # MFU relative to FA3's 75% ceiling
    __init__.py
    bench_mfu.py             # MFU at various (B, T, precision) configs
  fp8_pareto/                # FP8 speed-quality Pareto frontier
    __init__.py
    bench_fp8.py             # TFLOPS + cosine similarity at each precision
  run_all.py                 # Orchestrator: run all experiments
```

## Quick Start

```bash
# Run MoE sweeps only (~1 hour):
python -m benchmark.moe_sweep.bench_moe

# Run triple report (~30 min):
python -m benchmark.triple_report.bench_micro
python -m benchmark.triple_report.bench_component
python -m benchmark.triple_report.bench_e2e

# Run MFU ceiling analysis (~20 min):
python -m benchmark.mfu_ceiling.bench_mfu

# Run FP8 Pareto frontier (~30 min):
python -m benchmark.fp8_pareto.bench_fp8

# Run everything (~3 hours):
python -m benchmark.run_all --output-dir results/
```

## Methodology

Aligned with:
- **MoE-Inference-Bench (SC '25)**: batch {1,16,32,64}, tokens {128,256,512,1024,2048}
- **FlashAttention-3 (NeurIPS 2024)**: MFU as % of peak, 75% reference ceiling
- **Sarathi-Serve (OSDI '24)**: TTFT, TPOT, goodput under SLA
- **MLPerf v5.1**: p99 TTFT < 2s, p99 TPOT < 80ms

Statistical: 10 warmup, 100 iterations, bootstrap 95% CI, Mann-Whitney U for comparisons.
