# Benchmark Critique and Precision Handling Audit

## Part 1: Benchmark Critique for Conference Receptiveness

### What's Already Strong (Conference-Ready)

The `benchmark/` directory is well-structured for top-tier venues:

1. **Proper statistical methodology** (`shared/timer.py`):
   - 10 warmup iterations (accounts for JIT, thermal)
   - 100 measured iterations (sufficient for p99)
   - Bootstrap 95% CI on the median (1000 resamples)
   - Full latency distribution preserved — NeurIPS reproducibility checklist compliant

2. **Standard metrics** (`shared/metrics.py`):
   - MFU% (Model FLOPs Utilization) against H100 peak
   - HBM SOL% (Speed of Light) against 3.35 TB/s
   - Operational intensity + roofline classification
   - References FA3 (740 TFLOPS), DeepGEMM (1550 TFLOPS), FlashMLA (660/410 TFLOPS)

3. **FP8 Pareto frontier** (`fp8_pareto/bench_fp8.py`):
   - Reports BOTH speed AND quality — follows FA3's standard
   - Per-tile and global-scale quantization compared
   - RMSE normalized against FA3's claimed 2.6× improvement
   - Pareto-optimal point identification

4. **Environment capture** (`shared/report.py`):
   - GPU name, CUDA version, library versions, temperature, clock speed
   - Follows NeurIPS reproducibility checklist

5. **MoE-Inference-Bench sweep ranges** (`shared/config.py`):
   - Standard sweep ranges from SC '25
   - MLPerf v5.1 SLA thresholds (p99 TTFT < 2s, p99 TPOT < 80ms)

### Gaps and Improvements Needed

#### Gap 1: No Ablation Table Matching Paper Claims

**Problem:** The paper (Section 5, page 22) claims "single Chinese node achieves performance comparable to dual-GPU international clusters" and "50% cost reduction in long-sequence scenarios." No benchmark currently validates these specific claims.

**Fix:** Add `benchmark/ablation/` with:
- Single Ascend node vs 2× H100 comparison (throughput at B=32, T=4096/16384/65536)
- Long-sequence cost amortization: tokens/second/dollar at increasing context lengths
- MLAPO on/off ablation (paper mentions `VLLM_ASCEND_ENABLE_MLAPO` env var)

#### Gap 2: Missing DSA Indexer Quality Metrics

**Problem:** The FP8 Pareto benchmark measures MLA attention and MoE quality, but NOT the DSA indexer's quality. The indexer's FP8 scoring introduces quantization error that could change WHICH tokens are selected — this is a selection-quality problem, not just a numerical-precision problem.

**Fix:** Add to `bench_fp8.py`:
```python
# Indexer quality: Jaccard similarity between BF16 and FP8 token selections
bf16_logits = indexer_bf16(q, k, weights)
fp8_logits  = indexer_fp8(q_fp8, k_fp8, weights)
bf16_topk = bf16_logits.topk(2048).indices
fp8_topk  = fp8_logits.topk(2048).indices
jaccard = len(set(bf16_topk) & set(fp8_topk)) / len(set(bf16_topk) | set(fp8_topk))
# Target: Jaccard > 0.95 at all sequence lengths
```

#### Gap 3: No End-to-End Quality Measurement (Perplexity/Generation)

**Problem:** All benchmarks measure component speed and numerical precision, but NOT whether the model actually produces good text. A conference reviewer would ask: "Does FP8 quantization hurt generation quality?"

**Fix:** Add `benchmark/quality/bench_generation.py`:
- Compute perplexity on a held-out dataset (e.g., WikiText-103) for BF16 vs FP8
- Run a small eval suite (MMLU 5-shot, HumanEval) comparing BF16 and W4A8
- Report PPL degradation per quantization level

#### Gap 4: Missing Training-Inference Consistency Test

**Problem:** Paper Section 2.4.3 describes INT4 QAT with "bitwise-identical behavior between training and inference." Our W4A8 quantization in `fp8_utils.py` is a post-training function — it does NOT guarantee bitwise-identical behavior with the training-time quantization kernel.

**Fix:** Document this gap explicitly. If training checkpoints are available, add a test that compares:
- Our offline W4A8 quantization output
- The training-time quantized output from the checkpoint
- Any discrepancy means our quantization is wrong for that precision level

#### Gap 5: Thermal Stability Not Integrated into Main Benchmarks

**Problem:** The `h100_test_thermal.py` test runs separately from the main benchmarks. A sustained benchmark could show thermal degradation that inflates the reported median.

**Fix:** In `cuda_timer_extended`, add an optional thermal monitoring mode:
```python
# Record GPU temperature at start and end of measurement window
# Flag if temperature delta > 5°C (indicates thermal ramp-up)
# Report first-third vs last-third median ratio
```

#### Gap 6: No Cross-Implementation Quality Comparison Table

**Problem:** The 3-way benchmark (`h100_bench_3way.py`) compares SPEED but not QUALITY across implementations. Conference reviewers want to see: "Does the faster implementation sacrifice accuracy?"

**Fix:** Add a quality column to the 3-way table:
```
Component        Raw (ms)  Triton (ms)  Kernels (ms)  Speedup  Quality (cos_sim)
MLA Attention    12.3      12.3         2.1           5.9×     0.998
DSA Indexer      8.7       8.7          1.2           7.2×     Jaccard 0.97
MoE Forward      45.6      45.6         6.8           6.7×     0.999
```

---

## Part 2: Precision Handling Audit

### FP8 Format Inventory

| Location | Format | Scale Strategy | Precision |
|----------|--------|---------------|-----------|
| `flashmla-deepgemm/fp8_utils.py` `quantize_kv_flashmla()` | 656-byte interleaved | Per-128-dim-tile, power-of-2 (UE8M0) | FP8 E4M3 nope, BF16 rope |
| `flashmla-deepgemm/fp8_utils.py` `quantize_activations_deepgemm()` | Separate (tensor, scales) | Per-128-dim-block, float32 | FP8 E4M3 |
| `flashinfer/fp8_utils.py` `quantize_kv_flashinfer()` | 576-byte contiguous | Global (single float32 per tensor) | FP8 E4M3 (both nope + rope) |
| `flashinfer/fp8_utils.py` `quantize_activations_deepgemm()` | Same as FlashMLA path | Per-128-dim-block, float32 | FP8 E4M3 |
| `benchmark/fp8_pareto/bench_fp8.py` `quantise_fp8_global()` | Global scale | Single float32 per tensor | FP8 E4M3 |
| `benchmark/fp8_pareto/bench_fp8.py` `quantise_fp8_per_tile()` | Per-tile | Power-of-2 via int8 exponent | FP8 E4M3 |

### Precision Issues Found

#### Issue 1: Scale Direction Inconsistency

**FlashMLA path** (`quantize_kv_flashmla`): computes `scale_inv = amax / 448` then `quantized = data / scale_inv`. The scale stored IS the inverse scale (multiply to dequantize).

**Benchmark** (`quantise_fp8_global`): computes `scale = 448 / amax` then `quantized = data * scale`. The scale stored is the FORWARD scale (divide to dequantize).

These are mathematically equivalent but the convention is **reversed**. If the kernel expects one convention and gets the other, all values are wrong by a factor of `(amax/448)²`.

**Impact:** Low — each function is self-consistent. But if someone mixes `quantize_kv_flashmla` output with `dequantise_fp8_global`, they'll get garbage.

**Fix:** Add explicit docstrings: "This function uses INVERSE scaling (divide during quantize, multiply during dequantize)" vs "This function uses FORWARD scaling (multiply during quantize, divide during dequantize)."

#### Issue 2: FlashInfer Global Scale Loses Precision

**Problem:** `quantize_kv_flashinfer` uses a single global scale for the entire [num_pages, page_size, 576] tensor. If one token has a large outlier (e.g., in the attention sink position), the scale is dominated by that outlier, and all other tokens lose dynamic range.

**Comparison:**
- FlashMLA: per-128-tile scaling → each tile gets its own scale → 4× more precision headroom
- FlashInfer: global scaling → one outlier ruins the entire cache's precision
- DeepGEMM: per-128-block scaling → same precision as FlashMLA

**Impact:** HIGH for long sequences. At 200K context, the probability of an extreme outlier grows. The attention sink (position 0) often has much larger activation magnitudes.

**Fix:** Consider per-page or per-row scaling for FlashInfer format instead of global. Or document this as a known limitation with a warning in the README.

#### Issue 3: RoPE Quantization Asymmetry

**FlashMLA format:** RoPE dims (64) stored in **BF16** (unquantized) — 128 bytes per token.
**FlashInfer format:** RoPE dims (64) stored in **FP8** (quantized) — 64 bytes per token.

The paper (Section 3.2) shows that deterministic TopK is critical for RL. RoPE carries positional information — if FP8 quantization of the 64 RoPE dims introduces enough error, the model may confuse positional ordering, degrading the quality of long-range attention.

**Impact:** Medium — untested. Our `h100_test_precision_chain.py` tests generic activations, not specifically RoPE precision.

**Fix:** Add a targeted test in `h100_test_fp8_edge_cases.py`:
```python
def h100_test_fp8_rope_positional_ordering():
    """Verify that FP8-quantized RoPE keys still produce correct relative ordering."""
    # Create two sequences differing only in position
    # Verify that attention scores preserve the correct positional order
    # after FP8 quantization of the KV cache
```

#### Issue 4: INT4 QAT Not Represented

**Problem:** Paper Section 2.4.3 describes "INT4 QAT in the SFT stage" with a kernel that ensures "bitwise-identical behavior between training and inference." Our codebase only has:
- W4A8 post-training quantization (in the Ascend deployment configs)
- FP8 quantization utilities (for activations)

We have no INT4 quantization code and no way to verify training-inference consistency.

**Impact:** Critical for any claim about W4A8 quality. Our W4A8 quantization may NOT match the model's training-time quantization.

**Fix:** This requires access to the actual training-time INT4 quantization code (likely in MindSpeed or the model checkpoint format). Document as an open gap.

#### Issue 5: No E5M2 or E4M3FNUZ Handling

**Problem:** The codebase exclusively uses `torch.float8_e4m3fn` (E4M3). Some hardware platforms (AMD ROCm) use E4M3FNUZ or E5M2 for different stages:
- E4M3: better for weights/activations (more mantissa precision)
- E5M2: better for gradients (more exponent range)
- E4M3FNUZ: AMD-specific variant (flush-to-zero NaN)

**Impact:** Low for inference (E4M3 is standard). High for training FP8 (would need E5M2 for gradients).

---

## Part 3: Conference-Specific Recommendations

### For NeurIPS (Systems track)

**Required additions:**
1. Roofline plot (operational intensity vs achieved throughput) — currently computed but not plotted
2. Ablation table: each fusion kernel's contribution to overall speedup
3. Memory breakdown: static (weights) vs dynamic (activations) vs KV cache at 32K/128K/200K
4. Scalability: throughput vs number of GPUs (1, 2, 4, 8)

### For ICML (ML Efficiency workshop)

**Required additions:**
1. Quality vs speed Pareto curve with error bars (bootstrap CI)
2. Training cost reduction quantification (FLOPs saved by DSA at each context length)
3. Comparison to alternative efficient attention methods from paper Table 5

### For ACL (Computation and Language)

**Required additions:**
1. Generation quality: BLEU/ROUGE/pass@k on downstream tasks at each precision level
2. Long-context retrieval accuracy (RULER benchmark) at BF16 vs W4A8 vs FP8
3. Latency distribution (histogram, not just percentiles) for serving workloads

---

## Summary of Required Changes

| Priority | Change | Files Affected | Effort |
|----------|--------|---------------|--------|
| HIGH | Add DSA indexer Jaccard quality metric to FP8 Pareto | `bench_fp8.py` | Small |
| HIGH | Document scale direction convention in FP8 utils | Both `fp8_utils.py` | Trivial |
| HIGH | Add quality column to 3-way benchmark | `h100_bench_3way.py` | Small |
| MEDIUM | Add RoPE positional ordering FP8 test | `h100_test_fp8_edge_cases.py` | Small |
| MEDIUM | Add per-page scaling option for FlashInfer FP8 | `flashinfer/fp8_utils.py` | Medium |
| MEDIUM | Add end-to-end generation quality benchmark | New file | Large |
| LOW | Document INT4 QAT gap | README | Trivial |
| LOW | Add roofline plot generation | `shared/report.py` | Medium |
| LOW | Add MLAPO on/off ablation benchmark | New file | Medium |
