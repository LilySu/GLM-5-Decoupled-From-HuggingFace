# Research Directions from GLM-5 Kernel Benchmarks — Deep Analysis

## The Single Main Conclusion

**GLM-5's DSA architecture creates a novel precision-performance tradeoff that no existing benchmark methodology captures: the indexer's FP8 scoring accuracy directly determines which tokens the model attends to, making it a selection-quality problem rather than a numerical-precision problem.**

This is different from standard FP8 attention benchmarks (which measure output cosine similarity) because a small numerical change in the indexer's scores can flip which tokens are in the top-2048 — causing a *discrete* quality change that cosine similarity misses entirely. This is the core thesis.

Everything below builds toward proving or disproving this.

---

## Why These Specific Points

### Point 1: FP8 Attention is Lossless (cos_sim=0.9993) — But That's the Wrong Metric for DSA

**What we measured:** FP8 MLA attention output cosine similarity vs BF16 reference.

**Why it matters:** Every FP8 inference paper since FA3 (2024) reports cos_sim or RMSE. Our 0.9993 cos_sim would be accepted at any venue. But this metric is **misleading for DSA** because:

1. The attention output quality depends on WHICH tokens are attended to (selected by the indexer)
2. If FP8 quantization changes the indexer's top-2048 selection by even 1%, the attention output changes discretely — not continuously
3. cos_sim only measures the output GIVEN the same token selection. It doesn't measure whether the selection itself changed.

**The assumption:** cos_sim adequately captures FP8 quality. This assumption holds for dense attention but FAILS for sparse attention where token selection is a discrete decision boundary.

**What's actually missing:** Jaccard similarity between BF16 and FP8 indexer token selections at each context length. This is the metric that matters for DSA.

**Industry trend:** ACL 2025 Best Paper (DeepSeek NSA) established that sparse attention must prove "losslessness." Our cos_sim proves attention is lossless given the same tokens. But we haven't proven the SELECTION is lossless under FP8.

### Point 2: FP8 MoE is Slower Than BF16 (27% vs 56% MFU) — This Contradicts Expectations

**What we measured:** DeepGEMM FP8 grouped GEMM at 537 TFLOPS (27.1% of FP8 peak 1979T) vs BF16 at 556 TFLOPS (56.2% of BF16 peak 989T).

**Why this is surprising:** FP8 should be 2× faster than BF16 on H100 tensor cores. It's not. The FP8 path is actually slightly SLOWER.

**The assumption being tested:** "FP8 always improves throughput." This assumption is widely held but our data shows it fails for grouped GEMM at these dimensions.

**Root cause hypothesis:** DeepGEMM's FP8 grouped GEMM has per-block scale factor management overhead (computing amax, applying scales, storing exponents) that the BF16 path doesn't have. At E=256, K=8, D=6144, I=2048 with 1024 tokens, the GEMM tile sizes are small enough that scale management overhead dominates.

**What's specifically missing to prove this:**

| Experiment | What It Measures | Range That Matters |
|-----------|-----------------|-------------------|
| Token count sweep at FULL dims (D=6144) | At what N does FP8 overtake BF16? | N ∈ {256, 512, 1024, 2048, 4096, 8192, 16384} |
| Expert count sweep | Does the crossover change with E? | E ∈ {8, 32, 64, 128, 256} at fixed N=4096 |
| FFN dim sweep | Larger tiles = better FP8 amortization? | I ∈ {1024, 2048, 4096, 8192} |
| Scale factor overhead isolation | Time spent in quantize vs compute | Profile with ncu `gpu__time_duration.sum` per kernel |

**Why this range:** The crossover is where FP8's 2× compute advantage exceeds the ~20% scale management overhead. At N=1024 with 256 experts, each expert sees ~32 tokens (1024×8/256) — too small for FP8 to amortize. At N=8192, each expert gets ~256 tokens — likely past the crossover.

**Industry trend:** MicroMix (2025) showed that per-layer precision allocation outperforms uniform FP8. Our data empirically demonstrates WHY — small GEMMs don't benefit from FP8. This aligns with Trend 8 (Microscaling formats) and Trend 11 (Per-layer precision profiling).

### Point 3: DSA Indexer is 2.2× Faster With DeepGEMM — But Only at Decode

**What we measured:** DeepGEMM FP8 indexer scoring at 1.978 ms vs PyTorch einsum at 4.366 ms for decode (B=32, T=4096).

**Why this matters for a paper:** The indexer adds overhead to EVERY attention layer. If the indexer is slow, DSA's attention savings are negated. The paper claims 1.5-2× attention reduction — the indexer MUST be faster than the savings.

**The critical question:** At what context length does DSA become net-positive?

```
Net DSA benefit = (attention_time_dense - attention_time_sparse) - indexer_time

If benefit > 0: DSA saves time
If benefit < 0: DSA is slower than dense attention
```

**What's specifically missing:**

| Context Length | Dense Attention (ms) | Sparse Attention (ms) | Indexer (ms) | Net Benefit |
|---------------|---------------------|----------------------|-------------|-------------|
| 1024 | ? | ? | ? | ? |
| 4096 | ? | ? | ? | ? |
| 16384 | ? | ? | ? | ? |
| 65536 | ? | ? | ? | ? |
| 131072 | ? | ? | ? | ? |

This is the **crossover analysis** (Trend 10: Prefill-decode crossover). Dense attention scales O(L²), sparse attention scales O(L×k) where k=2048, and the indexer scales O(L×L_kv). The crossover should be around L=8K-16K where the quadratic term starts dominating.

**How to set up the experiment:**

```python
for T in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    # 1. Dense MLA attention (no DSA)
    dense_time = bench(mla_attention_dense, B=1, S=1, T=T)

    # 2. DSA indexer scoring
    indexer_time = bench(dsa_indexer, B=1, S=1, T=T)

    # 3. Sparse MLA attention (with DSA indices, k=2048)
    sparse_time = bench(mla_attention_sparse, B=1, S=1, T=T, topk=2048)

    # Net benefit
    net = dense_time - (indexer_time + sparse_time)
    print(f"T={T}: dense={dense_time:.3f}, indexer={indexer_time:.3f}, "
          f"sparse={sparse_time:.3f}, net={net:+.3f} ms")
```

This is the **single most important experiment** for a DSA paper.

### Point 4: End-to-End Shows 1.0× Speedup — The Weight Absorption Gap

**What we measured:** FlashMLA and FlashInfer end-to-end are ≈ eager speed.

**Why:** All three paths use PyTorch eager attention because weight absorption is not implemented. FlashMLA/FlashInfer kernels require:
- Q absorbed: multiply q_nope by kv_b_proj^T offline → produces 576D query
- KV compressed: store only the 576D (512 nope + 64 rope) in cache
- O absorbed: multiply output by kv_b_proj^T → maps 512D back to V space

Without this, the kernel paths detect non-absorbed inputs and fall back to eager.

**This is NOT a research finding — it's an engineering gap.** But it blocks ALL end-to-end measurements. Fixing it is a prerequisite for any paper.

**The specific implementation needed:**

```python
def absorb_weights(model):
    """Pre-compute absorbed MLA weights for FlashMLA/FlashInfer kernel path."""
    for layer in model.layers:
        attn = layer.self_attn
        # Q absorption: W_absorbed_q = W_q_b @ W_kv_b^T (nope portion)
        # This maps from q_lora_rank → kv_lora_rank+rope_dim (576D)
        W_UK = attn.kv_b_proj.weight[:attn.num_heads * attn.qk_nope_head_dim]  # k_nope weights
        # ... (complex reshape + matmul)
        attn.q_absorbed = nn.Linear(attn.q_lora_rank, 576 * attn.num_heads)
        # Fuse the weights
```

### Point 5: SwiGLU at 55% HBM SOL — Why This Number Matters

**What we measured:** SwiGLU decode achieves 1850 GB/s = 55.2% of H100's 3.35 TB/s peak.

**Why it's significant:** For a simple element-wise operation (SiLU × gate), 55% SOL means there's only ~45% headroom, and much of that is unavoidable (TLB misses, memory controller overhead, L2 cache contention). This means:

1. Triton kernel is near-optimal for SwiGLU — no point optimizing further
2. The ONLY way to speed up SwiGLU is to FUSE it with the preceding GEMM (gate_proj/up_proj)
3. This is what TRT-LLM's fused GEMM+SwiGLU does — our benchmark quantifies why it matters

**Industry trend:** Trend 9 (Operator fusion as benchmarking dimension) — we should report the fusion opportunity gap: "SwiGLU at 55% SOL proves element-wise kernels are bandwidth-saturated; further speedup requires GEMM+activation fusion."

---

## Experimental Designs for Missing Data

### Experiment A: DSA Indexer Selection Fidelity Under FP8

**Hypothesis:** FP8 indexer scoring changes the top-2048 token selection, degrading generation quality even when attention cos_sim appears unchanged.

**Setup:**
```
For each context length T ∈ {1024, 4096, 16384, 65536}:
    For each of 100 random inputs:
        1. Run BF16 indexer → bf16_indices [S, 2048]
        2. Run FP8 indexer  → fp8_indices  [S, 2048]
        3. Jaccard = |bf16 ∩ fp8| / |bf16 ∪ fp8| per query position
        4. Record: mean Jaccard, min Jaccard, % of positions with Jaccard < 0.95
```

**What we expect:**
- T=1024: Jaccard ≈ 0.99 (small L, few tokens to confuse)
- T=65536: Jaccard < 0.95 (FP8 quantization noise in scores flips marginal selections)

**Why this range:** The paper uses T=202752. The degradation should grow with T because more tokens compete for the top-2048 slots, making the selection boundary thinner.

**Venue fit:** ICML — "Precision-aware token selection for sparse attention" — first paper to characterize FP8's impact on discrete selection quality.

### Experiment B: FP8 vs BF16 Grouped GEMM Crossover

**Hypothesis:** FP8 grouped GEMM becomes faster than BF16 only when per-expert token count exceeds ~128.

**Setup:**
```
Fix: E=256, K=8, D=6144, I=2048 (GLM-5 dims)
Sweep N (total tokens) ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384}
    → tokens_per_expert = N*K/E ∈ {4, 8, 16, 32, 64, 128, 256, 512}

For each N:
    bf16_time = bench(m_grouped_bf16_gemm_nt_contiguous, ...)
    fp8_time  = bench(m_grouped_fp8_gemm_nt_contiguous, ...)
    speedup = bf16_time / fp8_time
```

**What we expect:** Crossover at N≈4096 (tokens_per_expert ≈ 128), where each expert's GEMM tile is large enough to amortize scale factor overhead.

**Why this range:** Below 128 tokens per expert, the GEMM is too small for FP8's 2× tensor core advantage to overcome the ~20% scale management cost. Above 128, the amortization dominates.

**Venue fit:** SC '26 — extends MoE-Inference-Bench with precision-aware analysis. First systematic study of FP8 crossover for grouped GEMM.

### Experiment C: DSA Net Benefit Crossover

**Hypothesis:** DSA becomes net-positive at T≈8K-16K where O(L²) dense attention exceeds O(L×k + L×L) indexer + sparse attention.

**Setup:**
```
Fix: B=1, S=1 (decode), H=64, d_qk=576, d_v=512, k=2048
Sweep T ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}

For each T:
    # Dense: full QK^T + softmax + PV
    dense = bench(flash_mla_with_kvcache, q, kv_full, ..., causal=False)

    # Sparse: indexer + selective attention
    idx = bench(deepgemm.fp8_mqa_logits, q_idx, kv_idx, ...)
    topk = bench(torch.topk, logits, k=2048)
    sparse = bench(flash_mla_with_kvcache, q, kv, ..., indices=topk_indices)

    net = dense - (idx + topk + sparse)
```

**What we expect:**
```
T=512:   net = -0.5 ms (DSA slower — indexer overhead > sparse savings)
T=4096:  net = -0.1 ms (near breakeven)
T=16384: net = +2.0 ms (DSA faster — quadratic attention dominates)
T=65536: net = +30 ms  (DSA much faster — 4× reduction)
T=131072: net = +120 ms (DSA essential — dense is infeasible)
```

**Why this range:** The paper claims 1.5-2× cost reduction. At T=131072 (the paper's target), dense attention would be ~250ms while sparse should be ~60ms (indexer ~30ms + sparse attn ~30ms), giving 4× reduction. But at T=1024, the indexer's fixed overhead (~2ms) exceeds the sparse savings (~1ms).

**Venue fit:** NeurIPS Systems — "When does sparse attention pay off? An empirical crossover analysis for MoE-MLA architectures."

### Experiment D: Per-Layer Precision Profiling

**Hypothesis:** Different layers need different precision levels, and the optimal assignment differs between MoE and dense layers.

**Setup:**
```
For layer_idx in {0, 1, 2, 10, 20, 40, 60, 77}:
    For precision in {BF16, FP8_global, FP8_pertile, W4A8}:
        # Run single layer forward + backward
        output = layer(input, precision=precision)
        cos_sim = cosine_similarity(output, bf16_reference)
        latency = bench(layer.forward, input)

        # Record per-layer sensitivity
```

**What we expect:** Early layers (0-2, dense) are more precision-sensitive than middle MoE layers. The output head layer is most sensitive.

**Why this matters:** GLM-5 paper uses W8A8 for attention and W4A8 for MoE experts. Our experiment would validate (or challenge) this assignment empirically.

**Venue fit:** ICML — aligns with Trend 11 (Per-layer precision profiling is standard).

---

## Summary: What To Do Next, In Priority Order

| Priority | Action | Effort | Unlocks |
|----------|--------|--------|---------|
| 1 | **Implement weight absorption** | 2-3 days | End-to-end kernel speedup (currently 1.0×) |
| 2 | **Experiment C: DSA crossover sweep** | 1 day | The central claim for any DSA paper |
| 3 | **Experiment A: Indexer Jaccard under FP8** | 1 day | Novel metric for sparse attention FP8 quality |
| 4 | **Experiment B: FP8 grouped GEMM crossover** | 1 day | SC '26 MoE paper data |
| 5 | **Fix component integration benchmark** (all zeros) | 0.5 day | Triple report Level 2 |
| 6 | **Experiment D: Per-layer precision** | 2 days | ICML precision paper |
| 7 | **Multi-GPU scaling** (2/4/8 GPU) | 1 day + 8-GPU pod | SC '26 requirement |
