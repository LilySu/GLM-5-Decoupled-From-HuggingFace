# Academic Conference Research: LLM Inference Benchmarking Standards

## Purpose

This document synthesizes evaluation methodology, metrics, and best practices from top systems and ML conferences (NeurIPS, MLSys, OSDI, SOSP, ACL, SC) for the period 2024-2026. It informs how the GLM-5 FlashMLA vs FlashInfer benchmark should be designed to meet publication-quality standards.

---

## 1. Dominant Trends (2024-2026) — What the Field Prioritizes

### 1.1 From "Throughput" to "Goodput" and SLA-Awareness

**The shift:** Raw throughput (tokens/s) is no longer sufficient. The field has moved toward **SLA-constrained metrics** that reflect actual user experience.

**Key paper:** "Revisiting Service Level Objectives and System Level Metrics in Large Language Model Serving" (arXiv:2410.14257, OpenReview 2025)
- Introduces **smooth goodput** — a unified metric combining throughput AND user experience
- Identifies flaws in existing metrics: (1) manually delaying tokens can "improve" SLOs, (2) abandoning slow requests "improves" system metrics — both are counterintuitive
- Proposes a framework that penalizes these gaming strategies

**MLPerf Inference v5.1 (Sep 2025)** — The industry benchmark standard now uses:
- **p99 TTFT threshold: 2 seconds** (time to first token)
- **p99 TPOT threshold: 80 ms** (time per output token)
- Throughput expressed as **tokens/second** with variable-length inputs/outputs
- DeepSeek-R1 671B added as the first "reasoning model" benchmark with 20K max output length

**Implication for our benchmark:** Report smooth goodput and SLA attainment rate, not just raw latency. Use MLPerf's p99 thresholds (TTFT < 2s, TPOT < 80ms) as reference SLOs.

Sources:
- [Revisiting SLOs in LLM Serving](https://arxiv.org/abs/2410.14257)
- [MLPerf Inference v5.1 DeepSeek](https://mlcommons.org/2025/09/deepseek-inference-5-1/)
- [MLPerf Inference v5.0](https://mlcommons.org/2025/04/llm-inference-v5/)

### 1.2 Roofline Analysis is Now Expected

**The standard:** Every kernel paper must show where it sits on the roofline — is it memory-bound or compute-bound, and how close to the hardware ceiling?

**Key paper:** "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363, 2024)
- Defines a **roofline framework** specifically for LLM inference
- Key insight: "why LLMs are memory-bound, how much memory and computation they need, and how to choose the right hardware"
- Categorizes optimizations by impact on memory access vs computation
- Used as a predictive model — "Predicting LLM Inference Latency: A Roofline-Driven ML Method" (NeurIPS 2024 ML for Systems Workshop)

**FlashAttention-3 (2024):** Benchmark gold standard for attention kernels
- Reports **75% MFU** (Model FLOPs Utilization) on H100 FP16 = 740 TFLOPS
- **1.2 PFLOPS** with FP8, 2.6x lower error than baseline FP8
- Explicitly measures warp specialization overhead
- Sweeps: sequence length, head dimension, causal vs non-causal

**Implication:** Our benchmark MUST include a roofline plot showing both implementations' operational intensity and achieved throughput relative to H100's compute ceiling (989 TFLOPS BF16) and memory ceiling (3.35 TB/s).

Sources:
- [LLM Inference Roofline Survey](https://arxiv.org/abs/2402.16363)
- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 blog by Tri Dao](https://tridao.me/blog/2024/flash3/)

### 1.3 MoE Evaluation Must Include Hyperparameter Sweeps

**Key paper:** "MoE-Inference-Bench" (SC '25 Workshops, arXiv:2508.17467)
- **The definitive MoE inference benchmark** — evaluates on NVIDIA H100 GPUs
- Sweeps: **input/output lengths {128, 256, 512, 1024, 2048}, batch sizes {1, 16, 32, 64}**
- Sweeps MoE-specific: **FFN dimension, total expert count, active expert ratio**
- Tests optimizations: pruning, Fused MoE, speculative decoding, quantization, parallelization
- Uses **4× H100 SXM5 80GB** with **vLLM** framework
- Reports: throughput, OOM boundaries, per-technique speedup

**Implication:** Our MoE GEMM benchmark should sweep batch size AND token count, not just one fixed configuration. Match MoE-Inference-Bench's sweep ranges.

Sources:
- [MoE-Inference-Bench (SC'25)](https://dl.acm.org/doi/10.1145/3731599.3767706)
- [MoE-Inference-Bench (arXiv)](https://arxiv.org/abs/2508.17467)

### 1.4 Sparse Attention Needs Losslessness Proof + Long-Context Scaling

**Key paper:** "Native Sparse Attention (NSA)" — **ACL 2025 Best Paper Award** (DeepSeek + Peking University)
- Three parallel branches: compressed attention, selected attention, sliding window
- **Hardware-aligned design** — arithmetic-intensity-balanced for modern GPUs
- Reports: performance "comparable to or better than Full Attention" while accelerating decoding, forward, and backward
- Evaluation includes **knowledge distillation + AIME benchmark** (mathematical reasoning)
- Acceleration ratio increases with sequence length — must show this scaling

**GLM-5 paper (Section 2.2, Table 5):** Compared DSA against 4 alternatives across 6 context lengths (4K→128K)
- This comparison IS the evaluation standard for sparse attention in 2025-2026
- Key: DSA is "lossless by construction" — this is the compelling claim

**NeurIPS 2025:** "Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning"
- Dynamic budget allocation (not fixed top-k) — shows the field is moving toward adaptive sparsity

**Implication:** Our DSA benchmark should test at multiple context lengths (256→200K) and report the attention pattern quality (Jaccard similarity of selected tokens) alongside latency.

Sources:
- [NSA: Native Sparse Attention (ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)
- [Twilight: Adaptive Attention Sparsity (NeurIPS 2025)](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf)

### 1.5 Speculative Decoding: Accept Length is the Key Metric

**Field consensus (2024-2025):** For MTP/speculative decoding papers, the metrics are:
- **Accept length** (mean accepted tokens per verification step)
- **Draft overhead ratio** (draft model cost / target model cost)
- **Effective speedup** = accept_length / (1 + K × draft_overhead)
- Wall-clock tokens/second (end-to-end, including overhead)

**GLM-5 Table 2:** Accept length 2.76 (GLM-5) vs 2.55 (DeepSeek V3.2)
**MLPerf v5.1:** DeepSeek-R1 with 20K max output — tests long-chain reasoning where speculation matters most

**Implication:** MTP benchmark should report accept length, draft overhead, and effective speedup — not just raw decode latency.

### 1.6 FP8 Evaluation Must Report Quality AND Speed

**FlashAttention-3 (2024):** Established the standard for FP8 attention evaluation:
- Report **TFLOPS** (speed metric) AND **RMSE** (quality metric) for FP8
- FA3's FP8 has "2.6x lower numerical error than baseline FP8 attention"
- Show the Pareto frontier: speed vs quality at different precision levels

**QServe (NeurIPS 2024 area):** W4A8KV4 quantization benchmark
- Reports throughput on A100 AND L40S (multiple GPUs)
- Compares against TensorRT-LLM baseline
- Reports economic metric: "3x cost reduction"

**Implication:** Our FP8 benchmark should report cosine similarity (quality) alongside TFLOPS (speed) at each precision configuration. Show the speed-quality Pareto frontier.

Sources:
- [FlashAttention-3 FP8 evaluation](https://tridao.me/publications/flash3/flash3.pdf)
- [QServe](https://arxiv.org/abs/2405.04532)

---

## 2. Standard Metrics and Terminology (2025 Consensus)

### 2.1 Kernel-Level Metrics (NeurIPS/MLSys Standard)

| Metric | Definition | Gold Standard Reference |
|--------|-----------|----------------------|
| **MFU** (Model FLOPs Utilization) | Achieved TFLOPS / peak TFLOPS × 100 | FlashAttention-2: 50-73% (A100), FA3: 75% (H100) |
| **HBM SOL%** (Speed of Light) | Achieved bandwidth / peak bandwidth × 100 | H100: 3.35 TB/s peak |
| **Operational Intensity** | FLOPs / bytes accessed | Roofline analysis (Williams et al.) |
| **TFLOPS** | Tera floating-point ops per second | FA3: 740 TFLOPS FP16, 1.2 PFLOPS FP8 |
| **Kernel Latency** | CUDA event timing, report p50/p95/p99 | All kernel papers |
| **Memory Footprint** | Peak GPU memory allocated | PagedAttention (SOSP '23) |

### 2.2 Serving-Level Metrics (OSDI/SOSP/MLPerf Standard)

| Metric | Definition | Threshold (MLPerf v5.1) |
|--------|-----------|------------------------|
| **TTFT** (Time to First Token) | Prefill completion latency | p99 < 2 seconds |
| **TPOT** (Time Per Output Token) | Inter-token latency (decode) | p99 < 80 ms |
| **TBT** (Time Between Tokens) | Same as TPOT (alternative name) | p99 < 80 ms |
| **Goodput** | Requests completed under SLA / total time | Sarathi-Serve (OSDI '24) |
| **Smooth Goodput** | Goodput weighted by user experience quality | Revisiting SLOs (2025) |
| **Throughput** (tokens/s) | Total tokens generated per second | MLPerf standard unit |
| **SLA Attainment** | % of requests meeting latency SLOs | SOLA (MLSys 2025) |

### 2.3 Quality Metrics for Approximate/Quantized Methods

| Metric | When to Use | Threshold |
|--------|-----------|-----------|
| **Perplexity** | End-to-end model quality | <0.5 ppl degradation acceptable |
| **Cosine Similarity** | Per-layer output comparison | >0.99 per layer, >0.90 cumulative |
| **RMSE** | FP8 attention output error | FA3: 2.6x lower than baseline |
| **Jaccard Similarity** | Sparse token selection overlap | >0.95 for top-k agreement |
| **Accept Length** | Speculative decoding quality | GLM-5: 2.76 tokens/step |
| **Greedy Token Match** | Deterministic decode comparison | Must be bit-identical |

---

## 3. What Parameter Sweeps Are Standard

### 3.1 Attention Kernel Benchmarks (FA3 / FlashInfer / FlashMLA pattern)

| Parameter | Values | Why |
|-----------|--------|-----|
| Sequence length | {128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K} | Context scaling behavior |
| Head dimension | {64, 128, 192, 256} | Architecture-dependent (GLM-5: QK=256, V=256) |
| Causal vs non-causal | Both | Different computational profiles |
| FP16 / BF16 / FP8 | All available | Precision-performance tradeoff |
| Batch size | {1, 4, 16, 32, 64, 128} | Throughput scaling |

### 3.2 MoE Benchmarks (MoE-Inference-Bench pattern)

| Parameter | Values | Why |
|-----------|--------|-----|
| Input/output tokens | {128, 256, 512, 1024, 2048} | Variable workload size |
| Batch size | {1, 16, 32, 64} | Throughput scaling |
| Expert count | {8, 16, 64, 128, 256} | Scaling behavior (GLM-5: 256) |
| Active experts | {2, 4, 8} | Routing density (GLM-5: top-8) |
| FFN dimension | Model-specific | Memory-compute tradeoff |

### 3.3 Serving System Benchmarks (OSDI/SOSP pattern)

| Parameter | Values | Why |
|-----------|--------|-----|
| Request rate | {0.5, 1, 2, 4, 8, 16} req/s | Load scaling |
| Prompt length | Lognormal distribution, μ=512 | Realistic workload |
| Output length | Exponential, μ=256 | Realistic workload |
| Batch concurrency | {1, 4, 16, 32, 64, 128} | GPU utilization |
| Dataset | ShareGPT traces / synthetic | Real vs controlled |

---

## 4. What Makes a Paper Compelling in This Field (2024-2025)

### 4.1 The "Triple Report" Standard

Top papers (FA3, Sarathi-Serve, DistServe, MegaBlocks) all report THREE levels:

1. **Microbenchmark** — Kernel-level TFLOPS, latency, bandwidth utilization
2. **Component integration** — How the kernel performs within a full model layer
3. **End-to-end** — Real task speedup (training time, inference throughput, generation quality)

Papers that report only microbenchmarks are considered incomplete. Papers that report only end-to-end without explaining WHY (which component improved) are considered lacking analysis.

### 4.2 The "Control Experiment" Requirement

Academic reviewers expect **ablation studies** that isolate each contribution:
- "What if we only changed the attention kernel but kept everything else the same?"
- "What if we only changed the KV cache format?"
- "What is the overhead of the FP8 quantization boundary?"

Our benchmark design with DSA indexer and MoE GEMM as **controls** (same kernel in both implementations) directly addresses this.

### 4.3 The "Reproducibility" Expectation

Since NeurIPS 2024, reproducibility checklists are mandatory. Systems papers must report:
- Exact hardware (GPU model, SM count, memory, interconnect)
- Exact software (CUDA version, framework version, kernel library commit)
- Environmental controls (GPU clock lock, power limit, thermal state)
- Random seeds and determinism settings
- Raw data availability (all measurements, not just aggregates)

### 4.4 The "Statistical Rigor" Bar

Single-run measurements are no longer acceptable at top venues:
- **Minimum:** 3 replicate runs with variance reported
- **Preferred:** 100+ iterations with bootstrap confidence intervals
- **Required:** p99 tail latency (not just mean/median)
- **Expected:** Statistical significance test (Mann-Whitney U for non-normal distributions)
- **Encouraged:** Effect size reporting alongside p-values

### 4.5 The "Hardware-Aware" Framing

Papers must acknowledge hardware-specific optimizations:
- H100: TMA (Tensor Memory Accelerator), warp specialization, 4th-gen tensor cores
- Roofline analysis must use the CORRECT hardware specs
- Different conclusions on different hardware is expected and valued (e.g., "FlashMLA wins at B=1, FlashInfer wins at B≥64")

---

## 5. Conference-Specific Expectations

### NeurIPS (Machine Learning — broad audience)
- **Language:** "efficiency," "scalability," "practical impact"
- **Values:** Novel algorithmic insight + empirical validation
- **Expected:** Theoretical motivation (IO-aware algorithm design, complexity analysis)
- **Standard:** Datasets & Benchmarks track has strict reproducibility requirements
- **Recent trend (2025):** Reasoning model evaluation (AIME, math), agentic evaluation

### MLSys (Systems for ML — systems audience)
- **Language:** "throughput," "latency," "utilization," "hardware efficiency"
- **Values:** Practical deployment impact, hardware-aware design
- **Expected:** Comparison against production baselines (vLLM, TRT-LLM)
- **Standard:** Performance numbers must be on real hardware (no simulation-only)
- **Recent trend (2025):** Disaggregated serving, MoE optimization, KV cache management

### OSDI/SOSP (Operating Systems — systems audience)
- **Language:** "SLO attainment," "tail latency," "goodput," "resource efficiency"
- **Values:** Fundamental systems insight (paging, scheduling, fault tolerance)
- **Expected:** Real workload traces, production-scale evaluation
- **Standard:** The most rigorous statistical requirements
- **Recent trend (2024):** PD disaggregation, request migration, SLA-aware scheduling

---

## 6. Emerging Trends in Low-Level Systems Benchmarking (2025-2026)

### 6.1 Unified FP8 Pipelines — Eliminating Train-Inference Mismatch

**Key paper:** "Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL" (LMSYS, Nov 2025)
- Using unified FP8 for both training and inference eliminates the quantization-error mismatch between train and deploy
- Specific finding: SwiGLU amplifies outliers, destabilizing FP8 training → solved by per-channel "Smooth-SwiGLU" quantization
- **Implication:** Our benchmark should track whether the FP8 format used during inference matches what the model was trained with

**Key paper:** "InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models" (arXiv:2509.22536)
- FP8 training successfully scales to 2 trillion tokens (up from 100B limit)
- Critical: per-channel quantization and careful placement of FP32 retention zones

**Implication for GLM-5:** The precision tracking document (`PRECISION_TRACKING.md`) maps all 4 implementations' precision at every stage. Benchmarks should verify that FP8 kernel outputs match the precision profile the model was trained under.

Sources:
- [Unified FP8 for MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [InfiR2: FP8 Training Recipe](https://arxiv.org/abs/2509.22536)

### 6.2 Microscaling (MX) Formats — The Next Precision Frontier

**OCP Microscaling Formats (MX Specification):** Vendor-neutral standard for reduced-precision formats below FP8:
- **MXFP4** (4-bit): 2 exponent + 1 mantissa + shared 8-bit E8M0 block exponent
- **MXFP6** (6-bit): More precision than FP4, less than FP8
- **MXFP8** (8-bit): Standardized FP8 with shared scaling

**Key paper:** "MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for LLMs" (arXiv:2508.02343)
- Per-channel mixed-precision: dynamically selects MXFP4, MXFP6, or MXFP8 per channel based on quantization error threshold
- Tailored for Blackwell architecture's native MXFP4 support
- **Blackwell achieves 3958 TFLOPS with MXFP4** (2× over H100 FP8)

**Key paper:** "Is Finer Better? The Limits of Microscaling Formats in Large Language Models" (arXiv:2601.19026)
- Finds that MXFP4 requires careful calibration — not all layers benefit equally from lower precision
- Per-layer precision allocation is essential (some layers need FP8, some can tolerate FP4)

**Implication:** Future GLM-5 optimizations on Blackwell GPUs will need per-layer precision profiling. Our precision tracking framework should support this.

Sources:
- [OCP MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [MicroMix](https://arxiv.org/abs/2508.02343)
- [Limits of Microscaling](https://arxiv.org/abs/2601.19026)
- [MXFP4 on AMD GPUs (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### 6.3 Operator Fusion and Graph Compilation as Benchmarking Dimension

**EuroSys 2025 trend:** Systems papers now evaluate the impact of operator fusion separately from individual kernel performance.

**Key approach:** SKIP (System-Aware Kernel Inference Profiler)
- Builds operator-kernel dependency graphs from profiling traces
- Proposes metrics based on fine-grained operator-kernel offload patterns
- Separates "kernel compute time" from "fusion overhead" and "scheduling gap"

**torch.compile + Inductor:** vLLM V1 uses torch.compile by default
- Inductor fuses pointwise ops into matmuls and autotunes backends (cuBLAS, CUTLASS, Triton)
- CUDA Graphs combine multiple kernel launches into replayable sequences
- **New benchmark dimension:** fused vs unfused execution, measuring fusion benefit

**Implication:** Our benchmark should report not just kernel time but also:
- Number of kernel launches per forward pass (fewer = better fusion)
- Gap time between launches (Python dispatch overhead)
- CUDA graph speedup vs eager mode

Sources:
- [Characterizing LLM Inference](https://arxiv.org/abs/2504.11750)
- [Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/abs/2512.01644)
- [vLLM with torch.compile](https://developers.redhat.com/articles/2025/09/03/vllm-torchcompile-efficient-llm-inference-pytorch)

### 6.4 Prefill-Decode Crossover Analysis

**Key finding (2025):** "A clear crossover point exists in latency breakdown between Prefill and Decode phases" — at short contexts, FFN dominates; at long contexts, Attention becomes the bottleneck.

**Multi-stage pipeline characterization:**
- Prefill: compute-bound → MFU is the right metric
- Decode: memory-bound → HBM SOL% is the right metric
- The crossover point depends on batch size, context length, and model architecture
- For MoE models specifically: the expert dispatch pattern changes between prefill (many tokens, contiguous) and decode (few tokens, scattered)

**Implication:** Our benchmark must separately characterize prefill and decode phases AND identify the crossover point where the bottleneck shifts.

Sources:
- [Mind the Memory Gap: GPU Bottlenecks in Large-Batch Inference](https://upcommons.upc.edu/bitstreams/82e2be60-b600-4fa1-90ff-08d66f1cac7a/download)
- [Understanding Multi-Stage AI Inference Pipelines](https://people.csail.mit.edu/suvinay/pubs/2025.hermes.arxiv.pdf)

### 6.5 Per-Layer Precision Profiling as Standard Practice

**DeepSeek-V3 established (Dec 2024):** Certain components MUST retain high precision:
- Embedding, output head, MoE gating, normalization, attention operators → BF16/FP32
- GEMM operations → FP8 input, FP32 accumulation
- Fine-grained quantization: 1×128 tiles for activations, 128×128 blocks for weights
- Result: <0.25% accuracy loss vs full BF16

**ICML 2025:** "Optimizing Large Language Model Training Using FP4 Quantization"
- Shows that FP4 can match BF16 accuracy with per-channel scaling and outlier clamping
- But requires **differentiable quantization estimator** for gradient-aware precision allocation

**The new standard:** Papers must report precision at EVERY stage, not just "we use FP8." The community now expects:
1. Per-stage dtype map (what precision at each operation)
2. Per-stage quality metric (cosine similarity or RMSE vs full-precision reference)
3. Cumulative quality tracking (how quality degrades across layers)
4. Critical precision zones identified (which stages cannot tolerate quantization)

**Implication:** Our `PRECISION_TRACKING.md` document addresses this directly. The benchmark should validate these precision claims empirically.

Sources:
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek FP8 Training Analysis](https://dataturbo.medium.com/deepseek-technical-analysis-5-fp8-training-ff34768727b8)
- [FP4 Quantization (ICML 2025)](https://icml.cc/virtual/2025/poster/43733)
- [FP8-LM: FP8 for Gradients and Optimizer](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf)

### 6.6 NSA/DSA as a New Attention Category Requiring Specialized Evaluation

**ACL 2025 Best Paper (NSA):** Sparse attention is now recognized as a distinct category requiring specialized benchmarking:
- Must evaluate on INCREASING context lengths (acceleration ratio should grow with L)
- Must compare against MULTIPLE baselines (dense, sliding window, linear attention)
- Must prove losslessness on specific tasks (AIME, long-context QA)
- Must report hardware utilization (not just wall-clock speedup)

**NeurIPS 2025 (Twilight):** Dynamic budget allocation (top-p instead of fixed top-k)
- Shows that fixed-k sparse attention is suboptimal — some queries need more tokens than others
- GLM-5's fixed top-2048 is the current standard, but adaptive selection is the next step

**Implication:** Our DSA benchmark should test at multiple context lengths and report the acceleration ratio curve (how speedup scales with L).

Sources:
- [NSA (ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)
- [Twilight: Adaptive Sparsity (NeurIPS 2025)](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf)

### ACL (Computational Linguistics — NLP audience)
- **Language:** "model quality," "task performance," "knowledge distillation"
- **Values:** Quality preservation alongside efficiency
- **Expected:** Benchmark scores on standard NLP tasks
- **Recent trend (2025):** NSA won Best Paper — shows systems work is valued when it enables better NLP

---

## 6. Application to GLM-5 FlashMLA vs FlashInfer Benchmark

### What our benchmark MUST include (to meet 2025 standards):

1. **Roofline plot** showing both implementations' operational intensity and achieved TFLOPS relative to H100 ceilings
2. **MFU reporting** using FA3's 75% as the reference point
3. **p99 latency** (not just median) at MLPerf's thresholds (TTFT < 2s, TPOT < 80ms)
4. **Smooth goodput** under realistic serving scenarios (chatbot, code assist, long-doc QA, agentic SWE)
5. **Parametric sweeps** matching MoE-Inference-Bench: batch {1,16,32,64}, tokens {128,512,1K,2K,4K}
6. **FP8 quality + speed** Pareto frontier (cosine similarity vs TFLOPS)
7. **Control experiments** — DSA indexer and MoE GEMM must match within 2%
8. **Ablation** — kernel time vs framework overhead isolation via nsys
9. **100+ iterations** with bootstrap 95% CI and Mann-Whitney U test
10. **Reproducibility artifact** — JSON raw data, environment snapshot, exact versions

### What our benchmark SHOULD include (for top-tier impact):

11. **Context scaling to 200K** (GLM-5's maximum, with DSA sparse attention)
12. **Multi-GPU** TP evaluation (NCCL all-reduce bandwidth as separate dimension)
13. **Comparison table** formatted like MoE-Inference-Bench: rows=configurations, columns=metrics
14. **Cost analysis** — "which implementation requires fewer H100 GPU-hours for equivalent quality?"
15. **Decision matrix** — "use FlashMLA when X, use FlashInfer when Y" (actionable recommendations)

---

## 7. Key References (Ordered by Relevance to Our Work)

### Directly Applicable Methodology
1. **FlashAttention-3** (2024) — H100 kernel benchmark gold standard. MFU, FP8 quality, warp specialization.
2. **MoE-Inference-Bench** (SC '25) — MoE sweep methodology on H100 with vLLM.
3. **Revisiting SLOs** (2025) — Smooth goodput framework for serving evaluation.
4. **MLPerf Inference v5.1** (Sep 2025) — Industry-standard thresholds: p99 TTFT<2s, p99 TPOT<80ms.
5. **NSA** (ACL 2025 Best Paper) — Sparse attention evaluation standard.

### Foundational Papers (Must-Cite)
6. **FlashAttention-2** (NeurIPS 2023) — Established MFU as the kernel metric.
7. **vLLM/PagedAttention** (SOSP 2023) — Established serving metrics (throughput at same latency).
8. **Sarathi-Serve** (OSDI 2024) — Introduced goodput and PD disaggregation evaluation.
9. **DistServe** (OSDI 2024) — TTFT/TPOT disaggregated evaluation.
10. **MegaBlocks** (MLSys 2023) — MoE kernel evaluation (end-to-end, not just GEMM).

### GLM-5 Specific
11. **GLM-5 paper** (arXiv:2602.15763) — Table 1 (MLA), Table 3 (DSA context scaling), Table 5 (sparse attention comparison).
12. **DeepSeek-V3** (arXiv:2412.19437) — MLA, MoE, MTP architecture that GLM-5 adopts.
13. **DeepSeek-V2** (arXiv:2405.04434) — Original MLA paper (93.3% KV cache reduction, 5.76× throughput).

Sources:
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [MoE-Inference-Bench](https://arxiv.org/abs/2508.17467)
- [Revisiting SLOs in LLM Serving](https://arxiv.org/abs/2410.14257)
- [MLPerf Inference v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- [NSA (ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)
- [LLM Inference Roofline Survey](https://arxiv.org/abs/2402.16363)
- [FlashInfer v0.2](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)
- [FlashMLA-ETAP](https://arxiv.org/abs/2506.01969)
- [Twilight: Adaptive Sparse Attention (NeurIPS 2025)](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf)
- [SOLA: SLO Attainment (MLSys 2025)](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/ce55a1f2-ff0d-45f6-8985-f4f251b2a0d4.pdf)
- [Predicting LLM Latency via Roofline (NeurIPS 2024 Workshop)](https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf)
- [Databricks LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
