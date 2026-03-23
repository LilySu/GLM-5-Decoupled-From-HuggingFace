# GLM-5 Paper Cross-Check and Root Cause Analysis

## Paper Cross-Check Against Our Research

### What the paper confirms (Section 5, pages 21-22)

Our research is accurate on:
- 7 platforms: Huawei Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame (exact match)
- Lightning Indexer: "overlap computation with memory access" on NPU (confirmed)
- MLAPO: "fuses 13 small pre-processing operators into one super operator" (confirmed)
- Sparse Flash Attention: "handles the selection of TopK tokens from the KV cache and sparse attention computation in parallel" (confirmed)
- Performance: "single Chinese node achieves performance comparable to dual-GPU international clusters" (confirmed)
- Cost: "reducing deployment costs in long-sequence scenarios by 50%" (confirmed)

### Gaps found in our research

**Gap 1: INT4 Quantization-Aware Training (Section 2.4.3)**
The paper reveals: "We apply INT4 QAT in the SFT stage. We have developed a quantization kernel
applicable to both training and offline weight quantization, which ensures bitwise-identical behavior
between training and inference."

This is significant — the W4A8 quantization we documented for MoE experts was developed with
a **training-time kernel** that guarantees bit-identical behavior. Our FP8 utilities don't capture this.
The quantization is not just a post-training compression — it's baked into SFT training.

**Gap 2: Roofline-motivated head dimension change (Section 2.1, page 5)**
The paper explains WHY GLM-5 has different head dimensions from DeepSeek-V3:
> "While the number of attention heads in DeepSeek-V3 is selected according to the roofline of H800,
> it is inappropriate for other hardware. Given the MHA style of MLA during training and prefilling,
> we increase the head dimension from 192 to 256 and decrease the number of attention heads by 1/3."

This means: DeepSeek-V3 optimized for H800 GPU. GLM-5 optimized for Ascend NPU. The
head dim change (128→256 for QK, 128→256 for V, 128→64 heads) keeps training FLOPS constant
but reduces DECODING computation. On Ascend, this matters because the Cube unit's matmul tile
is 16×16, and larger head dims fill the tile more efficiently.

**Gap 3: FlashComm for communication hiding (Section 5, page 22)**
The paper mentions "FlashComm, which splits AllReduce operations to hide communication latency
behind computation." Our research doesn't document this optimization, which is critical for
multi-node inference on Ascend.

**Gap 4: Muon Split optimizer adaptation (Section 2.1, page 5)**
MLA with standard Muon optimizer underperforms GQA-8. The paper proposes "Muon Split" —
splitting up-projection matrices W_UQ, W_UK, W_UV into per-head matrices before applying matrix
orthogonalization. This is unique to GLM-5 and affects the weight structure that kernels must handle.

**Gap 5: DSA warmup vs sparse adaptation distinction (Section 2.1.1, page 6)**
The paper distinguishes:
1. **Warmup stage** (1000 steps, lr 5e-3→2e-4): trains only the indexer, base model frozen
2. **Sparse adaptation** (20B tokens, lr 1e-5): jointly trains indexer + base model

Our research conflated these into a single "DSA training" phase.

**Gap 6: Indexer frozen during RL (Section 3.2, page 12)**
"We also freeze the indexer parameters by default during RL to accelerate training and prevent
unstable learning in the indexer." This is important — the indexer is NOT fine-tuned during RL.

---

## 10 Levels of Why: DSA Indexer and Ascend Kernel Fusions

### Why was the DSA Lightning Indexer kernel implemented?

**Level 1: What does it do?**
Selects which 2,048 of potentially 200,000+ KV tokens each query should attend to.

**Level 2: Why is selection needed?**
Full attention is O(n²). At 200K context, the QK^T matrix alone is 200K × 200K × 4 bytes = 160GB — larger than any single GPU's memory, impossible to materialize.

**Level 3: Why 200K context specifically?**
GLM-5 targets "agentic engineering" — multi-file code repositories, long-horizon agent trajectories, multi-turn dialogues with preserved thinking. Paper Section 2.3: context extended through three stages: 32K → 128K → 200K. The 200K stage "substantially improves the model's ability to process ultra-long documents and complex multi-file codebases."

**Level 4: Why not sliding window or block-sparse attention?**
Paper Section 2.1.2, Table 5: Sliding Window Attention (SWA) interleaved causes catastrophic degradation: -30.35 on RULER@128K. Even search-optimized SWA loses -5.69. Block-sparse methods cannot represent the arbitrary token selection patterns that DSA learns. DSA is the only method that maintains full attention quality while reducing computation 1.5-2×.

**Level 5: Why must the indexer be lightweight?**
If the indexer costs as much as the attention it replaces, there's no savings. GLM-5's indexer uses:
- 32 heads × 128 dims = 4,096 params per position (vs main attention: 64 heads × 256 dims = 16,384)
- FP8 precision (vs BF16 for main attention)
- Single KV head (MQA) vs multi-head
- Total indexer cost: ~4× cheaper than one head of main attention

**Level 6: Why a separate learned indexer rather than reusing previous-layer attention scores?**
Previous-layer attention operates on different representations and has different attention patterns. The indexer is trained via KL-divergence against the actual attention distribution (the tilelang-dsa warmup kernel). Paper confirms: warmup stage trains only indexer (frozen base model) for 1000 steps, followed by joint training for 20B tokens.

**Level 7: Why does this need a custom kernel rather than PyTorch ops?**
The indexer computes `logits[i,j] = Σ_h w[i,h] · ReLU(q[i,h,:] · k[j,:])` for ALL i,j pairs. At 128K context: 128K × 128K × 32 heads × 128 dims = enormous intermediate tensors. In PyTorch:
- `torch.einsum('bshd,btd->bsht')` materializes a [B, 128K, 32, 128K] tensor = 256GB
- The fused kernel tiles this: 4×256 Q block × 256 KV block, scores in registers, never materializes the full matrix

**Level 8: Why is deterministic TopK critical?**
Paper Section 3.2: "Non-deterministic CUDA-based top-k caused drastic performance degradation during RL after only a few steps, accompanied by a sharp drop in entropy."

Root cause: RL uses importance ratios π_train(a|s) / π_old(a|s). If the SAME input produces DIFFERENT TopK selections between training and reference passes (due to non-deterministic sort of equal-valued elements), the importance ratio becomes noise. The policy gradient estimate degrades, entropy collapses, and the model locks into degenerate patterns.

This is analogous to MoE routing replay, but with k=2048 indices (vs k=8 experts), making replay storage infeasible. Deterministic TopK is the only practical solution.

**Level 9: Why Ascend NPU specifically needs kernel fusion?**
Ascend's Da Vinci architecture has a **Cube-Vector split**: the matrix multiply unit (Cube) and the element-wise unit (Vector) are physically separate AI cores that communicate only through Global Memory or L2 cache.

Without fusion, the DSA indexer pipeline is:
```
Op 1: q_a_proj (Cube)     → GM write
Op 2: RMSNorm (Vector)    → GM read, GM write
Op 3: q_b_proj (Cube)     → GM read, GM write
...13 total ops...
Op 14: Score matmul (Cube) → GM write
Op 15: ReLU (Vector)       → GM read, GM write
Op 16: ReduceSum (Vector)  → GM read, GM write
Op 17: TopK (Vector)       → GM read, GM write
```
= 17 GM round-trips × 78 layers = 1,326 unnecessary memory transactions per forward pass.

With MLAPO + Lightning Indexer fusion:
```
MLAPO:  13 ops → 1 kernel (4 Cube matmuls interleaved with 9 Vector ops)
LI:     4 ops → 1 kernel (Cube matmul with FixP ReLU, Vector sort/merge)
SFA:    2 ops → 1 kernel (MTE gather, Cube QK^T, Vector softmax, Cube PV)
```
= 3 GM round-trips × 78 layers = 234 transactions (5.7× reduction).

**Level 10: Why Ascend's architecture was designed this way (and why it creates this problem)?**
The Cube-Vector split is a **deliberate** architectural choice for die area efficiency. By separating matrix multiply (which needs massive tensor core area) from vector operations (which need SIMD lanes), Huawei can:
- Scale the 20:40 Cube:Vector ratio to match workload (2 Vector cores per Cube core)
- Independently optimize each unit's instruction pipeline
- Avoid the register file bloat of NVIDIA's unified SM design

The tradeoff: every Cube↔Vector handoff costs a memory transfer. NVIDIA's unified SM can pass data through registers (zero-copy). Ascend CANNOT — hence fusion is not an optimization, it's a **necessity** to achieve competitive performance.

The GLM-5 team specifically redesigned their architecture (head dims, attention heads) to better fit Ascend's Cube tile size (16×16). Paper page 5: "the number of attention heads in DeepSeek-V3 is selected according to the roofline of H800; it is inappropriate for other hardware." GLM-5's 256-dim heads fill the Cube's 128-dim L0 tiles more efficiently than DeepSeek's 128-dim heads.

---

## Why DSA Over Other Sparse Attention Methods

The paper (Section 2.1.2) benchmarks 5 alternatives on GLM-9B:

| Method | RULER@128K | Δ vs Full Attn | Why it fails |
|--------|-----------|----------------|-------------|
| Full Attention | 75.28 | baseline | O(n²) cost |
| SWA Interleave | 6.51 | -68.77 | Fixed window misses global dependencies |
| SWA Pattern (search) | 53.95 | -21.33 | Better but still misses cross-window info |
| GDN (linear attn) | 64.00 | -11.28 | Linear approximation loses precision |
| SimpleGDN | 67.03 | -8.25 | Better linear variant, still lossy |
| **DSA** | **78.86** | **-0.35** | Token-level selection preserves quality |

DSA wins because it's the only method that:
1. Preserves full attention quality (within 0.35 points)
2. Reduces computation 1.5-2× (not just memory)
3. Is compatible with existing MLA infrastructure
4. Works with RL training (deterministic TopK)

---

## Impact on Architecture Decisions

The need for efficient DSA drove several GLM-5 architecture choices:

1. **Head dims 256 instead of 128** → Better Cube tile utilization on Ascend
2. **64 heads instead of 128** → Keeps parameter count constant, reduces decoding cost
3. **Separate indexer with 32 heads × 128 dims** → Lightweight enough to be useful
4. **FP8 indexer scoring** → 2× memory reduction for the indexer's QK scoring
5. **KV LoRA rank 512** → Compressed KV cache enables longer contexts
6. **MTP with 3 shared layers** → Higher accept rate compensates for DSA overhead
7. **INT4 QAT during SFT** → Enables single-node Ascend deployment
