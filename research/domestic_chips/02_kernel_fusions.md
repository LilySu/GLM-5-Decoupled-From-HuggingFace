# GLM-5 Ascend Kernel Fusions — Technical Deep Dive

## Three Fused Operators

All three fusions target the DSA (DeepSeek Sparse Attention) component of GLM-5, which is the computational bottleneck on NPU due to the Cube-Vector split architecture.

---

## 1. Lightning Indexer (闪电索引器)

### Mathematical Definition

```
I_{t,s} = Σ_{h=1}^{H_I} w_{t,h} · ReLU(q_{t,h} · k_s)

where:
  q: [B, S_q, H_I, D_I]  — indexer queries (H_I=32, D_I=128 for GLM-5)
  k: [B, S_kv, D_I]      — indexer keys (single-head)
  w: [B, S_q, H_I]       — per-head weights
  I: [B, S_q, S_kv]      — scoring logits (dense, before TopK)
```

### Fused Operations

| Step | Operation | Hardware Unit | Data Location |
|------|-----------|--------------|---------------|
| 1 | Score = q_tile @ k_tile^T | Cube (matmul) | L0A(q) × L0B(k) → L0C(scores) |
| 2 | ReLU(Score) | Cube FixP (during L0C→L1 writeback) | L0C → L1 |
| 3 | WeightedScore = Score * w | Vector (elementwise) | UB |
| 4 | ReduceSum across heads | Vector (reduce) | UB |
| 5 | TopK selection (k=2048) | Vector (sort+merge) | UB |

### Pipeline Execution Model

```
Time →  ────────────────────────────────────────────────────────────
        │ Preload │  Loop iteration 0  │  Loop iteration 1  │ ...
        │         │                     │                     │
Cube:   │ C0      │ C1                  │ C2                  │
        │ (load   │ (q_0 @ k_0^T       │ (q_0 @ k_1^T       │
        │  q,k)   │  + ReLU via FixP)   │  + ReLU via FixP)   │
        │         │                     │                     │
Vector: │ ---     │ ---                 │ V1 (sort+merge      │
        │         │                     │  on C0's output)     │
        │         │                     │                     │
MTE:    │ Load k0 │ Load k1             │ Load k2             │
        ─────────────────────────────────────────────────────────

Key: C1 and V1 execute IN PARALLEL on different data
     "One iteration of C1 is issued BEFORE entering the main loop" (preload)
```

### Memory Layout (L1 ~320KB total)

```
Q_index:  128 KB resident (512 × 128 × FP16), loaded ONCE per query batch
K_index:  192 KB with TRIPLE buffering (3 × 256 × 128 × FP16)
          → while Cube reads buffer 0, MTE fills buffer 1, Vector processes buffer 2

L0A/L0B:  64 KB each, double-buffered (2 × 128 × 128 × FP16)
L0C:      128 KB, double-buffered (2 × 128 × 128 × FP32)
UB:       Used by Vector for sort/merge operations
```

### TopK Implementation via Sort+Merge

The TopK is implemented as a bitonic-sort cascade:

```
Step 1: VBS32 sorts groups of 32 elements (stable descending)
Step 2: VMS4v2 merges sorted groups: 32 → 128 → 512 → 2048

Total complexity: 4.36·S_kv + 3.32·k operations
For S_kv=64K, k=2048: ~286K vector operations
```

**Determinism:** VBS32 is a stable sort — elements with equal values maintain their relative order. This ensures deterministic TopK (critical per GLM-5 paper Section 3.2).

### Performance

| Metric | 128K Sequence |
|--------|-------------|
| Memory savings | 30 GB less VRAM |
| Speed vs Flash Attention | 8× faster |
| Pipeline utilization | Cube-bound (Vector hides behind Cube) |

### Open-Source Implementations

1. **AscendC** — `cann-recipes-infer/ops/ascendc/` (Huawei, closed-source core, open API)
2. **TileLang** — `lemyx/tilelang-dsa` (open-source training kernel, BF16, MIT license)
3. **DeepGEMM** — `fp8_mqa_logits` (FP8 scoring on NVIDIA, same algorithm, different HW)

---

## 2. Sparse Flash Attention (稀疏闪存注意力)

### What It Computes

After Lightning Indexer selects top-2048 positions per query, this kernel computes attention over ONLY those positions:

```
For each query position t:
  selected_indices = TopK(I[t, :], k=2048)  # from Lightning Indexer
  K_sparse = K[selected_indices]             # gather from KV cache
  V_sparse = V[selected_indices]
  O[t] = softmax(Q[t] @ K_sparse^T / √d) @ V_sparse
```

### Parallelized Execution

The key innovation is executing **retrieval and attention in parallel**:

```
Traditional (sequential):
  1. Gather K_sparse from KV cache using indices  ← memory-bound
  2. Compute QK^T, softmax, QK·V                  ← compute-bound
  Total time: T_gather + T_attention

Fused (parallel):
  While Cube computes attention on tile N:
    MTE gathers KV for tile N+1 from cache
    Vector computes softmax rescaling for tile N-1
  Total time: max(T_gather, T_attention)  ← overlapped!
```

### Pipeline Detail

```
KV Tile 0:  [MTE: gather] → [Cube: QK^T] → [Vector: softmax] → [Cube: PV]
KV Tile 1:           [MTE: gather] → [Cube: QK^T] → [Vector: softmax] → [Cube: PV]
KV Tile 2:                    [MTE: gather] → [Cube: QK^T] → ...

The gather of tile N+1 overlaps with the compute of tile N
```

### Relation to FlashAttention

This is essentially FlashAttention with sparse token selection:
- Standard FlashAttention: iterates over ALL KV tiles
- Sparse FlashAttention: iterates over only the SELECTED KV tiles (from indexer)
- Uses online softmax (same as FlashAttention) for numerical stability
- Block size matches Lightning Indexer's output granularity (TOPK_BLOCK_SIZE = 64)

---

## 3. MLAPO — Multi-head Latent Attention Pre-processing Optimization

### The 13 Fused Operators

MLAPO targets the preprocessing pipeline that runs BEFORE the main attention:

```python
# In the unfused PyTorch implementation, each line = 1 kernel launch:
q_compressed = q_a_proj(hidden)                     # Op 1:  Linear [6144→2048]
q_normed = q_a_layernorm(q_compressed)              # Op 2:  RMSNorm
q_expanded = q_b_proj(q_normed)                     # Op 3:  Linear [2048→16384]
q_heads = q_expanded.view(B,S,64,256).transpose()   # Op 4:  Reshape+Transpose
q_nope, q_pe = q_heads.split([192,64])              # Op 5:  Split
q_pe = apply_rope(q_pe, cos, sin)                   # Op 6:  RoPE (multiply+rotate)
q = cat(q_nope, q_pe)                               # Op 7:  Concatenate

kv_compressed = kv_a_proj(hidden)                    # Op 8:  Linear [6144→576]
k_compressed, k_pe = kv_compressed.split([512,64])   # Op 9:  Split
k_normed = kv_a_layernorm(k_compressed)              # Op 10: RMSNorm
kv_expanded = kv_b_proj(k_normed)                    # Op 11: Linear [512→28672]
k_nope, v = kv_expanded.view().split().transpose()   # Op 12: Reshape+Split+Transpose
k_pe = apply_rope(k_pe).expand(64)                   # Op 13: RoPE+Expand
```

### Why Fusion Matters

Each unfused operator:
1. Launches a separate NPU kernel (~5-10μs overhead each)
2. Reads from GM, computes, writes back to GM
3. The intermediate tensors (q_compressed, q_normed, etc.) exist only to be consumed by the next op

**13 kernel launches × 78 layers = 1,014 launches per forward pass** just for preprocessing.

With MLAPO fusion: **78 launches** (1 per layer).

### Fusion Strategy: VV Fusion + Cube-Vector Pipeline

MLAPO uses **VV fusion (Vector-Vector fusion)** technology:

```
Within the single fused MLAPO operator:

Cube unit handles:
  - Op 1: q_a_proj (Linear, matmul)
  - Op 3: q_b_proj (Linear, matmul)
  - Op 8: kv_a_proj (Linear, matmul)
  - Op 11: kv_b_proj (Linear, matmul)

Vector unit handles (in parallel with next Cube op):
  - Op 2: RMSNorm (variance, rsqrt, scale)
  - Op 4: Reshape+Transpose (data movement)
  - Op 5: Split (pointer arithmetic)
  - Op 6: RoPE (sin/cos multiply + rotate)
  - Op 7: Cat (data movement)
  - Op 9: Split (pointer arithmetic)
  - Op 10: RMSNorm
  - Op 12: Reshape+Split+Transpose
  - Op 13: RoPE+Expand

Pipeline:
  Cube(Op1) → Vector(Op2) in parallel with Cube(Op3) → Vector(Op4-7) → ...
```

### Data Flow (Fused)

```
GM → L1: load hidden_states ONCE
L1 → L0A: hidden for q_a_proj matmul
Cube: q_a_proj
L0C → L1: q_compressed
L1 → UB: for RMSNorm (Vector)
Vector: RMSNorm + reshape (while Cube idle or preloading)
UB → L1: q_normed
L1 → L0A: q_normed for q_b_proj matmul
Cube: q_b_proj
... (continues, all within one kernel launch, no GM round-trips for intermediates)
```

### Memory Savings

Without fusion: each intermediate tensor allocated in GM
- q_compressed: B×S×2048×2 = 32KB per token
- q_expanded: B×S×16384×2 = 256KB per token
- kv_compressed: B×S×576×2 = 9KB per token
- ... total: ~400KB per token of temporary GM allocations

With fusion: all intermediates live in L1/UB (~320KB total, shared)
- At 128K tokens: saves ~50GB of GM allocations

---

## Summary: How the Three Fusions Work Together

```
Input: hidden_states [B, S, 6144]

Step 1: MLAPO (fused preprocessing)
  → Q [B, 64, S, 256], K [B, 64, S, 256], V [B, 64, S, 256]
  → Also produces: indexer_q [B, S, 32, 128], indexer_k [B, S, 128]
  [1 kernel launch instead of 13]

Step 2: Lightning Indexer (fused scoring + selection)
  → indexer_q × indexer_k → scores → ReLU → weighted sum → TopK
  → selected_indices [B, S, 2048]
  [1 kernel launch, Cube-Vector pipelined]

Step 3: Sparse Flash Attention (fused gather + attention)
  → Gather K,V at selected_indices → QK^T → softmax → QK·V
  → attention_output [B, S, 64, 256]
  [1 kernel launch, MTE-Cube-Vector pipelined]

Total: 3 kernel launches per layer (was ~20+ unfused)
```
