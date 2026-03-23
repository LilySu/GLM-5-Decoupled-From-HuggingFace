# Kernel Pseudocode and Implementation Requirements

Based on the Ascend architecture specs and open-source TileLang implementation, here is
pseudocode for each of the three GLM-5 fused kernels, with per-platform adaptation notes.

---

## 1. Lightning Indexer Kernel

### Pseudocode (Ascend NPU)

```
KERNEL lightning_indexer(
    Q: [S_q, H_I, D_I],     // indexer queries, FP16 or FP8
    K: [S_kv, D_I],          // indexer keys, FP16 or FP8
    W: [S_q, H_I],           // per-head weights, FP32
    OUT: [S_q, TopK],        // output indices, INT32
    cu_seqlens_q: [B],       // variable-length support
    cu_seqlens_k: [B]
):
    // Grid: (ceil(S_q / BLOCK_Q), H_I, B)
    // BLOCK_Q = 512 (processes 8 tokens when H_I=64, or 4 when H_I=32)
    // BLOCK_KV = 512 (KV tile size)

    // === L1 Allocation (~320KB) ===
    q_l1[BLOCK_Q, D_I]          = 128 KB  // Q resident (loaded once)
    k_l1[3][BLOCK_KV/2, D_I]    = 192 KB  // K triple-buffered

    // === L0 Allocation ===
    q_l0a[2][128, D_I]    // Cube input A, double-buffered
    k_l0b[2][128, D_I]    // Cube input B, double-buffered
    s_l0c[2][128, 128]    // Cube output, double-buffered (FP32 accum)

    // === UB Allocation (Vector) ===
    scores_ub[BLOCK_Q, BLOCK_KV]  // after ReLU + weight multiply
    topk_buffer[BLOCK_Q, TopK]    // running top-k state

    // Step 0: Load Q (resident, loaded once per query block)
    MTE.copy(GM[Q_block] → q_l1)

    // Step 1: Preload first K tile into L0B
    MTE.copy(GM[K_tile_0] → k_l1[0])
    MTE.copy(k_l1[0] → k_l0b[0])

    // Step 2: Main loop over KV tiles
    for kv_tile in range(num_kv_tiles):

        // --- Cube: Score matmul + ReLU via FixP ---
        // This runs on Cube core with L0A(q) × L0B(k) → L0C(scores)
        // FixP applies ReLU during L0C → L1 writeback
        CUBE.matmul(q_l0a, k_l0b[kv_tile % 2], s_l0c[kv_tile % 2])
        CUBE.writeback_with_relu(s_l0c → scores_l1)  // ReLU fused

        // --- MTE: Prefetch next K tile (overlaps with Cube) ---
        if kv_tile + 1 < num_kv_tiles:
            MTE.copy(GM[K_tile_{kv_tile+1}] → k_l1[(kv_tile+1) % 3])

        // --- Vector: Weight multiply + ReduceSum + TopK merge ---
        // This runs on Vector core IN PARALLEL with next Cube iteration
        VECTOR.load(scores_l1 → scores_ub)
        VECTOR.multiply(scores_ub, W_tile)           // weight per head
        VECTOR.reduce_sum(scores_ub, dim=heads)      // sum across H_I heads
        VECTOR.vbs32_sort(scores_ub)                 // sort groups of 32
        VECTOR.vms4v2_merge(scores_ub, topk_buffer)  // merge into running top-k

        // Synchronize before next Cube iteration
        BARRIER.cube_vector_sync()

    // Step 3: Write final top-k indices to GM
    MTE.copy(topk_buffer → GM[OUT_block])
```

### Requirements per Platform

| Platform | Matmul Unit | ReLU | Sort/TopK | Key Challenge |
|----------|-----------|------|-----------|---------------|
| **Ascend** | Cube (16×16 FP16) | FixP (free during writeback) | VBS32+VMS4v2 | Cube↔Vector data exchange via L2 |
| **MTT S5000** | MUSA Tensor Core | CUDA warp-level | cub::DeviceRadixSort | FP8 Tensor Core for 2× throughput |
| **Hygon K100** | Matrix Core (HIP) | ROCm kernel | hipCUB sort | Port from CUDA via HIP translation |
| **Cambricon** | Tensor Unit | BANG kernel | BANG sort primitive | MLU-native fusion via MagicMind |
| **Kunlun XPU** | XPU MatMul | XPU SIMD | XPU reduce+sort | INT8 quantized scoring |
| **MetaX C500** | CUDA-like Core | MXMACA kernel | MXMACA sort | CUDA-compatible, direct port |
| **Enflame GCU** | GCU-CARE compute | TopsCC kernel | TopsCC sort | GCU instruction scheduling |

---

## 2. Sparse Flash Attention Kernel

### Pseudocode

```
KERNEL sparse_flash_attention(
    Q: [S_q, H, D_qk],
    KV_cache: [num_pages, page_size, D_qk],
    indices: [S_q, TopK],            // from Lightning Indexer
    O: [S_q, H, D_v]
):
    // Grid: (ceil(S_q / BLOCK_Q), H)
    // BLOCK_Q = 64, BLOCK_KV = 64 (matches page_size)

    // Online softmax state
    m_prev[BLOCK_Q] = -infinity    // running max
    l_prev[BLOCK_Q] = 0.0         // running sum
    O_accum[BLOCK_Q, D_v] = 0.0   // running output

    // Load Q tile (resident)
    MTE.copy(GM[Q_block] → q_local)

    for kv_block in range(ceil(TopK / BLOCK_KV)):
        // --- MTE: Gather KV from cache using indices ---
        // This is the SPARSE part: only load selected positions
        selected_indices = indices[q_offset : q_offset + BLOCK_Q,
                                   kv_block * BLOCK_KV : (kv_block+1) * BLOCK_KV]
        // Gather KV entries from potentially non-contiguous cache pages
        for i in range(BLOCK_KV):
            if is_valid(selected_indices[i]):
                MTE.copy(KV_cache[page_of(idx), offset_of(idx)] → kv_local[i])

        // --- Cube: QK^T (OVERLAPS with MTE gather of next block) ---
        CUBE.matmul(q_local, kv_local_k, scores)  // [BLOCK_Q, BLOCK_KV]

        // --- Vector: Online Softmax ---
        m_new = max(m_prev, row_max(scores))
        correction = exp(m_prev - m_new)
        P = exp(scores - m_new)  // softmax numerator

        l_new = l_prev * correction + row_sum(P)
        O_accum = O_accum * correction  // rescale running output

        // --- Cube: P @ V ---
        CUBE.matmul(P, kv_local_v, O_accum, accumulate=True)

        m_prev = m_new
        l_prev = l_new

    // Final rescaling
    O = O_accum / l_prev
    MTE.copy(O → GM[O_block])
```

### Platform-Specific Notes

| Platform | Gather Strategy | Softmax | Key Optimization |
|----------|----------------|---------|-----------------|
| **Ascend** | MTE scatter-gather with page table | Vector unit online softmax | Cube(QK) overlaps with MTE(gather next) |
| **NVIDIA H100** | FlashMLA sparse indices tensor | Warp-level online softmax | TMA async copy overlaps with WGMMA |
| **MTT S5000** | MUSA global load with indices | FP8 reduces memory traffic | ACE overlaps communication |
| **Others** | Standard indexed load | Framework-provided softmax | Less optimization opportunity |

---

## 3. MLAPO Fusion Kernel

### Pseudocode (Single Fused Kernel)

```
KERNEL mlapo_preprocess(
    hidden: [B, S, 6144],       // input
    // Weights (all loaded once, resident):
    W_qa: [2048, 6144],         // q_a_proj
    W_qb: [16384, 2048],        // q_b_proj
    W_kva: [576, 6144],         // kv_a_proj_with_mqa
    W_kvb: [28672, 512],        // kv_b_proj
    qa_norm_weight: [2048],     // q_a_layernorm
    kva_norm_weight: [512],     // kv_a_layernorm
    cos_cache: [S, 64],         // RoPE cos
    sin_cache: [S, 64],         // RoPE sin
    // Outputs:
    Q: [B, 64, S, 256],
    K: [B, 64, S, 256],
    V: [B, 64, S, 256]
):
    // Process tokens in tiles
    for token_tile in range(ceil(B*S / TILE_TOKENS)):

        // --- Op 1: q_a_proj (Cube) ---
        q_compressed = CUBE.matmul(hidden_tile, W_qa.T)  // [tile, 6144] × [6144, 2048] → [tile, 2048]

        // --- Op 2: q_a_layernorm (Vector, OVERLAPS with Op 8 Cube) ---
        variance = VECTOR.reduce_mean(q_compressed², dim=-1)
        q_normed = VECTOR.multiply(q_compressed, rsqrt(variance + eps) * qa_norm_weight)

        // --- Op 8: kv_a_proj (Cube, runs IN PARALLEL with Op 2 Vector) ---
        kv_compressed = CUBE.matmul(hidden_tile, W_kva.T)  // → [tile, 576]

        // --- Op 9: Split kv (Vector, trivial pointer arithmetic) ---
        k_comp = kv_compressed[:, :512]
        k_pe_raw = kv_compressed[:, 512:]

        // --- Op 3: q_b_proj (Cube) ---
        q_expanded = CUBE.matmul(q_normed, W_qb.T)  // [tile, 2048] × [2048, 16384] → [tile, 16384]

        // --- Op 10: kv_a_layernorm (Vector, OVERLAPS with Op 3 Cube) ---
        k_normed = VECTOR.rmsnorm(k_comp, kva_norm_weight)

        // --- Op 4-7: Q reshape + split + RoPE + cat (Vector) ---
        q_heads = VECTOR.reshape(q_expanded, [64, 256])
        q_nope = q_heads[:, :192]
        q_pe = q_heads[:, 192:]
        q_pe_rotated = VECTOR.rope(q_pe, cos_cache, sin_cache)
        Q_tile = VECTOR.cat(q_nope, q_pe_rotated)

        // --- Op 11: kv_b_proj (Cube) ---
        kv_expanded = CUBE.matmul(k_normed, W_kvb.T)  // [tile, 512] × [512, 28672] → [tile, 28672]

        // --- Op 12-13: K/V reshape + RoPE (Vector, OVERLAPS with nothing — last step) ---
        kv_heads = VECTOR.reshape(kv_expanded, [64, 448])
        k_nope = kv_heads[:, :192]
        V_tile = kv_heads[:, 192:]
        k_pe_rotated = VECTOR.rope(k_pe_raw, cos_cache, sin_cache)
        k_pe_expanded = VECTOR.broadcast(k_pe_rotated, num_heads=64)
        K_tile = VECTOR.cat(k_nope, k_pe_expanded)

        // --- Write outputs to GM ---
        MTE.copy(Q_tile → GM[Q])
        MTE.copy(K_tile → GM[K])
        MTE.copy(V_tile → GM[V])
```

### Key Optimization: Cube-Vector Overlap Schedule

```
Timeline for one token tile:

Cube:   [Op1: q_a_proj]──[Op3: q_b_proj]──[Op8: kv_a_proj]──[Op11: kv_b_proj]
Vector: ─────────────────[Op2: norm]──[Op4-7: reshape+RoPE]──[Op10: norm]──[Op12-13: reshape+RoPE]

The 4 Cube matmuls and 9 Vector ops execute in an interleaved pipeline,
with Vector processing the output of the PREVIOUS Cube op while Cube
processes the NEXT matmul.
```

---

## Implementation Requirements Summary

To implement these kernels on a new platform, you need:

### Minimum Hardware Requirements

| Capability | Lightning Indexer | Sparse FA | MLAPO |
|-----------|-----------------|-----------|-------|
| Matrix multiply unit | Yes (16×16 or larger) | Yes | Yes (4 matmuls) |
| SIMD/vector unit | Yes (sort, reduce) | Yes (softmax) | Yes (norm, RoPE) |
| Async memory transfer | Critical (overlap) | Critical | Important |
| On-chip SRAM | ≥256KB (tiling) | ≥256KB | ≥512KB (multiple intermediates) |
| Double buffering | Required | Required | Required |
| FP8 support | Recommended | Recommended | Not needed (BF16 intermediates) |
| Deterministic sort | Required (TopK) | Not needed | Not needed |

### Software Requirements

| Capability | What For |
|-----------|---------|
| Kernel fusion framework | Combining multiple ops into one launch |
| Tiling API | Breaking large tensors into cache-friendly tiles |
| Async copy primitives | Overlapping memory transfer with compute |
| Barrier/sync primitives | Coordinating Cube↔Vector handoffs |
| Online softmax | Numerically stable attention without full materialization |
| Scatter-gather memory ops | Sparse KV cache access |
