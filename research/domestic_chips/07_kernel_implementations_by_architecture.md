# DSA Kernel Implementations by Hardware Architecture

This document catalogs every available code artifact, pseudocode pattern, and implementation
detail for the DSA (DeepSeek Sparse Attention) indexer, sparse attention, and MLAPO kernels
across all hardware architectures. The goal is to enable 1:1 performance replication of these
kernels, including reverse-engineering the Ascend-specific kernels in CUDA where applicable.

---

## Architecture 1: NVIDIA H100 / SM90 (CUDA — DeepGEMM)

### Source: `deepseek-ai/DeepGEMM` — `sm90_fp8_mqa_logits.cuh`

**Status:** Complete CUDA kernel source available (open-source, MIT license)

#### Kernel Signature

```cuda
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_mqa_logits(
    const uint32_t seq_len, const uint32_t seq_len_kv,
    const uint32_t max_seqlen_k, const uint64_t stride_logits,
    uint32_t* cu_seq_len_k_start,    // per-query KV range start
    uint32_t* cu_seq_len_k_end,      // per-query KV range end
    float* logits,                    // output [seq_len, stride_logits]
    const cute::TmaDescriptor tensor_map_q,
    const cute::TmaDescriptor tensor_map_kv,
    const cute::TmaDescriptor tensor_map_kv_scales,
    const cute::TmaDescriptor tensor_map_weights
);
```

#### Template Parameters (from JIT dispatch)

```
block_qh = 128 (constant)
block_q = 128 / num_heads  (e.g., 4 for num_heads=32, 2 for 64)
block_kv = 256
num_q_stages = 3
num_kv_stages = 3
num_specialized_threads = 128 (TMA threads)
num_math_threads = 512 (SM90) or 256 (SM100)
```

#### Architecture: Warp Specialization

```
Thread 0 .. num_math_threads-1:     Math warp-groups (WGMMA)
Thread num_math_threads .. total-1:  TMA warp-group (data loading)

Math warps: 112 registers each
TMA warps:  32 registers each
```

#### Shared Memory Layout

```
smem_q[kNumQStages]:           BLOCK_Q * kNumHeads * kHeadDim * sizeof(fp8)
smem_kv[kNumKVStages]:         BLOCK_KV * kHeadDim * sizeof(fp8)
smem_weights[kNumQStages]:     BLOCK_Q * kNumHeads * sizeof(float)
smem_kv_scales[kNumKVStages]:  BLOCK_KV * sizeof(float)
barriers:                       Q full/empty + KV full/empty
```

#### Computation Pattern

```
TMA Thread:
  while (q_blocks remain):
    issue_tma_q(q_block)          // Load Q + weights via TMA
    for kv_block in range(num_kv_blocks):
      wait(kv_empty_barrier)
      issue_tma_kv(kv_block)      // Load KV + scales via TMA

Math Thread:
  while (q_blocks remain):
    wait(q_full_barrier)
    read_weights_to_registers()
    for kv_block in range(num_kv_blocks):
      wait(kv_full_barrier)
      read_kv_scales()

      // WGMMA: [BLOCK_Q*kNumHeads, kHeadDim] × [BLOCK_KV, kHeadDim]^T
      // → accum[BLOCK_Q * kNumHeads/2]
      for k in range(kHeadDim / WGMMA::K):
        WGMMA.wgmma(smem_kv_desc, smem_q_desc, accum)

      release(kv_empty_barrier)

      // Post-WGMMA: ReLU + weight multiply + head reduction + store
      for each q_token i:
        for each accum element j:
          transform(j) = max(accum[j], 0) * weights[i][j_head_idx]

        // Intra-thread: sum transform elements
        sum = reduce_add(transform)
        // Scale by KV quantization scales
        v_0 = sum_even * scale_kv_0
        v_1 = sum_odd  * scale_kv_1

        // Inter-thread: warp shuffle reduction across 4 threads
        for mask in [1, 2]:
          v_0 += __shfl_xor_sync(v_0, mask)
          v_1 += __shfl_xor_sync(v_1, mask)

        // Store to global
        logits[q_idx, kv_offset] = v_0 or v_1
```

#### Key H100 Features Used

- **TMA (Tensor Memory Accelerator):** Async bulk global→shared copies via hardware descriptor
- **WGMMA (Warp Group MMA):** Native fp8 matrix multiply on Hopper tensor cores
- **Warp Specialization:** Separate TMA and math warps avoid instruction cache conflicts
- **Barrier synchronization:** `ClusterTransactionBarrier` for producer-consumer

---

## Architecture 2: Huawei Ascend 910B (AscendC — vllm-ascend)

### Source: `vllm-project/vllm-ascend` — `csrc/lightning_indexer_vllm/`

**Status:** Complete AscendC kernel source available (open-source, CANN Open Software License)

#### File Structure

```
lightning_indexer_vllm/
  op_kernel/
    lightning_indexer_service_cube.h     — Cube matmul pipeline (Q×K scoring)
    lightning_indexer_service_vector.h   — Vector sort/topk + weight reduce
    lightning_indexer_vector.h           — Low-level vector operations
    lightning_indexer_common.h           — Shared constants and types
    lightning_indexer_kernel.h           — Top-level kernel orchestration
    lightning_indexer_vllm.cpp           — Entry point
  op_host/
    lightning_indexer_vllm_tiling.h      — Tiling configuration
    lightning_indexer_vllm_tiling.cpp    — Tiling strategy computation
```

#### Cube Service — Key Constants

```cpp
M_BASIC_BLOCK     = 256     // Query block height in L1
D_BASIC_BLOCK     = 128     // Head dimension in L1
S2_BASIC_BLOCK    = 256     // KV block width in L1

M_BASIC_BLOCK_L0  = 128     // Tile height in L0 (Cube native tile)
D_BASIC_BLOCK_L0  = 128     // Head dimension in L0
S2_BASIC_BLOCK_L0 = 128     // KV tile width in L0

KEY_BUF_NUM   = 3   // Triple-buffered K in L1
QUERY_BUF_NUM = 2   // Double-buffered Q in L1
L0_BUF_NUM    = 2   // Double-buffered L0A/L0B/L0C
```

#### Buffer Allocation

```cpp
// L1 (scratchpad) buffers
bufQL1_:    QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T)
            = 2 * 256 * 128 * 1 = 64 KB (FP8) or 128 KB (FP16)
bufKeyL1_:  KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T)
            = 3 * 256 * 128 * 1 = 96 KB (FP8) or 192 KB (FP16)

// L0 (Cube-native) buffers
bufQL0_:    L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T)
            = 2 * 128 * 128 = 32 KB (FP8)
bufKeyL0_:  L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T)
            = 2 * 128 * 128 = 32 KB (FP8)
bufL0C_:    L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float)
            = 2 * 128 * 128 * 4 = 128 KB
```

#### Pipeline (Cube: matmul, MTE: data load)

```
Event synchronization:
  KEY_MTE1_MTE2_EVENT = EVENT_ID2     // K data ready
  QUERY_MTE1_MTE2_EVENT = EVENT_ID5   // Q data ready
  M_MTE1_EVENT = EVENT_ID3            // Matmul result ready

Pipeline pattern:
  for s2_offset in range(0, s2_process_size, S2_BASIC_BLOCK):
    Wait(KEY_MTE_EVENT + keyL1BufIdx % 3)          // Wait for K triple-buffer
    KeyNd2Nz(k_tile)                                // Convert K layout: ND → NZ format
    SetFlag(MTE1_M_EVENT)                           // Signal K ready for Cube

    for s1g_offset in range(0, s1g_process_size, M_BASIC_BLOCK_L0):
      LoadQueryToL0a(q_tile)                        // Load Q to L0A (if not resident)
      LoadKeyToL0b(k_tile)                          // Load K to L0B
      ComuteL0c(s1g_size, s2_size)                  // Cube: Q × K → L0C
      Fixp(s1g_offset, s2_offset)                   // FixP: ReLU during L0C→L1 writeback
```

#### Vector Service — TopK via Sort + Merge

```cpp
constexpr uint32_t BASE_TOPK = 2048;    // GLM-5 / DeepSeek indexer topk

// Buffer sizes for Vector operations:
inQueue:      groupInner_ * s2BaseSize_ * sizeof(float)     // ~69 KB matmul output
outQueue:     max(BASE_TOPK * 2 * 2 * sizeof(float),        // ~32 KB sort workspace
                  reduceCacheSize)
sortOutBuf:   (s1BaseSize_/2) * BASE_TOPK * 2 * sizeof(float) // ~64 KB running topk state
indexBuf:     s2BaseSize_ * sizeof(int32_t)                   // ~2 KB index generation
reduceOutBuf: s2BaseSize_ * 2 * sizeof(float)                 // ~4 KB reduce workspace

// TopK algorithm: progressive sort-merge
// 1. ArithProgression generates initial indices [0, 1, 2, ..., s2BaseSize_]
// 2. For each KV tile's matmul output:
//    a. Load matmul results from Cube (via shared GM)
//    b. Apply weights: score *= weights[head]
//    c. ReduceSum across heads
//    d. Sort current tile's scores (VBS32: sort groups of 32)
//    e. Merge with running top-2048 (VMS4v2: merge sorted lists)
// 3. After all KV tiles: write final top-2048 indices to global memory
```

#### Cube↔Vector Coordination

```
The Cube and Vector cores run on SEPARATE AI cores (AIC vs AIV).
Data exchange happens through Global Memory or L2 cache:

Cube core writes matmul results → mm1ResGm (Global Memory)
Vector core reads mm1ResGm → processes in Unified Buffer (UB)

Pipeline overlap:
  While Vector processes KV tile N's scores (sort/merge):
    Cube computes KV tile N+1's matmul
    MTE loads KV tile N+2's data from GM to L1
```

---

## Architecture 3: AMD ROCm / Hygon DCU (Triton — AITER)

### Source: `ROCm/aiter` — `aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py`

**Status:** Complete Triton kernel source available (open-source)

#### Kernel Structure

```python
@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,           # fp8e4m3 [seq_len, H, D]
    KV_ptr,          # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,   # fp32 [seq_len_kv]
    weights_ptr,     # fp32 [seq_len, H]
    cu_start_ptr,    # int32 [seq_len] — per-query KV range start
    cu_end_ptr,      # int32 [seq_len] — per-query KV range end
    logits_ptr,      # fp32 [seq_len, seq_len_kv]
    ...
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.num_programs(0) - tl.program_id(0) - 1  # reverse order for load balancing

    # Load Q[1, NUM_HEADS, HEAD_SIZE] and weights[1, NUM_HEADS]
    q_block = tl.load(Q_ptr + row_id * stride_q_s + ...)
    w_block = tl.load(weights_ptr + row_id * stride_w_s + ...)

    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.minimum(tl.load(cu_end_ptr + row_id), seq_len_kv)

    # Main loop: unmasked KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)           # [HEAD_SIZE, BLOCK_KV] fp8
        kv_scales = tl.load(kv_scales_ptrs)   # [BLOCK_KV] fp32

        # Score = Q @ KV^T → [NUM_HEADS, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = scores * kv_scales[None, :]  # scale by KV quantization
        scores = tl.maximum(scores, 0.0)       # ReLU
        scores = scores * w_block              # weight per head
        scores = tl.sum(scores, axis=0)        # reduce across heads → [BLOCK_KV]

        tl.store(logits_ptrs, scores)

    # Tail: masked KV tile (boundary handling)
    ...
```

#### Key Differences from CUDA (DeepGEMM)

| Aspect | DeepGEMM (CUDA/H100) | AITER (Triton/ROCm) |
|--------|---------------------|---------------------|
| Matmul | WGMMA (hardware TMA) | `tl.dot` (Triton compiler) |
| Data loading | Explicit TMA descriptors | `tl.load` with cache hints |
| Warp specialization | Yes (separate TMA+math warps) | No (all threads compute) |
| Pipeline staging | Explicit barriers (3-stage Q, 3-stage KV) | `tl.range` with Triton's auto-pipeline |
| ReLU | Fused in accum reduce | `tl.maximum(scores, 0.0)` |
| Head reduction | Manual `__shfl_xor_sync` | `tl.sum(scores, axis=0)` |
| Block scheduling | Persistent kernel (`blockIdx.x` reuse) | Reverse program_id for load balance |
| Precision | FP8 WGMMA → FP32 accum | `tl.dot(input_precision="ieee")` |

#### Applicability to Hygon DCU

Hygon DCU uses ROCm/HIP stack. This AITER Triton kernel runs directly on Hygon K100 AI via:
- ROCm's Triton backend compiles to AMD ISA
- Same `tl.dot` maps to AMD Matrix Cores
- Same `tl.load/store` maps to HIP memory operations
- No code changes needed — Triton abstracts the hardware

---

## Architecture 4: Sparse Flash Attention (Ascend — vllm-ascend)

### Source: `vllm-project/vllm-ascend` — `csrc/sparse_flash_attention/`

#### File Structure

```
sparse_flash_attention/
  op_kernel/
    sparse_flash_attention_common.h           — Types, configs (SFA_LAYOUT enum)
    sparse_flash_attention_kernel_mla.h       — MLA-specific kernel orchestration
    sparse_flash_attention_service_cube_mla.h — Cube: QK^T and PV matmuls
    sparse_flash_attention_service_vector_mla.h — Vector: online softmax
    sparse_flash_attention.cpp                — Entry point
  op_host/
    sparse_flash_attention_tiling.h/.cpp      — Tiling strategy
```

#### Key Types

```cpp
enum class SFA_LAYOUT { BSND = 0, TND = 1, PA_BSND = 2 };

// Template supports: FP16 Q/KV, FP8 KV, BF16, with/without flash decode,
// paged attention, and MLA-specific layouts
template <typename Q_T, typename KV_T, typename OUT_T,
          const bool FLASH_DECODE, SFA_LAYOUT LAYOUT_T, SFA_LAYOUT KV_LAYOUT_T, ...>
struct SFAType;
```

#### Softmax Configuration

```cpp
constexpr SoftmaxConfig SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {
    false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC
};
// FlashAttention-2 style online softmax: no BRC (broadcast) mode
```

---

## Architecture 5: NVIDIA Sparse Attention (FlashMLA)

### Source: `deepseek-ai/FlashMLA` — `csrc/sm90/`

See `02_kernel_fusions.md` for details. Key kernel files:

```
sm90/decode/sparse_fp8/        — Sparse FP8 decode (token-level sparsity)
sm90/prefill/sparse/           — Sparse prefill (k=512 and k=576 instantiations)
sm90/decode/dense/             — Dense decode (no sparsity)
```

FlashMLA's sparse kernels use the same CUTLASS-based WGMMA approach as DeepGEMM
but operate on gathered KV entries specified by an indices tensor rather than
computing over all positions.

---

## Architecture 6: Baidu Kunlun XPU

### Source: `baidu/vLLM-Kunlun` (open-source vLLM plugin)

No kernel-level source code has been published for Kunlun XPU's DSA adaptation.
The adaptation uses:
- Kunlun's built-in XPU operators (opaque)
- INT8 quantization for scoring
- MTP multi-token prediction support
- PP (pipeline parallelism) across 2 machines

**Reverse-engineering approach for CUDA replication:**
- Use the AITER Triton kernel as a base (most portable)
- Replace `tl.dot` with Kunlun's XPU matmul primitive
- The sorting/topk step uses standard `torch.topk` (same across platforms)

---

## Architecture 7: Cambricon MLU

### Source: Not publicly available

Cambricon uses BANG language for kernel development. The adaptation includes:
- FP8+INT4 mixed quantization (first domestic deployment)
- MagicMind inference engine with graph-level fusion
- Hardware multi-operator fusion (MLUarch03 native)

**Reverse-engineering approach:**
- BANG syntax is similar to CUDA
- Tensor Unit maps to Cube (matrix multiply)
- Use the DeepGEMM kernel as structural reference
- Cambricon's Supercharger module handles convolution-like optimizations

---

## Architecture 8: MetaX C500 / Enflame GCU

### MetaX (MXMACA — CUDA-compatible API)

MetaX's MXMACA stack is API-compatible with CUDA. The DeepGEMM kernel can
theoretically be compiled directly for MetaX C500 with minimal changes
(replace CUDA-specific intrinsics with MXMACA equivalents).

### Enflame (TopsCC — custom compiler)

Enflame's GCU architecture uses a different instruction set (TopsCC).
No DSA-specific kernels have been published. The adaptation likely uses
TopsATen (pre-compiled operator library) for the matmul components and
custom TopsCC kernels for the fusion.

---

## Cross-Architecture Kernel Equivalence Table

This table maps each operation in the DSA indexer to its implementation across architectures:

| Operation | NVIDIA H100 (CUDA) | Ascend 910B (AscendC) | AMD ROCm (Triton) | Portable (PyTorch) |
|-----------|--------------------|-----------------------|--------------------|--------------------|
| Q×K^T matmul | WGMMA via TMA desc | Cube unit (L0A×L0B→L0C) | `tl.dot(q, kv)` | `torch.einsum('bshd,btd->bsht')` |
| FP8 dequant | Built into WGMMA | Built into Cube (scale factors) | `scores * kv_scales` | `(q.float() @ k.float())` |
| ReLU | In accum reduction loop | FixP during L0C→L1 writeback | `tl.maximum(scores, 0)` | `F.relu(scores)` |
| Weight multiply | Register multiply | Vector unit (UB) | `scores * w_block` | `scores * weights` |
| Head reduce | `__shfl_xor_sync` | Vector ReduceSum (UB) | `tl.sum(scores, axis=0)` | `torch.einsum('bsht,bsh->bst')` |
| TopK | Not in scoring kernel (separate `torch.topk`) | VBS32 sort + VMS4v2 merge | Not in scoring kernel | `torch.topk(logits, k=2048)` |
| Data prefetch | TMA async copy | MTE triple-buffer | `tl.load(.cg cache hint)` | N/A |
| Pipeline | Warp specialization (TMA vs math) | Event-based (MTE↔Cube↔Vector) | Triton auto-pipeline | N/A |
| Memory layout | Swizzled SMEM (128B alignment) | ND→NZ format conversion in L1 | Row-major (Triton handles) | Contiguous tensors |

---

## Benchmarking Harness Requirements

To validate a CUDA reimplementation of the Ascend kernels, the benchmark harness needs:

### Correctness Tests

1. **Bit-exact logits** — Compare CUDA kernel output against PyTorch reference
2. **TopK agreement** — Jaccard similarity > 0.99 between kernel TopK and `torch.topk`
3. **Variable-length** — Test with `cu_seqlens` representing different sequence lengths per batch element
4. **FP8 precision** — Compare FP8-quantized scoring vs BF16 reference (tolerance: atol=0.5, rtol=0.1)
5. **Boundary conditions** — seq_len=1, seq_len_kv=1, topk > seq_len_kv, empty sequences

### Performance Tests (per architecture)

| Metric | How to Measure | Target (H100) |
|--------|---------------|----------------|
| Scoring TFLOPS | `2 * seq_len * seq_len_kv * num_heads * head_dim / time / 1e12` | >200 TFLOPS (FP8) |
| Scoring bandwidth | `(Q_bytes + KV_bytes + logits_bytes) / time` | >1500 GB/s |
| TopK latency | Time for sort+merge after scoring | <1ms for S_kv=64K, k=2048 |
| Pipeline efficiency | `compute_time / (compute_time + idle_time)` | >80% |
| Tensor core utilization | ncu: `sm__pipe_tensor_op_hmma_cycles_active` | >60% |

### Per-Architecture Profiling

| Architecture | Profiling Tool | Key Metric |
|-------------|---------------|------------|
| NVIDIA H100 | `ncu --metrics sm__pipe_tensor_op_hmma...` | Tensor core % |
| Ascend 910B | CANN Profiler (msprof) | Cube/Vector pipeline overlap % |
| AMD ROCm | `rocprof --metrics` | Matrix core utilization |
| Kunlun XPU | Baidu XPU Profiler | XPU occupancy |
| Others | Framework-level timing | End-to-end throughput |
