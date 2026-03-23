# Ascend 910B Da Vinci NPU — Microarchitecture Deep Dive

## Chip-Level Configuration

| Spec | Ascend 910B (Atlas 800T A3) |
|------|---------------------------|
| Process | 7nm+ EUV |
| AI Cores | 20 Cube units + 40 Vector units (2:1 ratio) |
| FP16 peak | 376 TFLOPS |
| INT8 peak | 512+ TOPS |
| HBM | 64GB HBM2e |
| HBM bandwidth | 800 GB/s (910B4 variant) |
| L2 cache | 32 MB on-chip shared |
| TDP | 350W |
| Inter-chip | HCCS (Huawei Cache Coherence System) |
| Nodes | 8 NPUs per Atlas 800T A3 server (128GB × 8 = 1TB aggregate) |

## AI Core Internal Architecture

Each AI Core consists of **one AI Cube (AIC) core** and **two AI Vector (AIV) cores** (910B series).

### Cube Unit (AIC — AI Cube Core)

The matrix multiplication engine.

```
Input:  matrix A [M, K] × matrix B [K, N]
Output: matrix C [M, N]

Native tile size (FP16): 16 × 16 × 16 = 4,096 MACs per cycle
Native tile size (INT8):  16 × 32 × 32 = 8,192 MACs per cycle

Supported output accumulation: FP32 (from FP16 inputs), INT32 (from INT8 inputs)
Built-in: result accumulation, selected activation functions (ReLU via FixP), quantization
```

**Memory hierarchy for Cube:**
```
Global Memory (HBM)
    ↓ MTE (Memory Transfer Engine)
L1 Buffer (scratchpad)
    ↓ MTE
L0A Buffer (left matrix input)     ~64KB, double-buffered
L0B Buffer (right matrix input)    ~64KB, double-buffered
    ↓ Cube computation
L0C Buffer (output accumulator)    ~128KB, double-buffered
    ↓ MTE
L1 Buffer → Global Memory
```

**Key capability: ReLU via FixP component.** The Cube core has a FixP (Fixed-Point) unit that can apply ReLU during the data movement from L0C to L1, **without requiring a separate Vector operation**. This is how the Lightning Indexer fuses ReLU into the matmul.

### Vector Unit (AIV — AI Vector Core)

SIMD processing engine for element-wise operations.

```
SIMD width: 2048 bits = 128 × FP16 elements per cycle
Operations: add, mul, exp, log, sigmoid, softmax, gather, scatter, reduce, sort

VBS32: Stable descending sort on groups of 32 elements
VMS4v2: Merge 2-4 sorted vectors (used in TopK)
```

**Memory for Vector:**
```
Global Memory (HBM)
    ↓ MTE
Unified Buffer (UB)    ~256KB scratchpad
    ↓ Vector computation
Unified Buffer → Global Memory
```

### Scalar Unit

Program counter, address generation, loop control, branch logic. Dispatches instructions to Cube, Vector, and MTE queues.

## Instruction Queue Model

```
┌─────────────────────────────────────────────┐
│                 Scalar Unit                   │
│   (program control, address generation)       │
├──────────┬──────────┬──────────┬─────────────┤
│ MTE Queue│Cube Queue│Vec Queue │ MTE Queue   │
│ (CopyIn) │ (MatMul) │(Elem-ws) │ (CopyOut)   │
└──────────┴──────────┴──────────┴─────────────┘

Instructions within a single queue: executed in ORDER
Instructions across queues: can execute CONCURRENTLY
Synchronization: programmer's responsibility (AscendC provides queue API)
```

## Three-Stage Pipeline Model (CopyIn / Compute / CopyOut)

AscendC programs decompose into three pipelined stages:

```
Stage 1 — CopyIn:   MTE moves data from GM → L1/UB (async, overlaps with compute)
Stage 2 — Compute:  Cube or Vector processes data in local buffers
Stage 3 — CopyOut:  MTE moves results from L1/UB → GM (async)

With double buffering (queue capacity = 2):
  Iteration N CopyIn  ←→  Iteration N-1 Compute  ←→  Iteration N-2 CopyOut
```

## Critical Architecture Constraint: Cube-Vector Split

**Cube and Vector cores are SEPARATE.** Data can only be exchanged between them via **Global Memory or L2 cache**. There is no direct register-level or shared-memory communication.

```
AIC (Cube Core)          AIV (Vector Core)
  L0A, L0B, L0C            Unified Buffer
       ↕                         ↕
    L1 Buffer                 L1 Buffer
       ↕                         ↕
  ─── L2 Cache / Global Memory ───
```

This means:
- A Cube operation's output (in L0C) must be written to L1 → L2/GM before Vector can read it
- The Lightning Indexer exploits this by overlapping: while Cube writes batch N, Vector processes batch N-1
- Each data transfer between AIC and AIV is expensive — fusion reduces these transfers

## AscendC Programming Model

AscendC is a C++17 superset with built-in keywords for NPU programming:

```cpp
// Kernel structure
__global__ void my_kernel(GM_ADDR x, GM_ADDR y) {
    // Allocate local buffers
    LocalTensor<float16_t> x_local = inQueueX.AllocTensor<float16_t>();

    // CopyIn: GM → UB
    DataCopy(x_local, x_gm[offset], count);
    inQueueX.EnQue(x_local);

    // Compute: process in UB
    LocalTensor<float16_t> x_compute = inQueueX.DeQue<float16_t>();
    Add(y_local, x_compute, bias_local, count);
    outQueueY.EnQue(y_local);

    // CopyOut: UB → GM
    LocalTensor<float16_t> y_out = outQueueY.DeQue<float16_t>();
    DataCopy(y_gm[offset], y_out, count);
}
```

**Pipe objects** manage synchronization between stages. Setting queue capacity to 2 enables double buffering automatically.

## Relevance to GLM-5 Kernel Fusions

The Ascend architecture's Cube-Vector split is both a **constraint and an opportunity**:

- **Constraint:** Every Cube→Vector handoff costs a memory round-trip
- **Opportunity:** Cube and Vector can run **in parallel on different data** — when fused into one kernel, batch N's Cube matmul overlaps with batch N-1's Vector activation/reduction
- **MLAPO** fuses 13 small operators to eliminate 12 Cube↔Vector round-trips
- **Lightning Indexer** overlaps Cube (score matmul) with Vector (ReLU+reduce+TopK) using the FixP trick for ReLU
