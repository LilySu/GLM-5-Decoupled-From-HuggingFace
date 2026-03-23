# GLM-5 Mixed-Precision Quantization Strategy

## Overview

GLM-5 uses a **module-aware mixed-precision** quantization scheme: different model components get different precision levels based on their sensitivity and memory footprint.

## Precision Assignment

| Component | Precision | Format | Rationale |
|-----------|-----------|--------|-----------|
| Attention (QKV projections) | W8A8 | INT8 weights, INT8 activations | Dense computation, moderate sensitivity |
| Attention (output projection) | W8A8 | INT8 | Same |
| Dense MLP (layers 0-2) | W8A8 | INT8 | Small number of layers |
| MoE Expert layers (layers 3-77) | W4A8 | INT4 weights, INT8 activations | 256 experts × large weights; INT4 critical for memory |
| MoE Shared Expert | W8A8 | INT8 | Only 1 shared expert per layer |
| KV Cache | FP8 or BF16 | E4M3 or BF16 | Platform-dependent |
| Embeddings | BF16 | BF16 | Not quantized |
| LM Head | BF16 | BF16 | Not quantized |

## Why W4A8 for MoE Experts

The 256 routed experts dominate model weight storage:
```
Per MoE layer: 256 × (gate_up: 2×2048×6144 + down: 6144×2048) × 2 bytes (BF16)
             = 256 × 3 × 2048 × 6144 × 2
             = 18.9 GB per layer in BF16

75 MoE layers × 18.9 GB = 1,417 GB total expert weights
```

With W4A8: expert weights compressed 4× → **354 GB**, fitting in single Atlas 800T A3 (1TB).

## Quantization Techniques

### QuaRot — Anomaly Suppression

Rotates the weight and activation spaces to reduce outlier magnitudes before quantization. Prevents the few extreme activation values from dominating the quantization range and degrading overall precision.

### Flex_AWQ_SSZ — Scaling Calibration

Flexible Activation-Weighted Quantization with Smooth Scale Zeropoint. Calibrates per-channel scaling factors using activation statistics from a calibration dataset, finding optimal scale/zero-point that minimize quantization error weighted by activation importance.

## Memory Budget (Single Atlas 800T A3 Node)

```
Available: 8 NPUs × 128 GB = 1,024 GB

W4A8 model:
  Expert weights (W4):  256 × 75 layers × 3 × 2048 × 6144 × 0.5 bytes = 354 GB
  Dense weights (W8):   attention + MLP layers 0-2 ≈ 50 GB
  Activations (A8):     runtime ≈ 50-100 GB (batch/seq dependent)
  KV Cache (FP8):       78 layers × seq_len × 576 × 1 byte = varies
  Overhead:             ~100 GB (framework, buffers)

Total: ~600-700 GB → FITS in single node
```

## Per-Platform Quantization Support

| Platform | W4A8 | W8A8 | FP8 | BF16 |
|----------|------|------|-----|------|
| Ascend 910B | Yes | Yes | Yes (KV) | Yes (2-node) |
| MTT S5000 | TBD | Yes | Yes (native HW) | Yes |
| Hygon K100 | Yes | Yes | No | Yes |
| Cambricon | Yes (FP8+INT4 mixed) | Yes | Yes | Yes |
| Kunlun XPU | No | Yes (INT8) | No | Yes |
| MetaX C500 | TBD | Yes | TBD | Yes |
| Enflame GCU | TBD | Yes | No | Yes |
