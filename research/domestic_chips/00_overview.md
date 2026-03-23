# GLM-5 Domestic Chip Platforms — Overview

GLM-5 (744B parameters, 40B active) is the first frontier-scale open-weight model:
- Trained entirely on domestic Huawei Ascend 910B chips (~100K chip cluster)
- Deployed across 7 domestic chip platforms with Day-0 support
- Features 3 specialized kernel fusions (MLAPO, Lightning Indexer, Sparse Flash Attention) designed for Ascend NPU

## The 7 Platforms

| # | Chinese | English | Company | Architecture | Flagship Chip |
|---|---------|---------|---------|-------------|---------------|
| 1 | 华为昇腾 | Huawei Ascend | Huawei | Da Vinci NPU | Ascend 910B (Atlas 800T A3) |
| 2 | 摩尔线程 | Moore Threads | Moore Threads | MUSA GPU | MTT S5000 |
| 3 | 海光 | Hygon | Hygon | DCU (AMD-derived) | K100 AI |
| 4 | 寒武纪 | Cambricon | Cambricon | MLU | MLU590 / MLU370 |
| 5 | 昆仑芯 | Kunlun Chip | Baidu Kunlun | XPU | Kunlun 2nd Gen |
| 6 | 沐曦 | MetaX (Muxi) | MetaX | GPU (MXMACA) | C500 (曦云) |
| 7 | 燧原 | Enflame | Enflame | GCU | CloudBlazer S60 |

## Performance Claims

- Single domestic node (Atlas 800T A3) ≈ 2× international GPU nodes for inference
- 50% cost reduction for long-sequence (>32K) deployment
- Lightning Indexer at 128K: 30GB memory savings, 8× faster than Flash Attention
- KV Cache overhead reduced 75% via MLA compression

## File Index

- `01_ascend_architecture.md` — Ascend 910B Da Vinci NPU microarchitecture
- `02_kernel_fusions.md` — MLAPO, Lightning Indexer, Sparse Flash Attention deep dive
- `03_per_platform_specs.md` — Hardware specs and adaptation details per platform
- `04_kernel_pseudocode.md` — Pseudocode and implementation requirements for each kernel
- `05_quantization.md` — Mixed-precision quantization strategy (W4A8/W8A8)
- `06_training_infrastructure.md` — Training on 100K Ascend chips
- `../reference.md` — All sources organized by topic
