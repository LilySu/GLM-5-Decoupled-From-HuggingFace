# Per-Platform Hardware Specs and GLM-5 Adaptation

## 1. Huawei Ascend 910B (华为昇腾)

| Spec | Value |
|------|-------|
| Architecture | Da Vinci v2 (NPU) |
| Process | 7nm+ EUV |
| AI Cores | 20 Cube + 40 Vector |
| FP16 | 376 TFLOPS |
| INT8 | 512+ TOPS |
| Memory | 64 GB HBM2e |
| Bandwidth | 800 GB/s |
| TDP | 350W |
| Interconnect | HCCS (8-card NVSwitch equivalent) |
| Server | Atlas 800T A3 (8 NPUs, 1TB aggregate) |

**GLM-5 adaptation:**
- **Kernels:** MLAPO, Lightning Indexer, Sparse Flash Attention (all Ascend-specific)
- **Frameworks:** vLLM-Ascend, SGLang, xLLM
- **Quantization:** W4A8 (single node), W8A8 (xLLM), BF16 (2-node)
- **Training:** Entire model trained on ~100K Ascend 910B, MindSpeed framework
- **Deployment:** Single A3 node fits W4A8 model; 2 nodes for BF16
- **Programming:** AscendC (C++17 superset), TileLang-Ascend, CANN SDK

## 2. Moore Threads MTT S5000 (摩尔线程)

| Spec | Value |
|------|-------|
| Architecture | MUSA v4 "Pinghu" (GPU) |
| AI Compute | 1000 TFLOPS peak |
| Memory | 80 GB VRAM |
| Bandwidth | 1.6 TB/s |
| Inter-card | 784 GB/s |
| Precision | FP8, BF16, FP16, FP32, FP64 |
| TDP | ~350W |
| Special | ACE (Asynchronous Communication Engine), high-ratio SFU |

**GLM-5 adaptation:**
- **Day-0 support** — full inference chain verified on release day
- **Framework:** SGLang-MUSA inference engine
- **Key optimization:** Native FP8 Tensor Core acceleration (doubles throughput vs BF16)
- **ACE engine:** Offloads communication to dedicated hardware, freeing ~15% compute
- **SFU:** Higher ratio of Special Function Units for exp/softmax acceleration
- **Programming:** MUSA SDK (CUDA-like API)

## 3. Hygon K100 AI (海光 DCU)

| Spec | Value |
|------|-------|
| Architecture | DCU / GPGPU (AMD GCN derivative) |
| FP32 | 49 TFLOPS |
| TF32 | 96 TFLOPS |
| BF16/FP16 | 192 TFLOPS |
| INT8 | 392 TOPS |
| Memory | 64 GB HBM |
| Bandwidth | 896 GB/s |
| TDP | 350W |
| Software | ROCm stack, HIP programming model |

**GLM-5 adaptation:**
- **Software stack:** ROCm-compatible, HIP programming (AMD-like)
- **Advantage:** x86 ecosystem compatibility, easier porting from CUDA via HIP
- **Performance:** ~60% of A100 baseline
- **Quantization:** INT8 and INT4 supported
- **Framework:** Paddle, custom inference engines

## 4. Cambricon MLU (寒武纪)

| Spec | Value |
|------|-------|
| Architecture | MLUarch03 |
| Process | 7nm (chiplet design, MLU590) |
| INT8 | 256 TOPS (MLU370) |
| Transistors | 39 billion (MLU370) |
| Memory | LPDDR5 (MLU370), HBM2e (MLU590) |
| Special | Hardware operator fusion, Supercharger module |

**GLM-5 adaptation:**
- **FP8+INT4 mixed quantization** — first production deployment on domestic chips
- **BANG language:** Cambricon's programming model (Host-Device heterogeneous)
- **MagicMind:** Inference engine with graph optimization
- **Hardware fusion:** MLUarch03 supports multi-operator hardware fusion natively
- **GLM-4.6 → GLM-5:** Built on prior Cambricon adaptation work

## 5. Baidu Kunlun XPU (昆仑芯)

| Spec | Value |
|------|-------|
| Architecture | XPU-KL2 (2nd gen) |
| Process | 7nm |
| INT8 | 256 TOPS |
| FP16 | 128 TFLOPS |
| Memory | 32 GB GDDR6 |
| Bandwidth | 512 GB/s |
| TDP | 120W |

**GLM-5 adaptation:**
- **DSA + MoE natively adapted** on Kunlun XPU
- **Quantization:** INT8
- **MTP:** Multi-Token Prediction supported
- **Parallelism:** Dual-machine PP (pipeline parallelism)
- **Framework:** vLLM-Kunlun Plugin (open-sourced on GitHub: `baidu/vLLM-Kunlun`)
- **Provider:** Baidu Baige (百舸) cloud platform
- **Frameworks:** vLLM + SGLang both verified

## 6. MetaX C500 / 沐曦 曦云 (沐曦)

| Spec | Value |
|------|-------|
| Architecture | XCORE 1.0 (GPU) |
| FP32 | 36 TFLOPS (OAM) |
| FP16/BF16 | 280 TFLOPS |
| INT8 | 560 TOPS |
| Memory | 64 GB HBM2e |
| Bandwidth | 1.8 TB/s |
| Interconnect | MetaXLink (7 ports, 8-card full mesh) |

**GLM-5 adaptation:**
- **Software:** MXMACA stack (CUDA-compatible API)
- **Performance:** Close to NVIDIA A800 in benchmarks
- **Advantage:** High memory bandwidth (1.8 TB/s > A100's 2.0 TB/s)
- **Ecosystem:** API-level CUDA compatibility accelerates porting

## 7. Enflame CloudBlazer (燧原)

| Spec | Value |
|------|-------|
| Architecture | GCU-CARE v2 (CloudBlazer S60) |
| Process | MCM packaging |
| FP32 | 32 TFLOPS (Suisi 2.5, i20) |
| TF32 | 128 TFLOPS (i20) |
| INT8 | 256 TOPS (i20) |
| Memory | 16 GB HBM2e (i20) |
| Bandwidth | 819 GB/s (i20) |
| Interconnect | NOC (Network-on-Chip) |
| CPU | Multi-core ARM Cortex A55 + RISC-V |
| PCIe | 5.0 |

**GLM-5 adaptation:**
- **Software:** TopsRider SDK (TopsCC compiler, TopsATen operator library)
- **Training card:** Suisi 2.0 (40 TFLOPS FP32)
- **Inference card:** CloudBlazer S60
- **Framework:** TopsPlatform (heterogeneous compute platform)
- **Precision:** FP32, TF32, FP16, BF16, INT8

---

## Cross-Platform Comparison

| Platform | FP16 TFLOPS | Memory | BW (TB/s) | Programming | CUDA-like? |
|----------|------------|--------|-----------|-------------|-----------|
| Ascend 910B | 376 | 64 GB | 0.80 | AscendC | No (unique) |
| MTT S5000 | ~500 | 80 GB | 1.60 | MUSA | Yes |
| Hygon K100 | 192 | 64 GB | 0.90 | HIP/ROCm | Yes (AMD) |
| Cambricon MLU | ~128 | varies | varies | BANG | No (unique) |
| Kunlun XPU | 128 | 32 GB | 0.51 | KLANG | No (unique) |
| MetaX C500 | 280 | 64 GB | 1.80 | MXMACA | Yes |
| Enflame S60 | ~64 | 16 GB | 0.82 | TopsCC | No (unique) |
