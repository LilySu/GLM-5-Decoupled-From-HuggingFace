# Research Reference — All Sources by Topic

Organized by topic from all research conducted for the GLM-5 domestic chip platform and kernel analysis.

---

## GLM-5 Paper and Official Documentation

- [GLM-5 Paper (arXiv 2602.15763)](https://arxiv.org/abs/2602.15763) — Full paper, Section 5: Adapting GLM-5 to Chinese Chip Infrastructure
- [GLM-5 Paper HTML](https://arxiv.org/html/2602.15763v2) — HTML version for text extraction
- [GLM-5 GitHub (zai-org/GLM-5)](https://github.com/zai-org/GLM-5) — Official repository
- [GLM-5 README (Chinese)](https://github.com/zai-org/GLM-5/blob/main/README_zh.md) — Chinese README with deployment instructions
- [GLM-5 Ascend Deployment Guide](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md) — Atlas 800T A3 deployment configs
- [GLM-5 HuggingFace](https://huggingface.co/zai-org/GLM-5) — Model weights and documentation
- [GLM-5 Zhipu Official Docs](https://docs.bigmodel.cn/cn/guide/models/text/glm-5) — API documentation

## 7 Domestic Chip Platforms (General)

- [IT之家: GLM-5深度适配7大国产芯片](https://www.ithome.com/0/921/272.htm) — Day-0 support announcement, 7 platforms listed
- [财联社: GLM-5深度适配华为昇腾等七大国产芯片](https://www.cls.cn/detail/2287928) — 7 platform confirmation
- [观察者网: 多款国产芯片Day0支持GLM-5](https://www.guancha.cn/economy/2026_02_12_806895.shtml) — Day-0 vendor announcements
- [AIBase: GLM-5 Supports Seven Domestic Chip Platforms](https://news.aibase.com/news/25588) — English summary
- [IT之家: 智谱公开GLM-5技术细节](https://www.ithome.com/0/922/853.htm) — Technical details overview

## Ascend NPU Architecture

- [Parallel Scan on Ascend AI Accelerators](https://arxiv.org/html/2505.15112v1) — **Key source**: AiCore architecture (20 Cube + 40 Vector units, 2:1 ratio, 910B4 800 GB/s bandwidth, split Cube/Vector design)
- [AscendCraft: Automatic Ascend NPU Kernel Generation](https://arxiv.org/html/2601.22760v1) — CopyIn/Compute/CopyOut pipeline, instruction queue model
- [Accelerating Sparse Matrix-Matrix Multiplication with Ascend AI Core](https://accml.dcs.gla.ac.uk/papers/2023/5th_AccML_paper_10.pdf) — Cube 16×16 FP16, Vector 2048-bit SIMD, L0A/L0B/L0C buffer hierarchy
- [昇腾910 AI芯片技术全面概述](https://www.eefocus.com/article/1840602.html) — Da Vinci architecture overview, 32 AI Cores, 256 TFLOPS FP16
- [IC设计: 昇腾910架构学习](https://blog.csdn.net/qq_42622433/article/details/141096273) — Cube 4096 MACs/cycle, Vector unit details
- [GPU进阶笔记: 华为昇腾910B](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/) — Operational aspects, npu-smi, A100 comparison
- [FastAttention: Extend FlashAttention2 to NPUs](https://openreview.net/pdf?id=76NYyOrnfk) — Two-level tiling, Cube-Vector pipeline for attention on Ascend
- [Huawei Atlas AI Computing Solution (Springer)](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_6) — Da Vinci architecture formal spec

## Ascend Kernel Development (AscendC, CANN, TBE)

- [CANN自定义算子开发教程 (清华大学出版社)](http://www.tup.com.cn/upload/books/yz/092962-01.pdf) — Official textbook on CANN operator development
- [昇腾课: 漫谈异构计算架构CANN](https://zhuanlan.zhihu.com/p/1893967687966773644) — CANN architecture overview
- [模型推理: 昇腾CANN TBE算子开发方式](https://zhuanlan.zhihu.com/p/417014599) — TBE operator development patterns
- [AscendC算子编程范式解析](https://ascendai.csdn.net/693a965a96fa167eeeccc119.html) — Kernel structure, tiling, pipeline
- [Ascend C自定义算子开发实战](https://ascendai.csdn.net/6927066e2087ae0db79cddd6.html) — From basics to performance tuning
- [TileLang-Ascend (GitHub)](https://github.com/tile-ai/tilelang-ascend) — Open-source TileLang adapter for Ascend NPU

## Lightning Indexer Kernel

- [Lightning Indexer Operator (DeepWiki)](https://deepwiki.com/chenqi123/cann-recipes-infer/5.1-lightning-indexer-operator) — **Key source**: Memory tiling (320KB L1), triple-buffered K, VBS32 sort, VMS4v2 merge, Cube preload pipeline, ReLU via FixP
- [cann-recipes-infer: DeepSeek V3.2 AscendC Operator Guide](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek-v3.2-exp/deepseek_v3.2_exp_ascendc_operator_guide.md) — AscendC implementation guide
- [cann-recipes-infer: TileLang Operator Guide](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek-v3.2-exp/deepseek_v3.2_exp_tilelang_operator_guide.md) — TileLang implementation guide
- [TileLang-DSA (GitHub lemyx/tilelang-dsa)](https://github.com/lemyx/tilelang-dsa) — **Open-source** BF16 training Lightning Indexer kernel with KL-divergence loss
- [fp8_lighting_indexer.py Analysis (GitHub Gist)](https://gist.github.com/createthis/0cce8a250daa3a117cb2986c743c02f2) — FP8 scoring kernel analysis with TileLang

## MLAPO Fusion

- [昇腾0day支持GLM-5: 744B模型单机高效推理](https://www.hiascend.com/activities/dynamic-news/648) — **Key source**: MLAPO 13→1 fusion, VV fusion technology, Cube-Vector parallel, 30GB savings, 8× speedup
- [GLM-5技术报告全解读 (53AI)](https://www.53ai.com/news/OpenSourceLLM/2026022239486.html) — MLAPO, quantization details
- [北京智谱GLM-5大模型技术细节剖析大全](https://www.cnblogs.com/Yanjy-OnlyOne/p/19633185) — Cube/Vector pipeline, performance numbers
- [GLM-5从擅长编码进化到复杂系统工程](https://cloud.tencent.com/developer/article/2637114) — Technical overview

## DeepSeek-V3.2 Ascend Open-Source (Same Kernels as GLM-5)

- [华为昇腾0Day支持DeepSeek-V3.2-Exp](https://www.ithome.com/0/886/722.htm) — **Key source**: Lightning Indexer + Sparse Flash Attention open-sourced, AscendC code
- [DeepSeek-V3.2-Exp (GitHub)](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) — Official DeepSeek model repo
- [vLLM-Ascend: DeepSeek V3.2 support PR](https://github.com/vllm-project/vllm-ascend/pull/3270) — MLAPO, PCP, DCP, ACLGraph integration
- [vLLM-Ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html) — Sparse Flash Attention backend, MLAPO
- [vLLM-Ascend GLM-5 Tutorial](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.html) — GLM-5 deployment on Ascend via vLLM

## Moore Threads MTT S5000

- [摩尔线程GLM-5 Day-0适配 (IT之家)](https://www.ithome.com/0/921/268.htm) — MTT S5000 specs, MUSA, FP8, ACE, SGLang-MUSA
- [MTT S5000 Day-0适配 (CSDN)](https://blog.csdn.net/csdnnews/article/details/157992844) — Detailed specs
- [MTT S5000产品页](https://www.mthreads.com/product/S5000) — Official product page
- [MUSA SDK](https://www.mthreads.com/product/MUSASDK) — Programming toolkit
- [硅基流动+摩尔线程: Prefill 4000, Decode 1000](https://zhuanlan.zhihu.com/p/1986717430689596001) — Performance benchmarks on S5000

## Baidu Kunlun XPU

- [百度百舸Day0完成昆仑芯GLM-5适配](https://zhuanlan.zhihu.com/p/2005227027679183663) — DSA+MoE adapted, INT8, MTP, vLLM-Kunlun
- [vLLM-Kunlun (GitHub baidu/vLLM-Kunlun)](https://github.com/baidu/vLLM-Kunlun) — Open-source Kunlun XPU vLLM plugin
- [百度百舸GLM-5适配 (PingWest)](https://www.pingwest.com/a/311502) — Full-stack Day-0 adaptation
- [Baidu Kunlun HotChips 2020 Presentation](https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Inference_Baidu_Kunlun_v5.pdf) — Architecture slides

## Cambricon MLU

- [寒武纪MLU架构 (CSDN)](https://blog.csdn.net/djfjkj52/article/details/134411851) — MLUarch03, BANG language
- [寒武纪BANG语言发布](https://www.cambricon.com/index.php?m=content&c=index&a=show&catid=127&id=29) — BANG programming model
- [寒武纪AI芯片全解析](https://zhuanlan.zhihu.com/p/1941064011023119190) — MLU370 specs, chiplet design
- [思元370系列产品](https://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=360) — MLU370 product page
- [GLM-4.6适配寒武纪 (OSChina)](https://www.oschina.net/news/375459) — FP8+INT4 mixed quantization

## Hygon DCU

- [海光K100 AI规格 (CSDN)](https://blog.csdn.net/weixin_43737299/article/details/139154744) — FP32 49T, BF16 192T, INT8 392T, 64GB, 896 GB/s
- [国产海光DCU深度解析 (CSDN)](https://blog.csdn.net/thesky123456/article/details/147719977) — Architecture, ROCm stack
- [海光DCU部署攻略 (CSDN)](https://blog.csdn.net/liu1983robin/article/details/144493349) — Deployment guide

## MetaX C500

- [沐曦曦云C500 (Gitee 模力方舟)](https://ai.gitee.com/docs/compute/clusters_gpu/mx_gpu) — FP16 280T, 64GB HBM2e, 1.8TB/s
- [沐曦C500发布: 国产GPU千亿参数训推一体机](https://www.metax-tech.com/ndetail/12487.html) — Product announcement
- [曦云C500产品页](https://www.metax-tech.com/prod.html?cid=107&id=21) — Official specs
- [沐曦MXC500推理性能压测](https://wangjunjian.com/mxc500/benchmark/2025/02/13/Performance-Stress-Testing-of-the-MuXin-MXC500-for-Large-Model-Inference.html) — Benchmark results

## Enflame GCU

- [燧原S60 (Gitee 模力方舟)](https://ai.gitee.com/docs/compute/clusters_gpu/ef_gpu) — Product specs
- [燧原S60产品手册](https://support.enflame-tech.com/onlinedoc_hw/5-s6x/S60/product_manual/content/source/S60_product_manual.html) — Hardware documentation
- [TopsRider软件栈](https://support.enflame-tech.com/onlinedoc_dev_3.4/2-install/sw_install/content/source/installation.html) — Software installation
- [燧原软件栈白皮书](https://support.enflame-tech.com/onlinedoc_dev_2.5.115/1-introduce/sw_intro/content/source/index.html) — TopsATen, TopsCC, TopsRider overview
- [云燧i20推理加速卡](https://www.enflame-tech.com/cloudblazer-i2x/i20) — Suisi 2.5, 32 TFLOPS FP32, HBM2e

## Chinese Tech Journalism (GLM-5 Analysis)

- [智谱GLM-5技术全公开 (QbitAI)](https://www.qbitai.com/2026/02/381712.html) — Comprehensive technical breakdown
- [智谱GLM-5技术全公开 (BAAI)](https://hub.baai.ac.cn/view/52692) — BAAI mirror of QbitAI article
- [GLM-5技术报告全解读 (知乎)](https://zhuanlan.zhihu.com/p/2010315149228147952) — Zhihu deep analysis
- [GLM-5行业技术报告深度解析 (CSDN)](https://blog.csdn.net/luomao2012/article/details/158345068) — CSDN technical breakdown
- [股价暴涨32%! GLM-5登顶全球开源第一 (BAAI)](https://hub.baai.ac.cn/view/52533) — Market impact
- [GLM-5封神: 智谱市值五天翻倍](https://wallstreetcn.com/articles/3765661) — Business analysis
- [China's GLM-5 Rivals GPT-5.2 on Zero Nvidia Silicon](https://awesomeagents.ai/news/glm-5-china-frontier-model-huawei-chips/) — English analysis
- [GLM-5: The World's Strongest Open-Source LLM Trained on Chinese Chips](https://www.trendingtopics.eu/glm-5-the-worlds-strongest-open-source-llm-solely-trained-on-chinese-huawei-chips/) — English overview

## Kernel Source Code Repositories (Deep Dive Research)

### NVIDIA H100 — DeepGEMM fp8_mqa_logits CUDA Kernel
- [DeepGEMM sm90_fp8_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh) — **Complete CUDA kernel**: WGMMA, TMA, warp specialization, shared memory layout, barrier synchronization, persistent scheduling
- [DeepGEMM sm100_fp8_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh) — Blackwell variant
- [DeepGEMM smxx_fp8_mqa_logits.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/jit_kernels/impls/smxx_fp8_mqa_logits.hpp) — JIT dispatch (block_q = 128/num_heads, block_kv=256)
- [DeepGEMM sm90_fp8_paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh) — Paged variant with block_table
- [DeepGEMM test_attention.py](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py) — Reference implementation and test cases

### Huawei Ascend 910B — AscendC Lightning Indexer Kernel
- [vllm-ascend lightning_indexer_service_cube.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_kernel/lightning_indexer_service_cube.h) — **Cube service**: L0A/L0B/L0C buffers, triple-buffered K, ND→NZ format, FixP ReLU, event-based pipeline
- [vllm-ascend lightning_indexer_service_vector.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_kernel/lightning_indexer_service_vector.h) — **Vector service**: VBS32 sort, VMS4v2 merge, BASE_TOPK=2048, running topk state
- [vllm-ascend lightning_indexer_common.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_kernel/lightning_indexer_common.h) — Shared types
- [vllm-ascend lightning_indexer_kernel.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_kernel/lightning_indexer_kernel.h) — Orchestration
- [vllm-ascend lightning_indexer_vector.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_kernel/lightning_indexer_vector.h) — Low-level vector ops
- [vllm-ascend lightning_indexer_vllm_tiling.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/lightning_indexer_vllm/op_host/lightning_indexer_vllm_tiling.h) — Tiling parameters
- [vllm-ascend lightning_indexer_quant/](https://github.com/vllm-project/vllm-ascend/tree/main/csrc/lightning_indexer_quant) — Quantized variant (separate Cube/Vector service files)

### Huawei Ascend 910B — AscendC Sparse Flash Attention Kernel
- [vllm-ascend sparse_flash_attention_common.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_common.h) — SFA_LAYOUT enum, SFAType template, SoftmaxConfig
- [vllm-ascend sparse_flash_attention_service_cube_mla.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_service_cube_mla.h) — Cube: QK^T and PV matmuls for MLA
- [vllm-ascend sparse_flash_attention_service_vector_mla.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_service_vector_mla.h) — Vector: online softmax for MLA
- [vllm-ascend sparse_flash_attention_kernel_mla.h](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_kernel_mla.h) — MLA-specific orchestration

### AMD ROCm — AITER Triton fp8_mqa_logits Kernel
- [AITER fp8_mqa_logits.py (kernel)](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py) — **Complete Triton kernel**: `tl.dot` for scoring, `tl.maximum` for ReLU, `tl.sum` for head reduction, variable-length via cu_start/cu_end
- [AITER fp8_mqa_logits.py (wrapper)](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/attention/fp8_mqa_logits.py) — Python wrapper with autotuning
- [AITER pa_mqa_logits.py](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/attention/pa_mqa_logits.py) — Paged attention variant
- [AITER bench_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/main/op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py) — Benchmarking harness
- [Accelerate DeepSeek-R1 Inference: Integrate AITER into SGLang (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html) — Integration guide

### TileLang — Portable Training Kernel
- [tilelang-dsa kernel_bf16_training_dsa_warmup_lightning_indexer.py](https://github.com/lemyx/tilelang-dsa/blob/main/kernel_bf16_training_dsa_warmup_lightning_indexer.py) — **Complete training kernel**: BF16, fwd+bwd, KL-divergence loss, TileLang JIT, block_M/block_N autotuning
- [fp8_lighting_indexer.py analysis (Gist)](https://gist.github.com/createthis/0cce8a250daa3a117cb2986c743c02f2) — FP8 scoring analysis with TileLang

### SGLang — DSA Integration Layer
- [SGLang nsa_indexer.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa/nsa_indexer.py) — Dispatch layer: calls DeepGEMM (NVIDIA) or AITER (AMD), handles paged/ragged layouts, chunked OOM handling
- [SGLang triton_kernel.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa/triton_kernel.py) — FP8 activation quantization, KV index extraction Triton kernels
- [SGLang nsa_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa_backend.py) — Sparse attention backend orchestration

### Ascend Architecture Papers
- [Parallel Scan on Ascend (arxiv 2505.15112)](https://arxiv.org/html/2505.15112v1) — 20 Cube + 40 Vector units, 2:1 ratio, 800 GB/s, Cube-Vector split, queue API
- [AscendCraft: Automatic NPU Kernel Generation](https://arxiv.org/html/2601.22760v1) — CopyIn/Compute/CopyOut pipeline
- [FastAttention: FlashAttention2 on NPUs](https://openreview.net/pdf?id=76NYyOrnfk) — Two-level tiling, Cube-Vector pipeline for attention
- [Sparse Matrix-Matrix on Ascend AI Core](https://accml.dcs.gla.ac.uk/papers/2023/5th_AccML_paper_10.pdf) — Cube 16×16 FP16, Vector 2048-bit SIMD

### vllm-ascend Integration
- [vllm-ascend release notes](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/release_notes.md) — MLAPO env var, SFA backend, Lightning Indexer, PCP/DCP
- [vllm-ascend PR #3260](https://github.com/vllm-project/vllm-ascend/pull/3260) — PCP/DCP with MLAPO integration
- [vllm-ascend DeepSeek-V3.2 tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html) — Deployment guide
- [vLLM blog: DeepSeek-V3.2 Sparse Attention](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html) — Architecture overview
- [Leon Ericsson DSA explainer](https://leonericsson.github.io/blog/2025-10-16-dsa) — Clear algorithm explanation

## FlashMLA and DeepGEMM Kernel Analysis (Prior Research)

See: `/home/lily/.claude/plans/scalable-cuddling-metcalfe.md` — Contains 42+ URLs consulted for the FlashMLA/DeepGEMM kernel selection, including:
- FlashMLA repo analysis (12 URLs)
- DeepGEMM repo analysis (6 URLs)
- FlashInfer repo analysis (9 URLs)
- TensorRT-LLM analysis (1 URL)
- SGLang analysis (7 URLs)
- xLLM analysis (4 URLs)
