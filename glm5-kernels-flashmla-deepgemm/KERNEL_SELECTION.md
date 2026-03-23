# GLM-5 Kernel Selection — FlashMLA + DeepGEMM Path (H100 SM90)

## Final Component → Kernel Mapping

| Component | Source | Kernel | Status |
|-----------|--------|--------|--------|
| **MLA Attention** | FlashMLA | `flash_mla_with_kvcache()`, `flash_mla_sparse_fwd()` | **NEW** — replaces PyTorch |
| **DSA Indexer** | DeepGEMM | `fp8_mqa_logits()`, `fp8_paged_mqa_logits()` | **NEW** — replaces PyTorch |
| **DSA Sparse Attn** | FlashMLA | Sparse mode of MLA kernels (indices tensor) | **NEW** — replaces PyTorch |
| **MoE Grouped GEMM** | DeepGEMM | `m_grouped_fp8_gemm_nt_contiguous/masked()` | **UPGRADE** — FP8 replaces Triton BF16 |
| **MoE Router** | Pure PyTorch | sigmoid+bias+topk (n_group=1) | **SIMPLIFIED** — no kernel needed |
| **RMSNorm** | Unsloth Triton | Keep existing | **KEEP** |
| **SwiGLU** | Unsloth Triton | Keep existing | **KEEP** |
| **Partial RoPE** | PyTorch | Keep existing | **KEEP** |
| **CE Loss** | Unsloth Triton | Keep existing | **KEEP** |
| **FP8 Quant** | Utility functions | PyTorch quantize/dequant helpers | **NEW** utility |
| **Paged KV Cache** | FlashMLA-compat | Python cache manager + FlashMLA format | **NEW** cache impl |
| **MTP** | vLLM/TRT-LLM | MTP backend (Phase 2) | **PHASE 2** |

## Dependencies (only 2 external repos)

```bash
# Build from source (requires CUDA 12.8+ and SM90 GPU):
git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA && cd FlashMLA && pip install -v .
git clone https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && pip install -v .

# JIT compilation mitigations:
export DG_JIT_USE_NVRTC=1               # 10x faster JIT during development
export DG_JIT_CACHE_DIR=/path/to/cache   # Persist cache across restarts
```

## Key Design Decision: n_group=1

GLM-5 uses `n_group=1, topk_group=1`, meaning group-level routing is a no-op.
This is DIFFERENT from DeepSeek-V3 (`n_group=8, topk_group=4`).

The routing simplifies to flat sigmoid + top-8:
```python
scores = sigmoid(linear(x)) + bias       # [batch, 256]
topk_w, topk_i = torch.topk(scores, k=8) # [batch, 8]
topk_w = topk_w / topk_w.sum(-1, True) * 2.5  # normalize + scale
```

No custom CUDA kernel needed for routing. If the full 744B model uses n_group>1,
install `sgl-kernel` and use `moe_fused_gate` instead.

## Full analysis

See `/home/lily/.claude/plans/scalable-cuddling-metcalfe.md` for the complete
component-by-component analysis with:
- 42+ URLs consulted
- Deep dives on MLA, DSA, and MoE kernels
- Evaluation rubric (8 weighted criteria)
- 16 uncertainties tracked (8 resolved, 5 opinionated, 3 unverified)
- Cross-reference with `kernel_decisions.md`
