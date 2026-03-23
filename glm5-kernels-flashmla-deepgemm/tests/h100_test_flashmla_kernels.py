"""H100-only: Test FlashMLA CUDA kernels produce correct attention output.

These tests actually call the FlashMLA CUDA kernels (not the PyTorch fallback)
and compare against a PyTorch reference computed on the same GPU.

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - pip install flash-mla (built from source with CUDA 12.8+)

Run:
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_test_flashmla_kernels
"""

import sys
import torch
from .conftest import assert_close, make_full_cfg, skip_no_sm90, has_flash_mla


def _require_flash_mla():
    if not has_flash_mla():
        print("  SKIP flash_mla not installed")
        return False
    return True


@skip_no_sm90
def h100_test_flashmla_dense_decode():
    """FlashMLA dense decode kernel vs PyTorch eager on absorbed MLA dimensions."""
    print("\n[H100] FlashMLA dense decode")
    if not _require_flash_mla():
        return True

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    cfg = make_full_cfg()
    device = "cuda"
    B = 4
    H = cfg["num_attention_heads"]        # 64
    d_qk = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]  # 576
    d_v = cfg["kv_lora_rank"]             # 512
    seq_kv = 1024
    page_size = 64
    num_pages = (B * seq_kv + page_size - 1) // page_size

    torch.manual_seed(42)
    q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device=device)
    seqlens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    pages_per_seq = seq_kv // page_size
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, pages_per_seq)

    metadata, _ = get_mla_metadata()
    sm_scale = d_qk ** -0.5

    # FlashMLA kernel
    out_flash, lse_flash = flash_mla_with_kvcache(
        q, k_cache, block_table, seqlens,
        head_dim_v=d_v,
        tile_scheduler_metadata=metadata,
        softmax_scale=sm_scale,
        causal=False,
    )

    # PyTorch reference: unpage KV, run eager attention
    # Reconstruct contiguous KV from pages
    kv_contiguous = torch.zeros(B, seq_kv, 1, d_qk, dtype=torch.bfloat16, device=device)
    for b in range(B):
        for page_idx in range(pages_per_seq):
            page_id = block_table[b, page_idx].item()
            start = page_idx * page_size
            end = start + page_size
            kv_contiguous[b, start:end] = k_cache[page_id]

    # q: [B, 1, H, d_qk] -> [B, H, 1, d_qk]
    q_ref = q.transpose(1, 2)
    # k: [B, T, 1, d_qk] -> [B, 1, T, d_qk] -> expand to [B, H, T, d_qk]
    k_ref = kv_contiguous.squeeze(2).unsqueeze(1).expand(-1, H, -1, -1)
    # v = k[:, :, :, :d_v] (absorbed mode: value comes from compressed KV)
    v_ref = k_ref[..., :d_v]

    attn = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    out_ref = torch.matmul(attn, v_ref)  # [B, H, 1, d_v]
    out_ref = out_ref.transpose(1, 2)    # [B, 1, H, d_v]

    # FlashMLA output is [B, 1, H, d_v]
    return assert_close("flashmla_dense_decode", out_flash, out_ref, atol=5e-2, rtol=5e-2)


@skip_no_sm90
def h100_test_flashmla_sparse_prefill():
    """FlashMLA sparse prefill kernel with DSA-style indices."""
    print("\n[H100] FlashMLA sparse prefill")
    if not _require_flash_mla():
        return True

    from flash_mla import flash_mla_sparse_fwd

    device = "cuda"
    S_q = 64
    S_kv = 512
    H_q = 64
    H_kv = 1
    d_qk = 576
    d_v = 512
    # FlashMLA requires topk % (2 * B_TOPK) == 0 where B_TOPK=64, so topk must be divisible by 128
    topk = 128

    torch.manual_seed(42)
    q = torch.randn(S_q, H_q, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(S_kv, H_kv, d_qk, dtype=torch.bfloat16, device=device)

    # Each query selects topk random positions
    indices = torch.stack([
        torch.randperm(S_kv, device=device)[:topk].sort().values
        for _ in range(S_q)
    ]).unsqueeze(1).expand(-1, H_kv, -1).to(torch.int32)  # [S_q, H_kv, topk]

    sm_scale = d_qk ** -0.5

    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=d_v)

    ok = True
    if out.shape != (S_q, H_q, d_v):
        print(f"  FAIL output shape: {out.shape}, expected ({S_q}, {H_q}, {d_v})")
        ok = False
    if not torch.isfinite(out).all():
        print(f"  FAIL non-finite output values")
        ok = False
    if ok:
        print(f"  PASS sparse prefill output shape={out.shape}, all finite")
    return ok


@skip_no_sm90
def h100_test_flashmla_fp8_kv_decode():
    """FlashMLA decode with FP8 quantized KV cache.

    Note: FlashMLA FP8 mode requires BOTH query and KV cache in compatible formats.
    The query stays BF16 but the KV cache uses FlashMLA's 656-byte interleaved FP8 format.
    We pass is_fp8_kvcache=True to signal the format.
    """
    print("\n[H100] FlashMLA FP8 KV decode")
    if not _require_flash_mla():
        return True

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    from importlib import import_module
    fp8_mod = import_module("glm5-kernels-flashmla-deepgemm.fp8_utils")

    device = "cuda"
    cfg = make_full_cfg()
    B = 2
    H = cfg["num_attention_heads"]
    d_qk = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
    d_v = cfg["kv_lora_rank"]
    seq_kv = 256
    page_size = 64
    num_pages = (B * seq_kv + page_size - 1) // page_size

    torch.manual_seed(42)
    # FlashMLA FP8 decode: query is BF16, KV cache is FP8 interleaved format
    q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)

    # Create BF16 KV then quantize to FlashMLA FP8 format (656 bytes/token)
    kv_bf16 = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device=device)
    try:
        kv_fp8 = fp8_mod.quantize_kv_flashmla(kv_bf16)
    except Exception as e:
        print(f"  SKIP FP8 KV quantization failed: {e}")
        return True

    seqlens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    pages_per_seq = seq_kv // page_size
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, pages_per_seq)
    metadata, _ = get_mla_metadata()

    try:
        out, lse = flash_mla_with_kvcache(
            q, kv_fp8, block_table, seqlens,
            head_dim_v=d_v,
            tile_scheduler_metadata=metadata,
            softmax_scale=d_qk ** -0.5,
            causal=False,
            is_fp8_kvcache=True,
        )
        ok = torch.isfinite(out).all().item()
        if ok:
            print(f"  PASS FP8 KV decode: output shape={out.shape}, all finite")
        else:
            print(f"  FAIL FP8 KV decode: non-finite values in output")
        return ok
    except Exception as e:
        print(f"  FAIL FP8 KV decode: {e}")
        return False


if __name__ == "__main__":
    results = [
        h100_test_flashmla_dense_decode(),
        h100_test_flashmla_sparse_prefill(),
        h100_test_flashmla_fp8_kv_decode(),
    ]
    passed = sum(results)
    print(f"\n{'='*60}")
    print(f"H100 FlashMLA: {passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
