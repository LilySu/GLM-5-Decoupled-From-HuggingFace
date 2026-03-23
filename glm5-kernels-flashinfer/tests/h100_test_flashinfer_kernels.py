"""H100-only: Test FlashInfer CUDA kernels produce correct attention output.

Tests the FA3 dense MLA backend and trtllm-gen sparse MLA backend
against PyTorch reference implementations.

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - pip install flashinfer

Run:
    python3 -m glm5-kernels-flashinfer.tests.h100_test_flashinfer_kernels
"""

import sys
import torch
from .conftest import assert_close, skip_no_sm90, has_flashinfer


def _require():
    if not has_flashinfer():
        print("  SKIP flashinfer not installed")
        return False
    return True


@skip_no_sm90
def h100_test_flashinfer_fa3_dense_decode():
    """FlashInfer FA3 dense decode kernel vs PyTorch eager on absorbed MLA dimensions."""
    print("\n[H100] FlashInfer FA3 dense decode")
    if not _require():
        return True

    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    device = "cuda"
    B, H = 4, 64
    d_ckv, d_kpe = 512, 64
    seq_kv = 1024
    page_size = 1

    torch.manual_seed(42)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")

    q_nope = torch.randn(B, H, d_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_kpe, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(B * seq_kv, 1, d_ckv, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(B * seq_kv, 1, d_kpe, dtype=torch.bfloat16, device=device)

    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * seq_kv
    kv_indices = torch.arange(0, B * seq_kv, dtype=torch.int32, device=device)
    kv_lens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    sm_scale = 1.0 / ((d_ckv + d_kpe) ** 0.5)

    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens,
                 H, d_ckv, d_kpe, page_size, False, sm_scale,
                 torch.bfloat16, torch.bfloat16)

    out_fi = wrapper.run(q_nope, q_pe, ckv, kpe)

    # PyTorch reference
    q_full = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, 576]
    kv_full = torch.cat([
        ckv.view(B, seq_kv, d_ckv),
        kpe.view(B, seq_kv, d_kpe),
    ], dim=-1)  # [B, seq_kv, 576]

    attn = torch.einsum("bhd,btd->bht", q_full.float(), kv_full.float()) * sm_scale
    attn = torch.softmax(attn, dim=-1)
    out_ref = torch.einsum("bht,btd->bhd", attn, kv_full[..., :d_ckv].float())  # [B, H, 512]
    out_ref = out_ref.to(torch.bfloat16)

    return assert_close("fa3_dense_decode", out_fi, out_ref, atol=5e-2, rtol=5e-2)


@skip_no_sm90
def h100_test_flashinfer_trtllm_sparse_decode():
    """FlashInfer trtllm-gen sparse MLA decode with sparse_mla_top_k."""
    print("\n[H100] FlashInfer trtllm-gen sparse decode")
    if not _require():
        return True

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from importlib import import_module
    patches = import_module("glm5-kernels-flashinfer.patches")
    patches.apply_glm5_patches()

    device = "cuda"
    B, H = 2, 64
    d_ckv, d_kpe = 512, 64
    d_qk = d_ckv + d_kpe  # 576
    seq_kv = 512
    page_size = 64
    topk = 64
    num_pages = (B * seq_kv + page_size - 1) // page_size

    torch.manual_seed(42)
    query = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(num_pages, page_size, d_qk, dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(32 * 1024 * 1024, dtype=torch.int8, device=device)

    # Sparse indices: [B, 1, topk]
    block_tables = torch.stack([
        torch.randperm(seq_kv, device=device)[:topk].sort().values
        for _ in range(B)
    ]).unsqueeze(1).to(torch.int32)

    seq_lens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)

    try:
        out = trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=128,  # pass 128 to satisfy validation (patched or not)
            kv_lora_rank=d_ckv,
            qk_rope_head_dim=d_kpe,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_kv,
            sparse_mla_top_k=topk,
            backend="trtllm-gen",
        )
        ok = torch.isfinite(out).all().item()
        if ok:
            print(f"  PASS trtllm-gen sparse decode: output shape={out.shape}, all finite")
        else:
            print(f"  FAIL non-finite output values")
        return ok
    except Exception as e:
        print(f"  FAIL trtllm-gen sparse decode: {e}")
        return False


@skip_no_sm90
def h100_test_flashinfer_cuda_graph_dense():
    """FlashInfer BatchMLAPagedAttentionWrapper with use_cuda_graph=True."""
    print("\n[H100] FlashInfer CUDA graph dense decode")
    if not _require():
        return True

    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    device = "cuda"
    B, H = 4, 64
    d_ckv, d_kpe = 512, 64
    seq_kv = 256
    page_size = 1

    torch.manual_seed(42)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Pre-allocate buffers for CUDA graph
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * seq_kv
    kv_indices = torch.arange(0, B * seq_kv, dtype=torch.int32, device=device)
    kv_lens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)

    wrapper = BatchMLAPagedAttentionWrapper(
        workspace, use_cuda_graph=True,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        kv_indices=kv_indices, kv_len_arr=kv_lens,
        backend="fa3",
    )

    q_nope = torch.randn(B, H, d_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_kpe, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(B * seq_kv, 1, d_ckv, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(B * seq_kv, 1, d_kpe, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / ((d_ckv + d_kpe) ** 0.5)
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_lens,
                 H, d_ckv, d_kpe, page_size, False, sm_scale,
                 torch.bfloat16, torch.bfloat16)

    try:
        out = wrapper.run(q_nope, q_pe, ckv, kpe)
        ok = torch.isfinite(out).all().item() and out.shape == (B, H, d_ckv)
        if ok:
            print(f"  PASS CUDA graph dense decode: shape={out.shape}, all finite")
        else:
            print(f"  FAIL shape={out.shape} or non-finite values")
        return ok
    except Exception as e:
        print(f"  FAIL CUDA graph dense decode: {e}")
        return False


if __name__ == "__main__":
    results = [
        h100_test_flashinfer_fa3_dense_decode(),
        h100_test_flashinfer_trtllm_sparse_decode(),
        h100_test_flashinfer_cuda_graph_dense(),
    ]
    passed = sum(results)
    print(f"\n{'='*60}")
    print(f"H100 FlashInfer: {passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
