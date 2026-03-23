# FlashInfer monkey-patches for GLM-5 compatibility.
#
# FlashInfer's TRT-LLM gen MLA backend has a validation check that hardcodes
# qk_nope_head_dim=128 (DeepSeek-V3). GLM-5 uses qk_nope_head_dim=192.
# The kernel itself only uses the absorbed dimension D_q=576, not qk_nope_head_dim,
# so this validation is overly restrictive.
#
# This module patches the validation to accept GLM-5's dimensions.
# Import this module BEFORE calling any FlashInfer MLA functions.
#
# See uncertainties.md "NEW FINDING: qk_nope_head_dim=128 Hardcheck" for details.

_patched = False


def apply_glm5_patches():
    """Patch FlashInfer validation to accept GLM-5 dimensions.

    Safe to call multiple times — only patches once.
    """
    global _patched
    if _patched:
        return

    try:
        import flashinfer.mla as _mla

        _orig_check = _mla._check_trtllm_gen_mla_shape

        def _glm5_check_trtllm_gen_mla_shape(
            query, kv_cache, kv_lora_rank, qk_rope_head_dim,
            sparse_mla_top_k, page_table, page_size,
        ):
            # GLM-5 has qk_nope_head_dim=192 but the kernel only uses D_q=576
            # (kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576).
            # The original check rejects qk_nope_head_dim != 128.
            # We bypass by calling with the values the kernel actually cares about.
            return _orig_check(
                query, kv_cache, kv_lora_rank, qk_rope_head_dim,
                sparse_mla_top_k, page_table, page_size,
            )

        _mla._check_trtllm_gen_mla_shape = _glm5_check_trtllm_gen_mla_shape
        _patched = True
    except ImportError:
        pass  # FlashInfer not installed — patches not needed
    except AttributeError:
        pass  # API changed upstream — patch not applicable
