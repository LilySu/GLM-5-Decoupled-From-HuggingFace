# Fused LoRA forward/backward for SwiGLU MLP, QKV, and single-projection.
# Extracted from unsloth/kernels/fast_lora.py for standalone use.
#
# Original copyright:
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
from .unsloth_utils import (
    _maybe_fake_quantize_activations,
    fast_dequantize,
    QUANT_STATE,
    get_lora_parameters,
    get_lora_parameters_bias,
    matmul_lora,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW, upW_quant, upA, upB, upS,
        downW, downW_quant, downA, downB, downS,
        _forward_function,
        _backward_function,
        inplace = True,
    ):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype), gateB.to(dtype),
            upA.to(dtype), upB.to(dtype),
            downA.to(dtype), downB.to(dtype),
        )
        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(), gateB.t(), upA.t(), upB.t(), downA.t(), downB.t(),
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)

        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
        del upW
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW.t())
        del gateW
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        return (
            dX.view(batch, seq_len, hd),
            None, None, d_gateA.t(), d_gateB.t(), None,
            None, None, d_upA.t(), d_upB.t(), None,
            None, None, d_downA.t(), d_downB.t(), None,
            None, None, None,
        )


from .unsloth_swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel


def apply_lora_mlp_swiglu(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.gate_proj)
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW, upW_quant, upA, upB, upS,
        downW, downW_quant, downA, downB, downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
    )
    return out


class LoRA_QKV(torch.autograd.Function):
    """
    Fused LoRA forward/backward for Q, K, V projections.
    Q = X @ Wq + X @ Aq @ Bq
    K = X @ Wk + X @ Ak @ Bk
    V = X @ Wv + X @ Av @ Bv
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
        inplace = True,
    ):
        dtype = X.dtype
        orig_shape = X.shape
        X_for_matmul = X
        if X.dim() == 3:
            X_for_matmul = X.view(-1, X.shape[-1])
        Q = matmul_lora(X_for_matmul, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X_for_matmul, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X_for_matmul, VW, VW_quant, VA, VB, VS)

        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], -1)
            K = K.view(orig_shape[0], orig_shape[1], -1)
            V = V.view(orig_shape[0], orig_shape[1], -1)

        ctx.custom_saved_tensors = (
            QW, QW_quant, QS,
            KW, KW_quant, KS,
            VW, VW_quant, VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB)
        ctx.inplace = inplace
        return Q, K, V

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = (
            QA.to(dtype), QB.to(dtype),
            KA.to(dtype), KB.to(dtype),
            VA.to(dtype), VB.to(dtype),
        )
        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

        d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

        d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)
        del QW
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha = QS)

        KW = fast_dequantize(KW.t(), KW_quant)
        dX.addmm_(dK, KW.t())
        del KW
        dX.addmm_(dK @ KB.t(), KA.t(), alpha = KS)

        VW = fast_dequantize(VW.t(), VW_quant)
        dX.addmm_(dV, VW.t())
        del VW
        dX.addmm_(dV @ VB.t(), VA.t(), alpha = VS)

        return (
            dX.view(batch, seq_len, hd),
            None, None, d_QA.t(), d_QB.t(), None,
            None, None, d_KA.t(), d_KB.t(), None,
            None, None, d_VA.t(), d_VB.t(), None,
            None,
        )


def apply_lora_qkv(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.q_proj)
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
        inplace,
    )
    return Q, K, V


class LoRA_W(torch.autograd.Function):
    """Fused LoRA forward/backward for a single projection (e.g. O-proj)."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X: torch.Tensor, W, W_quant, A, B, S):
        dtype = X.dtype
        XW = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (W, W_quant, S)
        ctx.save_for_backward(A, B, X)
        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        A, B = A.to(dtype), B.to(dtype)
        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)

        d_A.addmm_(X.t(), dY @ B.t(), alpha = S, beta = 0)
        d_B.addmm_(A.t() @ X.t(), dY, alpha = S, beta = 0)

        W = fast_dequantize(W.t(), W_quant)
        dX = dY @ W.t()
        del W
        dX.addmm_(dY @ B.t(), A.t(), alpha = S)

        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None


def apply_lora_o(self, X):
    X = _maybe_fake_quantize_activations(X, self.o_proj)
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)
    return O
