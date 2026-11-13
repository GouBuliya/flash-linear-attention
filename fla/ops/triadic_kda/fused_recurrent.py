# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'HAS_A': lambda args: args['A_log'] is not None,
    'HAS_DT_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_A2': lambda args: args['A2_log'] is not None,
    'HAS_DT2_BIAS': lambda args: args['dt2_bias'] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_triadic_kda_fwd_kernel(
    q,
    k,
    v,
    k2,
    q2,
    gk,
    g2,
    beta,
    A_log,
    dt_bias,
    A2_log,
    dt2_bias,
    lower_bound,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    A_PER_CHANNEL: tl.constexpr,
    A2_PER_SLICE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_A2: tl.constexpr,
    HAS_DT2_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_e = tl.arange(0, BE).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)

    mask_e = o_e < E
    mask_k = o_k < K
    mask_v = o_v < V
    off_ek = tl.reshape(o_e[:, None] * K + o_k[None, :], [BE * BK])
    mask_ek = tl.reshape(mask_e[:, None] & mask_k[None, :], [BE * BK])
    mask_h = mask_ek[:, None] & mask_v[None, :]

    b_h = tl.zeros([BE * BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * E * K * V + off_ek[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    if USE_GATE_IN_KERNEL:
        if HAS_A:
            if A_PER_CHANNEL:
                b_Ak = exp(tl.load(A_log + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32))
            else:
                b_Ak = exp(tl.load(A_log + i_h).to(tl.float32))
        else:
            b_Ak = 1.0
        if HAS_DT_BIAS:
            b_dtk = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)

        if HAS_A2:
            if A2_PER_SLICE:
                b_Ae = exp(tl.load(A2_log + i_h * E + o_e, mask=mask_e, other=0.0).to(tl.float32))
            else:
                b_Ae = exp(tl.load(A2_log + i_h).to(tl.float32))
        else:
            b_Ae = 1.0
        if HAS_DT2_BIAS:
            b_dte = tl.load(dt2_bias + i_h * E + o_e, mask=mask_e, other=0.0).to(tl.float32)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_k2 = k2 + (bos * H + i_h) * E + o_e
    p_q2 = q2 + (bos * H + i_h) * E + o_e
    p_gk = gk + (bos * H + i_h) * K + o_k
    p_g2 = g2 + (bos * H + i_h) * E + o_e
    p_beta = beta + bos * H + i_h
    p_o = o + (bos * H + i_h) * V + o_v

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_k2 = tl.load(p_k2, mask=mask_e, other=0.0).to(tl.float32)
        b_q2 = tl.load(p_q2, mask=mask_e, other=0.0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=mask_k, other=0.0).to(tl.float32)
        b_g2 = tl.load(p_g2, mask=mask_e, other=0.0).to(tl.float32)

        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                b_gk = b_gk + b_dtk
            if HAS_DT2_BIAS:
                b_g2 = b_g2 + b_dte
            if USE_LOWER_BOUND:
                b_gk = tl.where(mask_k, lower_bound * tl.sigmoid(b_Ak * b_gk), 0.0)
                b_g2 = tl.where(mask_e, lower_bound * tl.sigmoid(b_Ae * b_g2), 0.0)
            else:
                b_gk = tl.where(mask_k, -b_Ak * softplus(b_gk), 0.0)
                b_g2 = tl.where(mask_e, -b_Ae * softplus(b_g2), 0.0)

        b_beta = tl.load(p_beta).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2.0

        b_decay = tl.reshape(exp(b_g2[:, None] + b_gk[None, :]), [BE * BK])
        b_kappa = tl.reshape(b_k2[:, None] * b_k[None, :], [BE * BK])
        b_rho = tl.reshape(b_q2[:, None] * b_q[None, :], [BE * BK])

        b_h *= b_decay[:, None]
        b_u = b_beta * (b_v - tl.sum(b_h * b_kappa[:, None], 0))
        b_h += b_kappa[:, None] * b_u[None, :]
        b_o = tl.sum(b_h * b_rho[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_k2 += H * E
        p_q2 += H * E
        p_gk += H * K
        p_g2 += H * E
        p_beta += H
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * E * K * V + off_ek[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_triadic_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    gk: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    A2_log: torch.Tensor | None = None,
    dt2_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = k.shape
    V, E = v.shape[-1], k2.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BE = triton.next_power_of_2(E)
    BK = triton.next_power_of_2(K)
    BV = max(1, min(8, triton.next_power_of_2(V), 4096 // (BE * BK)))
    NV = triton.cdiv(V, BV)
    state_elems = BE * BK * BV
    num_warps = 4 if state_elems >= 1024 else (2 if state_elems >= 256 else 1)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, H, E, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NV * N * H,)
    fused_recurrent_triadic_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        A2_log=A2_log,
        dt2_bias=dt2_bias,
        lower_bound=lower_bound,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BK=BK,
        BV=BV,
        BE=BE,
        A_PER_CHANNEL=A_log is not None and A_log.numel() == H * K,
        A2_PER_SLICE=A2_log is not None and A2_log.numel() == H * E,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        num_warps=num_warps,
        num_stages=2,
    )
    return o, final_state


class FusedRecurrentTriadicKDAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k2: torch.Tensor,
        q2: torch.Tensor,
        gk: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        A2_log: torch.Tensor | None = None,
        dt2_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        return fused_recurrent_triadic_kda_fwd(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            A2_log=A2_log,
            dt2_bias=dt2_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            cu_seqlens=cu_seqlens,
        )

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented for fused_recurrent_triadic_kda; "
            "use chunk_triadic_kda for training.",
        )


def fused_recurrent_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    gk: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    A2_log: torch.Tensor | None = None,
    dt2_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Fused recurrent Triadic Kimi Delta Attention with dual-axis decay (`gk` and `g2`) and scalar write gate `beta`.
    Backward is unsupported; use :func:`chunk_triadic_kda` when gradients are required.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        k2 (torch.Tensor):
            Second keys of shape `[B, T, H, E]`.
        q2 (torch.Tensor):
            Second queries of shape `[B, T, H, E]`.
        gk (torch.Tensor):
            First-axis decays of shape `[B, T, H, K]`. Log-space when `use_gate_in_kernel=False`,
            or raw pre-activation inputs when `use_gate_in_kernel=True`.
        g2 (torch.Tensor):
            Second-axis decays of shape `[B, T, H, E]`. Log-space when `use_gate_in_kernel=False`,
            or raw pre-activation inputs when `use_gate_in_kernel=True`.
        beta (torch.Tensor, Optional):
            Write strengths of shape `[B, T, H]`. Default: `None` (ones).
        scale (float, Optional):
            Scale factor of the output. Default: `None` (`K ** -0.5`).
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, E, K, V]` for `N` input sequences. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape `[N, H, E, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool, Optional):
            Whether to apply L2 normalization to `q` and `k` inside the kernel. Default: `False`.
        use_gate_in_kernel (bool, Optional):
            Whether to fuse decay activation on `gk` and `g2` inside the kernel. Default: `False`.
        A_log (torch.Tensor, Optional):
            First-axis log decay scale of shape `[H]` or `[H * K]`. Default: `None`.
        dt_bias (torch.Tensor, Optional):
            First-axis decay bias of shape `[H * K]`. Default: `None`.
        A2_log (torch.Tensor, Optional):
            Second-axis log decay scale of shape `[H]` or `[H * E]`. Default: `None`.
        dt2_bias (torch.Tensor, Optional):
            Second-axis decay bias of shape `[H * E]`. Default: `None`.
        use_beta_sigmoid_in_kernel (bool, Optional):
            Whether to apply `sigmoid(beta)` inside the kernel. Default: `False`.
        allow_neg_eigval (bool, Optional):
            Whether to scale `sigmoid(beta)` by `2` when `use_beta_sigmoid_in_kernel=True`. Default: `False`.
        lower_bound (float, Optional):
            Optional negative lower bound in log space when `use_gate_in_kernel=True`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N + 1]` for variable-length inputs (`B == 1`). Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            FP32 final state of shape `[N, H, E, K, V]` if `output_final_state=True`, else `None`.
    """
    B, T, H, K = k.shape
    E, V = k2.shape[-1], v.shape[-1]
    if q.shape != k.shape or gk.shape != k.shape:
        raise ValueError(f"q, k and gk must share shape [B, T, H, K]={[B, T, H, K]}.")
    if v.shape[:3] != (B, T, H):
        raise ValueError(f"v must have shape [B, T, H, V]=[{B}, {T}, {H}, V], got {list(v.shape)}.")
    if k2.shape != (B, T, H, E) or q2.shape != (B, T, H, E) or g2.shape != (B, T, H, E):
        raise ValueError(f"k2, q2 and g2 must share shape [B, T, H, E]={[B, T, H, E]}.")
    if K > 256 or V > 256 or E > 16:
        raise ValueError(f"fused_recurrent_triadic_kda supports K, V <= 256 and E <= 16, got K={K}, V={V}, E={E}.")

    N = B
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        N = len(cu_seqlens) - 1
    if initial_state is not None and initial_state.shape != (N, H, E, K, V):
        raise ValueError(
            f"initial_state must have shape [N, H, E, K, V]={[N, H, E, K, V]}, got {list(initial_state.shape)}.",
        )
    if scale is None:
        scale = K ** -0.5
    if beta is None:
        beta = torch.ones(B, T, H, device=q.device, dtype=torch.float32)
    elif beta.shape != (B, T, H):
        raise ValueError(f"beta must have shape [B, T, H]={[B, T, H]}, got {list(beta.shape)}.")

    if use_gate_in_kernel:
        if lower_bound is None and (A_log is None or A2_log is None):
            raise ValueError("`A_log` and `A2_log` are required when `use_gate_in_kernel=True` without `lower_bound`.")
        if A_log is not None and A_log.numel() not in (H, H * K):
            raise ValueError(f"`A_log` must have {H} or {H * K} elements, got {A_log.numel()}.")
        if A2_log is not None and A2_log.numel() not in (H, H * E):
            raise ValueError(f"`A2_log` must have {H} or {H * E} elements, got {A2_log.numel()}.")
        if dt_bias is not None and dt_bias.numel() != H * K:
            raise ValueError(f"`dt_bias` must have {H * K} elements, got {dt_bias.numel()}.")
        if dt2_bias is not None and dt2_bias.numel() != H * E:
            raise ValueError(f"`dt2_bias` must have {H * E} elements, got {dt2_bias.numel()}.")
    else:
        A_log = None
        dt_bias = None
        A2_log = None
        dt2_bias = None
        lower_bound = None
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    return FusedRecurrentTriadicKDAFunction.apply(
        q,
        k,
        v,
        k2,
        q2,
        gk,
        g2,
        beta,
        scale,
        A_log,
        dt_bias,
        A2_log,
        dt2_bias,
        lower_bound,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        cu_seqlens,
    )
