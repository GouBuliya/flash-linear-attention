# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.triadic_kda import (
    chunk_triadic_kda,
    fused_recurrent_triadic_kda,
    naive_chunk_triadic_kda,
    naive_recurrent_triadic_kda,
)
from fla.utils import assert_close, device


def _make_inputs(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, dtype: torch.dtype):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    gk = F.logsigmoid(torch.rand(B, T, H, K, dtype=torch.float32)) * gate_scale
    g2 = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) * gate_scale
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device) for x in (q, k, v, k2, q2, gk, g2, beta)]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gate_scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-E{}-gate_scale{}".format(*test))
        for test in [
            (2, 64, 2, 32, 32, 1, 0.5),
            (2, 100, 2, 32, 48, 4, 0.5),
            (1, 63, 3, 16, 16, 8, 0.5),
            (1, 130, 2, 32, 32, 16, 2.0),
        ]
    ],
)
def test_naive(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float):
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, gate_scale, torch.float32)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    ref, ref_ht = naive_recurrent_kda(
        q=(q2[..., None] * q[..., None, :]).flatten(-2),
        k=(k2[..., None] * k[..., None, :]).flatten(-2),
        v=v,
        g=(g2[..., None] + gk[..., None, :]).flatten(-2),
        beta=beta,
        scale=scale,
        initial_state=h0.flatten(2, 3),
        output_final_state=True,
    )
    tri, tri_ht = naive_recurrent_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 1e-5)
    assert_close('ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)

    chunk_o, chunk_ht = naive_chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('chunk_o', ref, chunk_o, 1e-5)
    assert_close('chunk_ht', ref_ht, chunk_ht.flatten(2, 3), 1e-5)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'scale', 'gate_scale', 'use_state', 'disable_recompute', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-scale{}-gate_scale{}-use_state{}-disable_recompute{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 64, 1, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 200, 2, 64, 64, 4, 1.0, 0.2, True, False, torch.float32),
            (2, 129, 3, 60, 64, 2, 1.0, 0.2, True, True, torch.bfloat16),
            (2, 512, 2, 128, 128, 4, 0.1, 1.5, False, False, torch.bfloat16),
            (2, 512, 2, 128, 128, 8, 1.0, 0.2, False, True, torch.bfloat16),
            (1, 257, 2, 100, 96, 8, 1.0, 0.1, True, False, torch.bfloat16),
            (1, 130, 2, 64, 64, 16, 1.0, 0.2, True, True, torch.bfloat16),
            (2, 100, 2, 8, 12, 2, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 193, 2, 256, 128, 4, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 129, 2, 128, 128, 4, 1.0, 0.2, True, True, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    scale: float,
    gate_scale: float,
    use_state: bool,
    disable_recompute: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device) if use_state else None
    inputs = [x.requires_grad_(True) for x in _make_inputs(B, T, H, K, V, E, gate_scale, dtype)]
    if use_state:
        inputs.append(h0.requires_grad_(True))
    q, k, v, k2, q2, gk, g2, beta = inputs[:8]
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) if use_state else None

    oracle = naive_recurrent_triadic_kda if T <= 256 else naive_chunk_triadic_kda
    ref, ref_ht = oracle(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
    )
    ((ref * do).sum() + ((ref_ht * dht).sum() if use_state else 0)).backward()
    ref_grads = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None

    tri, tri_ht = chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + ((tri_ht * dht).sum() if use_state else 0)).backward()
    tri_grads = [x.grad for x in inputs]

    assert_close('o', ref, tri, 0.005)
    if use_state:
        assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dh0'),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.008),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gk_scale', 'g2_scale', 'disable_recompute', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-gk_scale{}-g2_scale{}-disable_recompute{}-{}".format(*test),
        )
        for test in [
            (2, 130, 2, 64, 64, 4, 10.0, 0.5, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 4, 25.0, 0.5, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 4, 0.5, 25.0, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 4, 32.0, 32.0, False, torch.float32),
            (1, 257, 2, 100, 96, 8, 32.0, 32.0, True, torch.bfloat16),
        ]
    ],
)
def test_chunk_strong_decay(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    gk_scale: float,
    g2_scale: float,
    disable_recompute: bool,
    dtype: torch.dtype,
):
    # per-token log decays average about -0.5 * scale, so a 16-token diagonal block spans up to about 240
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, 1.0, dtype)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    inputs = [x.requires_grad_(True) for x in (q, k, v, k2, q2, gk * gk_scale, g2 * g2_scale, beta, h0)]
    q, k, v, k2, q2, gk, g2, beta, h0 = inputs
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
    )
    ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)

    tri, tri_ht = chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        disable_recompute=disable_recompute,
    )
    tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dh0'),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.008),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ('H', 'K', 'V', 'E', 'cu_seqlens', 'in_kernel_gates', 'lower_bound', 'disable_recompute'),
    [
        pytest.param(
            *test,
            id="H{}-K{}-V{}-E{}-cu_seqlens{}-in_kernel{}-lb{}-save{}".format(*test),
        )
        for test in [
            (2, 64, 64, 4, [0, 15, 80, 145], False, None, False),
            (2, 128, 128, 4, [0, 1, 65, 130], True, None, True),
            (3, 64, 96, 8, [0, 33, 128, 210], True, -5.0, False),
        ]
    ],
)
def test_chunk_varlen_and_in_kernel_gates(
    H: int,
    K: int,
    V: int,
    E: int,
    cu_seqlens: list[int],
    in_kernel_gates: bool,
    lower_bound: float | None,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    h0 = torch.randn(N, H, E, K, V, dtype=torch.float32, device=device, requires_grad=True)

    if in_kernel_gates:
        q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device, requires_grad=True)
        k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device, requires_grad=True)
        v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device, requires_grad=True)
        k2 = F.normalize(
            F.softplus(torch.randn(1, T, H, E, dtype=torch.float32, device=device)), p=2, dim=-1,
        ).detach().requires_grad_(True)
        q2 = F.normalize(
            F.softplus(torch.randn(1, T, H, E, dtype=torch.float32, device=device)), p=2, dim=-1,
        ).detach().requires_grad_(True)
        gk = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device, requires_grad=True)
        g2 = torch.randn(1, T, H, E, dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
        A_log = torch.empty(H, dtype=torch.float32, device=device).uniform_(1.0, 8.0).log().detach().requires_grad_(True)
        dt_bias = torch.full((H * K,), -3.0, dtype=torch.float32, device=device, requires_grad=True)
        A2_log = torch.empty(H, dtype=torch.float32, device=device).uniform_(1.0, 8.0).log().detach().requires_grad_(True)
        dt2_bias = torch.full((H * E,), -3.0, dtype=torch.float32, device=device, requires_grad=True)
        inputs = [q, k, v, k2, q2, gk, g2, beta, A_log, dt_bias, A2_log, dt2_bias, h0]
        names = ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dA', 'ddt', 'dA2', 'ddt2', 'dh0')

        tri, tri_ht = chunk_triadic_kda(
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
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            lower_bound=lower_bound,
            cu_seqlens=cu,
            disable_recompute=disable_recompute,
        )
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)

        q_n = (q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k_n = (k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
        gk_pre = gk.float() + dt_bias.view(H, K)
        g2_pre = g2.float() + dt2_bias.view(H, E)
        if lower_bound is None:
            gk_eff = -A_log.view(H, 1).exp() * F.softplus(gk_pre)
            g2_eff = -A2_log.view(H, 1).exp() * F.softplus(g2_pre)
        else:
            gk_eff = lower_bound * torch.sigmoid(A_log.view(H, 1).exp() * gk_pre)
            g2_eff = lower_bound * torch.sigmoid(A2_log.view(H, 1).exp() * g2_pre)
        beta_eff = 2.0 * beta.float().sigmoid()
        ref, ref_ht = naive_recurrent_triadic_kda(
            q=q_n,
            k=k_n,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk_eff,
            g2=g2_eff,
            beta=beta_eff,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)
    else:
        base_inputs = [x.requires_grad_(True) for x in _make_inputs(1, T, H, K, V, E, 0.2, torch.bfloat16)]
        q, k, v, k2, q2, gk, g2, beta = base_inputs
        inputs = [*base_inputs, h0]
        names = ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dh0')
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        tri, tri_ht = chunk_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            disable_recompute=disable_recompute,
        )
        tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)
        ref, ref_ht = naive_recurrent_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(names, ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.02)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'in_kernel_gates', 'lower_bound', 'varlen', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-in_kernel{}-lb{}-varlen{}-{}".format(*test),
        )
        for test in [
            (2, 17, 2, 64, 64, 1, False, None, False, torch.float16),
            (2, 33, 2, 128, 128, 4, False, None, False, torch.bfloat16),
            (1, 29, 2, 128, 128, 8, True, None, True, torch.bfloat16),
            (1, 15, 2, 100, 96, 8, True, -5.0, False, torch.bfloat16),
            (1, 9, 2, 256, 128, 16, False, None, True, torch.bfloat16),
            (2, 19, 2, 64, 64, 4, False, None, False, torch.float32),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    in_kernel_gates: bool,
    lower_bound: float | None,
    varlen: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    cu = torch.tensor([0, 7, T], dtype=torch.int32, device=device) if varlen else None
    N = 2 if varlen else B
    h0 = torch.randn(N, H, E, K, V, dtype=torch.float32, device=device)

    if in_kernel_gates:
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32, device=device)), p=2, dim=-1)
        q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32, device=device)), p=2, dim=-1)
        gk_raw = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        g2_raw = torch.randn(B, T, H, E, dtype=torch.float32, device=device)
        beta_raw = torch.randn(B, T, H, dtype=torch.float32, device=device)
        A_log = torch.empty(H, dtype=torch.float32, device=device).uniform_(1.0, 8.0).log()
        dt_bias = torch.full((H * K,), -3.0, dtype=torch.float32, device=device)
        A2_log = torch.empty(H, dtype=torch.float32, device=device).uniform_(1.0, 8.0).log()
        dt2_bias = torch.full((H * E,), -3.0, dtype=torch.float32, device=device)
        tri, tri_ht = fused_recurrent_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk_raw,
            g2=g2_raw,
            beta=beta_raw,
            A_log=A_log,
            dt_bias=dt_bias,
            A2_log=A2_log,
            dt2_bias=dt2_bias,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            lower_bound=lower_bound,
            cu_seqlens=cu,
        )
        q_n = q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        k_n = k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        gk_pre = gk_raw + dt_bias.view(H, K)
        g2_pre = g2_raw + dt2_bias.view(H, E)
        if lower_bound is None:
            gk_eff = -A_log.view(H, 1).exp() * F.softplus(gk_pre)
            g2_eff = -A2_log.view(H, 1).exp() * F.softplus(g2_pre)
        else:
            gk_eff = lower_bound * torch.sigmoid(A_log.view(H, 1).exp() * gk_pre)
            g2_eff = lower_bound * torch.sigmoid(A2_log.view(H, 1).exp() * g2_pre)
        beta_eff = 2.0 * beta_raw.sigmoid()
        ref, ref_ht = naive_recurrent_triadic_kda(
            q=q_n,
            k=k_n,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk_eff,
            g2=g2_eff,
            beta=beta_eff,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
    else:
        q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, 0.2, dtype)
        tri, tri_ht = fused_recurrent_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        ref, ref_ht = naive_recurrent_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)

    if not in_kernel_gates:
        q_req = q.detach().clone().requires_grad_(True)
        with pytest.raises(NotImplementedError):
            out, _ = fused_recurrent_triadic_kda(
                q=q_req,
                k=k,
                v=v,
                k2=k2,
                q2=q2,
                gk=gk,
                g2=g2,
                beta=beta,
            )
            out.sum().backward()
