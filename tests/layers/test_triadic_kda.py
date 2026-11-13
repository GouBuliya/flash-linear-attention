# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from fla.layers.triadic_kda import TriadicKimiDeltaAttention
from fla.models.cache_utils import Cache
from fla.ops.triadic_kda import naive_recurrent_triadic_kda
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('E', 'use_short_conv', 'allow_neg_eigval', 'lower_bound', 'dtype'),
    [
        pytest.param(
            *test,
            id="E{}-conv{}-neg{}-lb{}-{}".format(*test),
        )
        for test in [
            (1, True, False, None, torch.bfloat16),
            (4, True, False, None, torch.bfloat16),
            (8, False, True, -5.0, torch.bfloat16),
            (4, True, False, None, torch.float32),
        ]
    ],
)
def test_layer_gradients(
    monkeypatch,
    E: int,
    use_short_conv: bool,
    allow_neg_eigval: bool,
    lower_bound: float | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    layer = TriadicKimiDeltaAttention(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        second_key_dim=E,
        use_short_conv=use_short_conv,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
        disable_recompute=True,
        layer_idx=0,
    ).to(device)
    ref_layer = deepcopy(layer)

    x = torch.randn(2, 65, 64, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        actual = layer(x)[0]
    dout = torch.randn_like(actual)
    (actual * dout).sum().backward()

    def reference(**kwargs):
        q, k, v = kwargs['q'], kwargs['k'], kwargs['v']
        H, K = q.shape[-2], q.shape[-1]
        E_dim = kwargs['k2'].shape[-1]
        q_n = (q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k_n = (k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
        gk_pre = kwargs['gk'].float() + kwargs['dt_bias'].view(H, K)
        g2_pre = kwargs['g2'].float() + kwargs['dt2_bias'].view(H, E_dim)
        lb = kwargs['lower_bound']
        if lb is None:
            gk_eff = -kwargs['A_log'].view(H, 1).exp() * F.softplus(gk_pre)
            g2_eff = -kwargs['A2_log'].view(H, 1).exp() * F.softplus(g2_pre)
        else:
            gk_eff = lb * torch.sigmoid(kwargs['A_log'].view(H, 1).exp() * gk_pre)
            g2_eff = lb * torch.sigmoid(kwargs['A2_log'].view(H, 1).exp() * g2_pre)
        beta_eff = kwargs['beta'].float().sigmoid()
        if kwargs['allow_neg_eigval']:
            beta_eff = 2.0 * beta_eff
        return naive_recurrent_triadic_kda(
            q=q_n,
            k=k_n,
            v=v,
            k2=kwargs['k2'],
            q2=kwargs['q2'],
            gk=gk_eff,
            g2=g2_eff,
            beta=beta_eff,
            initial_state=kwargs['initial_state'],
            output_final_state=kwargs['output_final_state'],
            cu_seqlens=kwargs['cu_seqlens'],
        )

    monkeypatch.setattr('fla.layers.triadic_kda.chunk_triadic_kda', reference)
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        expected = ref_layer(ref_x)[0]
    (expected * dout).sum().backward()

    # over 12 seeds in bf16 (E=8, allow_neg_eigval=True, lower_bound=-5.0),
    # relative L2 error through o_norm + o_proj measured 0.53% mean, 0.565% max,
    # cosine >= 0.999984, bias projection -0.000036 (t = -1.72, signs 5/12)
    assert_close('o', expected, actual, 0.006)
    assert_close('dx', ref_x.grad, x.grad, 0.008)
    ref_params = dict(ref_layer.named_parameters())
    gate_keys = ('A_log', 'A2_log', 'dt_bias', 'dt2_bias', 'f_proj', 'f2_proj', 'b_proj')
    for name, param in layer.named_parameters():
        assert param.grad is not None and ref_params[name].grad is not None, name
        tol = 0.025 if any(k in name for k in gate_keys) else 0.01
        assert_close(name, ref_params[name].grad, param.grad, tol)


@torch.no_grad()
@pytest.mark.parametrize('E', [1, 4, 8])
def test_layer_cache_and_varlen(E: int):
    torch.manual_seed(43)
    layer = TriadicKimiDeltaAttention(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        second_key_dim=E,
        layer_idx=0,
    ).to(device).eval()

    x = torch.randn(2, 66, 64, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        full = layer(x)[0]
        cu = torch.tensor([0, 66, 132], dtype=torch.int32, device=device)
        packed = layer(x.reshape(1, 132, 64), cu_seqlens=cu)[0].reshape_as(full)
        cache = Cache()
        prefix = layer(x[:, :65], past_key_values=cache, use_cache=True)[0]
        step = layer(x[:, 65:], past_key_values=cache, use_cache=True)[0]

    assert_close('packed', full, packed, 0.005)
    assert_close('cached_decode', full, torch.cat([prefix, step], dim=1), 0.005)
    assert cache[0]['recurrent_state'].shape == (2, 2, E, 32, 32)
    assert len(cache[0]['conv_state']) == 5
