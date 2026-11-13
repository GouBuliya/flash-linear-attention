# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triadic Kimi Delta Attention layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.norm.l2norm import l2norm
from fla.ops.triadic_kda import chunk_triadic_kda, fused_recurrent_triadic_kda

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache

# Symmetric log-uniform dt range across both axes keeps the composite half-life
# aligned with single-axis KDA/GDN instead of shrinking by ~3x under additive rates.
DT_RANGE = (1e-4, 5.5e-2)
DT_FLOOR = 1e-4


class TriadicKimiDeltaAttention(nn.Module):
    """
    Triadic Kimi Delta Attention (Triadic KDA) layer implementation.

    Extends Kimi Delta Attention (`KimiDeltaAttention`) to a third-order state `[N, H, E, K, V]`
    with Kronecker joint keys `k2 ⊗ k`, dual-axis per-channel/per-slice forget gates (`gk` and `g2`),
    and a scalar per-head write gate `beta`:

        S_{e,k} <- exp(g2_{t,e} + gk_{t,k}) * S_{e,k}
        S       <- S + beta_t * (k2_t ⊗ k_t) (v_t - (k2_t ⊗ k_t)^T S)^T
        o_t     = scale * S^T (q2_t ⊗ q_t)

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The key dimension of each head (`K <= 256`). Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        second_key_dim (int, Optional):
            The second-axis slice count `E` (`E` in `{1, 2, 4, 8, 16}`). Default: 4.
        mode (str, Optional):
            Which Triadic KDA kernel to use (`chunk` or `fused_recurrent`). Default: `'chunk'`.
        use_short_conv (bool, Optional):
            Whether to apply causal short convolutions on `q, k, v, k2, q2`. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Whether to scale `sigmoid(beta)` by `2` to allow `(0, 2)` eigenvalues. Default: `False`.
        lower_bound (float, Optional):
            Optional negative lower bound in log space for both decay axes. Default: `None`.
        conv_size (int, Optional):
            Short convolution kernel width. Default: 4.
        conv_bias (bool, Optional):
            Whether to include bias in short convolutions. Default: `False`.
        disable_recompute (bool, Optional):
            Whether to save chunk states and solved residuals during training. Default: `False`.
        layer_idx (int, Optional):
            Layer index used for cache indexing. Default: `None`.
        norm_eps (float, Optional):
            Epsilon for gated output RMSNorm. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        second_key_dim: int = 4,
        mode: str = 'chunk',
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        disable_recompute: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        if mode not in ('chunk', 'fused_recurrent'):
            raise ValueError(f"Unsupported mode `{mode}`; expected 'chunk' or 'fused_recurrent'.")
        if second_key_dim not in (1, 2, 4, 8, 16):
            raise ValueError(f"second_key_dim must be in {{1, 2, 4, 8, 16}}, got {second_key_dim}.")
        head_v_dim = int(head_dim * expand_v)
        if head_v_dim < 1 or not math.isclose(head_dim * expand_v, head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} must yield a positive integer head_v_dim with head_dim={head_dim}.")
        if min(hidden_size, num_heads, head_dim, conv_size) < 1 or head_dim > 256 or head_v_dim > 256:
            raise ValueError('TriadicKimiDeltaAttention requires positive dimensions and K, V <= 256.')
        if lower_bound is not None and (not math.isfinite(lower_bound) or lower_bound >= 0):
            raise ValueError('lower_bound must be finite and negative when specified.')

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.lower_bound = lower_bound
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.disable_recompute = disable_recompute
        self.head_dim = head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = head_v_dim
        self.num_heads = num_heads
        self.second_key_dim = second_key_dim
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * head_v_dim
        self.slice_dim = num_heads * second_key_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.k2_proj = nn.Linear(hidden_size, self.slice_dim, bias=False)
        self.q2_proj = nn.Linear(hidden_size, self.slice_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k2_conv1d = ShortConvolution(self.slice_dim, kernel_size=conv_size, bias=conv_bias, activation=None)
            self.q2_conv1d = ShortConvolution(self.slice_dim, kernel_size=conv_size, bias=conv_bias, activation=None)

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.f2_proj = nn.Linear(hidden_size, self.slice_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        self.A2_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt2_bias = nn.Parameter(torch.empty(self.slice_dim, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.A2_log._no_weight_decay = True
        self.dt2_bias._no_weight_decay = True
        self.reset_parameters()

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        dt_low, dt_high = DT_RANGE
        for log_scale, bias in ((self.A_log, self.dt_bias), (self.A2_log, self.dt2_bias)):
            log_scale._no_weight_decay = True
            bias._no_weight_decay = True
            if bias.device.type == 'meta':
                continue
            if not getattr(log_scale, '_is_hf_initialized', False):
                log_scale.uniform_(1.0, 16.0).log_()
            if not getattr(bias, '_is_hf_initialized', False):
                dt = torch.empty_like(bias, dtype=torch.float32).uniform_(math.log(dt_low), math.log(dt_high))
                dt = dt.exp_().clamp_(min=DT_FLOOR)
                bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError('Expected attention_mask as a 2-D padding mask of shape [batch_size, seq_len].')

        batch_size, q_len, _ = hidden_states.shape
        if torch.is_grad_enabled():
            mode = 'chunk'
        elif q_len <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode
        if self.training and mode != 'chunk':
            raise ValueError('Only chunk mode is supported during training.')

        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get('cu_seqlens')
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        with torch.autocast(hidden_states.device.type, enabled=False):
            x_fp32 = hidden_states.float()
            k2 = F.linear(x_fp32, self.k2_proj.weight.float())
            q2 = F.linear(x_fp32, self.q2_proj.weight.float())

        if self.use_short_conv:
            cs_q, cs_k, cs_v, cs_k2, cs_q2 = last_state['conv_state'] if last_state is not None else (None,) * 5
            q, cs_q = self.q_conv1d(x=q, cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, cs_k = self.k_conv1d(x=k, cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, cs_v = self.v_conv1d(x=v, cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k2, cs_k2 = self.k2_conv1d(x=k2, cache=cs_k2, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            q2, cs_q2 = self.q2_conv1d(x=q2, cache=cs_q2, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            next_conv = (cs_q, cs_k, cs_v, cs_k2, cs_q2)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
            next_conv = None

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        k2 = l2norm(F.softplus(k2.float()).reshape(*q.shape[:2], self.num_heads, self.second_key_dim))
        q2 = l2norm(F.softplus(q2.float()).reshape(*q.shape[:2], self.num_heads, self.second_key_dim))

        gk = rearrange(self.f_proj(hidden_states), '... (h d) -> ... h d', d=self.head_k_dim)
        g2 = rearrange(self.f2_proj(hidden_states), '... (h e) -> ... h e', e=self.second_key_dim)
        beta = self.b_proj(hidden_states)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_triadic_kda(
                q=q,
                k=k,
                v=v,
                k2=k2,
                q2=q2,
                gk=gk,
                g2=g2,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                A2_log=self.A2_log,
                dt2_bias=self.dt2_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
                disable_recompute=self.disable_recompute,
            )
        else:
            o, recurrent_state = fused_recurrent_triadic_kda(
                q=q,
                k=k,
                v=v,
                k2=k2,
                q2=q2,
                gk=gk,
                g2=g2,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                A2_log=self.A2_log,
                dt2_bias=self.dt2_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=next_conv,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim))
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
