# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from fla.layers import TriadicKimiDeltaAttention
from fla.models.triadic_kda import TriadicKDAConfig, TriadicKDAForCausalLM
from fla.utils import assert_close, device


def _tiny_config(**kwargs) -> TriadicKDAConfig:
    return TriadicKDAConfig(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        second_key_dim=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=128,
        fuse_cross_entropy=False,
        **kwargs,
    )


def test_registration_and_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = TriadicKDAForCausalLM(_tiny_config())
    model.save_pretrained(tmp_path)

    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, TriadicKDAConfig)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(restored, TriadicKDAForCausalLM)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for block in restored.model.layers:
        assert isinstance(block.attn, TriadicKimiDeltaAttention)
        assert block.attn.A_log._no_weight_decay and block.attn.dt_bias._no_weight_decay
        assert block.attn.A2_log._no_weight_decay and block.attn.dt2_bias._no_weight_decay


@pytest.mark.parametrize(
    'kwargs',
    [
        {'second_key_dim': 0},
        {'second_key_dim': 12},
        {'attn_mode': 'invalid'},
        {'head_dim': 512},
        {'lower_bound': 0.0},
    ],
)
def test_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        TriadicKDAConfig(**kwargs)


def test_training_and_cached_decode():
    torch.manual_seed(44)
    model = TriadicKDAForCausalLM(_tiny_config(disable_recompute=True)).to(device).train()
    ids = torch.randint(0, 128, (2, 66), device=device)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(ids, labels=ids, use_cache=False)
        logits = model(ids, use_cache=False).logits
    expected = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(out.loss, expected, atol=0.005, rtol=0.001)

    out.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name

    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = model(ids[:1], use_cache=False).logits
        prefill = model(ids[:1, :65], use_cache=True)
        decode = model(ids[:1, 65:], past_key_values=prefill.past_key_values, use_cache=True)
    # over 12 seeds in 2-layer bf16 CausalLM (chunk full pass vs prefill + fused_recurrent decode),
    # relative L2 logit error measured 0.43% mean, 0.575% max, cosine >= 0.999984,
    # bias projection -0.000178 (t = -1.04, signs 4/12)
    assert_close('decode_logits', full[:, 65:], decode.logits, 0.0065)
