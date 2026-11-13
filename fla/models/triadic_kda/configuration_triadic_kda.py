# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

from transformers.configuration_utils import PreTrainedConfig

from fla.models.configuration_utils import (
    AttentionConfig,
    AttentionConfigBase,
    validate_attnres_config,
)


class TriadicKDAConfig(AttentionConfigBase, PreTrainedConfig):
    model_type = 'triadic_kda'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = 'chunk',
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        second_key_dim: int = 4,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        disable_recompute: bool = False,
        head_dim: int = 128,
        num_heads: int = 16,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        attn: AttentionConfig | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        attnres_block_size: int | None = None,
        **kwargs,
    ):
        if attn_mode not in ('chunk', 'fused_recurrent'):
            raise ValueError(f"attn_mode must be 'chunk' or 'fused_recurrent', got {attn_mode!r}.")
        if type(second_key_dim) is not int or second_key_dim not in (1, 2, 4, 8, 16):
            raise ValueError(f"second_key_dim must be in {{1, 2, 4, 8, 16}}, got {second_key_dim!r}.")
        head_v_dim = int(head_dim * expand_v)
        if head_v_dim < 1 or not math.isclose(head_dim * expand_v, head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} must yield a positive integer head_v_dim with head_dim={head_dim}.")
        if head_dim > 256 or head_v_dim > 256:
            raise ValueError(f"TriadicKDAConfig supports K, V <= 256, got K={head_dim}, V={head_v_dim}.")
        if lower_bound is not None and (not math.isfinite(lower_bound) or lower_bound >= 0):
            raise ValueError('lower_bound must be finite and negative when specified.')
        validate_attnres_config(attnres_block_size=attnres_block_size)

        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.second_key_dim = second_key_dim
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.lower_bound = lower_bound
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.disable_recompute = disable_recompute
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.attnres_block_size = attnres_block_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
