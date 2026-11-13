# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.triadic_kda.configuration_triadic_kda import TriadicKDAConfig
from fla.models.triadic_kda.modeling_triadic_kda import TriadicKDAForCausalLM, TriadicKDAModel

AutoConfig.register(TriadicKDAConfig.model_type, TriadicKDAConfig, exist_ok=True)
AutoModel.register(TriadicKDAConfig, TriadicKDAModel, exist_ok=True)
AutoModelForCausalLM.register(TriadicKDAConfig, TriadicKDAForCausalLM, exist_ok=True)

__all__ = ['TriadicKDAConfig', 'TriadicKDAForCausalLM', 'TriadicKDAModel']
