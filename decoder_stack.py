# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The decoder stack in inference mode.

Patched for modern meliad compatibility (2026+).
Requires meliad_lib/meliad to be on sys.path.
"""

from typing import Any, Tuple

import gin
import jax.numpy as jnp
import flax.struct as struct

# Import from meliad transformer using direct module path
# (meliad_lib/meliad must be on sys.path)
from transformer.decoder_stack import DecoderStack, DStackWindowState, TransformerTaskConfig
from transformer import nn_components
from transformer import position
from transformer import attention

import transformer_layer as tl

Array = Any

DStackDecoderState = Tuple[tl.DecoderState, ...]


@gin.configurable
class DecoderStackGenerate(DecoderStack):
  """Stack of transformer decoder layers."""

  layer_factory = tl.TransformerLayerGenerate

  def init_decoder_state_vanilla(
      self, sequence_length: int, start_of_sequence: Array
  ) -> DStackDecoderState:
    """Return initial state for autoregressive generation."""
    return tuple(
        [
            layer.init_decoder_state_vanilla(sequence_length, start_of_sequence)
            for layer in self.transformer_layers
        ]
    )
