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
Uses direct imports instead of from transformer import decoder_stack.
"""

from typing import Any, Tuple

import gin
import jax.numpy as jnp

# Try multiple import paths for meliad compatibility
try:
    from transformer import decoder_stack
    _HAS_DECODER_STACK = True
except ImportError:
    _HAS_DECODER_STACK = False

if _HAS_DECODER_STACK:
    struct = decoder_stack.struct
    nn_components = decoder_stack.nn_components
    position = decoder_stack.position
    attention = decoder_stack.attention
    DStackWindowState = decoder_stack.DStackWindowState
    TransformerTaskConfig = decoder_stack.TransformerTaskConfig
    DecoderStackBase = decoder_stack.DecoderStack
else:
    # Fallback: import directly from meliad submodules
    try:
        from transformer import nn_components
        from transformer import position
        from transformer import attention
        from transformer import decoder_stack as ds_module
        import flax.struct as struct
        
        DStackWindowState = getattr(ds_module, 'DStackWindowState', None)
        TransformerTaskConfig = getattr(ds_module, 'TransformerTaskConfig', None)
        DecoderStackBase = getattr(ds_module, 'DecoderStack', None)
    except ImportError:
        # Final fallback - import from meliad root
        from meliad.transformer import nn_components
        from meliad.transformer import position
        from meliad.transformer import attention
        from meliad.transformer import decoder_stack as ds_module
        import flax.struct as struct
        
        DStackWindowState = ds_module.DStackWindowState
        TransformerTaskConfig = ds_module.TransformerTaskConfig
        DecoderStackBase = ds_module.DecoderStack

import transformer_layer as tl

Array = Any

DStackDecoderState = Tuple[tl.DecoderState, ...]


@gin.configurable
class DecoderStackGenerate(DecoderStackBase):
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
