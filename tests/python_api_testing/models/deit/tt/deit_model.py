from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional, Set, Tuple, Union
import torch
from torch import nn

import tt_lib
from utility_functions_new import tt_to_torch_tensor, torch_to_tt_tensor_rm
from deit_config import DeiTConfig
from deit_embeddings import DeiTEmbeddings
from deit_patch_embeddings import DeiTPatchEmbeddings
from deit_encoder import TtDeiTEncoder
from deit_pooler import TtDeiTPooler
from tt_lib.fallback_ops import fallback_ops

class TtDeiTModel(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="", add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__()

        self.config = config
        self.device = device

        self.embeddings = DeiTEmbeddings(config,state_dict=state_dict, base_address=f"{base_address}.embeddings",use_mask_token=use_mask_token)
        self.encoder = TtDeiTEncoder(config, device=device, state_dict=state_dict, base_address=f"{base_address}.encoder")

        ln_weight = state_dict[f"{base_address}.layernorm.weight"]
        ln_bias = state_dict[f"{base_address}.layernorm.bias"]
        self.layernorm = fallback_ops.LayerNorm(normalized_shape= config.hidden_size, eps=config.layer_norm_eps, weights = ln_weight, biases=ln_bias)

        self.pooler = TtDeiTPooler(config, state_dict, f"{base_address}.pooler") if add_pooling_layer else None


    def forward(
        self,
        pixel_values: tt_lib.tensor.Tensor = None,
        bool_masked_pos: bool = None,
        head_mask: bool= None,
        output_attentions: bool= None,
        output_hidden_states: bool= None,
        return_dict: bool = None,
    )-> Tuple[tt_lib.tensor.Tensor]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        pixel_values = tt_to_torch_tensor(pixel_values)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        embedding_output = torch_to_tt_tensor_rm(embedding_output, self.device)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)

        return head_outputs + encoder_outputs[1:]
