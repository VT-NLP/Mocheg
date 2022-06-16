import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union
from transformers import BartConfig,BartForConditionalGeneration
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BartForControllableGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        truthfulness=None
    )  :
        return super().forward(input_ids,attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        head_mask,
        decoder_head_mask,
        cross_attn_head_mask,
        encoder_outputs,
        past_key_values,
        inputs_embeds,
        decoder_inputs_embeds,
        labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict )

        