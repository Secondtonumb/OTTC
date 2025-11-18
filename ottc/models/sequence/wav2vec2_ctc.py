# SPDX-FileCopyrightText: 2023 HazyResearch (https://github.com/HazyResearch/safari)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-Comment: Adapted for OTTC (configuration hooks, SoftDTW loss integration, wav2vec2 settings)

""" Copied from : https://github.com/HazyResearch/safari/ """
"""
Simplified standalone version of Hyena: https://arxiv.org/abs/2302.10866, designed for quick experimentation.
A complete version is available under `xfsc.models.sequence.hyena`.
"""

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput
from functools import partial
import torchaudio
from ottc.ops_loss.soft_dtw_loss import SoftDTW
from ottc.config.path_config import LARGE_MODELS_PATH

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
)

class wvCTCConfig(PretrainedConfig):
    model_type = 'wv'
    def __init__(self, 
                wav2vec2_path= LARGE_MODELS_PATH / 'wav2vec2-large-lv60',
                wav2vec2_output_dropout= 0.1,
                                
                hidden_size= 768,
                vocab_size= 32,

                loss_type= 'ctc',
                
                wav2vec2_config=None,
                **kwargs):
        super().__init__(**kwargs)
        self.wav2vec2_path = wav2vec2_path
        self.wav2vec2_output_dropout = wav2vec2_output_dropout
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.loss_type = loss_type
        
        self.wav2vec2_config = wav2vec2_config
        
class wvForCTC(PreTrainedModel):
    
    config_class = wvCTCConfig

    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
            
        self.step = -1

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.wav2vec2_path)   
             
        self.dropout = nn.Dropout(config.wav2vec2_output_dropout)
        
        self.lm_head = nn.Linear(self.wav2vec2.config.hidden_size, self.config.vocab_size)
        
        # use of softdtw loss
        self.loss_type = config.loss_type
        if self.loss_type == "softdtweuc":
            self.softdtw = SoftDTW(gamma=0.1, normalize=False, cost='euc')
        elif self.loss_type == "softdtw":
            self.softdtw = SoftDTW(gamma=0.1, normalize=False, cost='ce')
        elif self.loss_type == "softdtw_1":
            self.softdtw = SoftDTW(gamma=1.0, normalize=False, cost='ce')
        elif self.loss_type == "softdtw_0.01":
            self.softdtw = SoftDTW(gamma=0.01, normalize=False, cost='ce')

        print("loss type used: ", self.loss_type)

        self.wave2vec2_config = self.wav2vec2.config
        self.config = config
                        
        # self.post_init()
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        self.wav2vec2.gradient_checkpointing_enable()
                    
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()
        
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _get_feat_extract_output_lengths(
        self, config, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, config.adapter_stride)

        return input_lengths


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Wav2Vec2Encoder, Wav2Vec2EncoderStableLayerNorm, Wav2Vec2FeatureEncoder)):
            module.gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs = None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        
    def create_attention_mask_from_input_sequence(self, input_sequence):
        ''' create attention mask from input sequence 
                input_sequence: tensor of shape (batch_size) containing the length of each sequence in the batch
            output:
                attention_mask: tensor of shape (batch_size, max_length in input_sequence) 
        '''
        tampon = (torch.arange(1,max(input_sequence)+1)).repeat(len(input_sequence), 1).to('cuda')

        attention_mask = (tampon <= input_sequence.unsqueeze(1)).float()

        return attention_mask
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_hyena: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """   
        self.step = self.step + 1
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )[0]
        
        hidden_states = self.dropout(hidden_states)
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(self.wave2vec2_config, attention_mask.sum(-1)).to(torch.long)
            
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            if self.loss_type == "ctc":
                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.wave2vec2_config.pad_token_id,
                        reduction=self.wave2vec2_config.ctc_loss_reduction,
                        zero_infinity=self.wave2vec2_config.ctc_zero_infinity,
                    )
            elif "softdtw" in self.loss_type:
                output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
                labels_mask = (labels >= 0).float()
                labels[labels == -100] = 0 # value choosen not important since it will be masked
                one_hot_labels = F.one_hot(labels, num_classes=self.config.vocab_size)
                loss = 0
                for k in range(logits.shape[0]):
                    x = logits[k,output_mask[k,:] > 0,:]
                    y = one_hot_labels[k,labels_mask[k,:]>0,:]
                    loss += self.softdtw(x, y)
                loss = loss / logits.shape[0]
            elif self.loss_type == "best_ctc_path" :
                # only maxmise the most probable path
                output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
                oracle_out_pred = logits.argmax(-1)
                FA = [ torchaudio.functional.forced_align(log_probs=logits[k:k+1,:output_mask.sum(axis=-1)[k].int()],
                                                targets=labels[k:k+1,:labels_mask.sum(axis=-1)[k].int()],
                                                input_lengths=output_mask.sum(axis=-1)[k:k+1],
                                                target_lengths=labels_mask.sum(axis=-1)[k:k+1])[0]  for k in range(oracle_out_pred.shape[0])]
                labels = torch.nn.utils.rnn.pad_sequence([f.T for f in FA], batch_first = True).squeeze()
                one_hot_labels = F.one_hot(labels, num_classes=self.config.vocab_size)
                x = logits.reshape([-1,logits.shape[-1]])
                y = one_hot_labels.reshape([-1,one_hot_labels.shape[-1]]).float()
                w = output_mask.reshape([-1]).float()
                cross_list_ = F.cross_entropy(x , y, reduction= 'none')
                loss = (w * cross_list_).sum()
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states= None, attentions= output_mask
        )
