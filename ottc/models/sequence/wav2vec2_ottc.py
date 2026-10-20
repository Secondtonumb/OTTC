# SPDX-FileCopyrightText: 2023 HazyResearch (https://github.com/HazyResearch/safari)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-Comment: Extended wav2vec2 model with OTTC alignment, oracle weights loading, and Idiap-specific training hooks

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from pathlib import Path
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureProjection
from functools import partial
# from xfsc.config.path_config import DATASETS_ROOT_PATH, HUGGINGFACE_CACHE, LARGE_MODELS_PATH
import torchaudio
import os
import copy

from ottc.ops_loss.ot_loss import batched_ottc_loss_bucketized
from ottc.models.sequence.wav2vec2_ctc import wvForCTC
from ottc.config.path_config import LARGE_MODELS_PATH

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
)

_HIDDEN_STATES_START_POSITION = 2

def get_oracle_weights_model(oracle_weights_model_name: str) -> wvForCTC:
    """
    Load a Wav2Vec2-CTC model trained with OTTC to compute oracle weights for target sequences.
    These weights enforce the same alignment as the most probable valid alignment from the CTC model.

    Args:
        oracle_weights_model_name (str): The name of the trained model directory.

    Returns:
        Wav2Vec2ForCTC: The loaded model instance.

    Raises:
        ValueError: If the specified model directory does not exist.
    """
    print('Loading oracle weights model...')
    
    model_output_dir = LARGE_MODELS_PATH / 'large_models_results' / 'wav2vec2-large-lv60' / oracle_weights_model_name
    prefix = "checkpoint-"
    if model_output_dir.exists() and len(os.listdir(model_output_dir)) > 1:
        print(f'Model found in {model_output_dir}')
        # Find the latest checkpoint
        highest_folder = max(
            (f for f in os.listdir(model_output_dir) if f.startswith(prefix)), 
            key=lambda x: int(x[len(prefix):]), 
            default=None
        )
    
        if highest_folder is None:
            raise ValueError(f"No valid checkpoint found in {model_output_dir}")

        checkpoint_path = str(model_output_dir / highest_folder)
        print(f'Loading latest checkpoint from {checkpoint_path}')
        
        model = wvForCTC.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f'Model "{oracle_weights_model_name}" does not exist')
    return model
        

from transformers import PretrainedConfig
from typing import Optional, Union

class wvOTTCConfig(PretrainedConfig):
    """
    Configuration class for the Wav2Vec2-OTTC model.

    This configuration extends `PretrainedConfig` and includes parameters for
    the Wav2Vec2 model path, dropout rates, vocabulary size, final layer type,
    weight models for alignment, and forward computation type.
    """


    model_type = 'wav2vec2-ottc'

    def __init__(
        self, 
        wav2vec2_path: Union[str, Path] = LARGE_MODELS_PATH / 'wav2vec2-large-lv60',
        wav2vec2_output_dropout: float = 0.1,
        bins_metric: str = 'crossentropy', 
        hidden_size: int = 768,
        vocab_size: int = 32,
        final_layers: str = 'linear',  # 'linear' or 'mlp'
        final_dropout: float = 0.1,
        wav2vec2_config: Optional[dict] = None,
        oracle_weights_model_name: Optional[str] = None,
        alignment_weights_model_name: Optional[str] = None,
        forward_type: str = 'regular',
        frozen_steps_for_regfrozen: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if isinstance(wav2vec2_path, Path):
            wav2vec2_path = str(wav2vec2_path)
        self.wav2vec2_path = str(wav2vec2_path)
        self.wav2vec2_output_dropout = wav2vec2_output_dropout
        self.bins_metric = bins_metric
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.final_layers = final_layers
        self.final_dropout = final_dropout
        self.wav2vec2_config = wav2vec2_config
        self.oracle_weights_model_name = oracle_weights_model_name
        self.alignment_weights_model_name = alignment_weights_model_name
        self.forward_type = forward_type
        self.frozen_steps_for_regfrozen = frozen_steps_for_regfrozen
        
class wvForOTTC(PreTrainedModel):
    """
         A model that combine Wav2Vec2 with OTTC alignment mechanisms.
    """
        
    config_class = wvOTTCConfig

    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        
        self.count = 0

        self.jsd = False
        if self.config.bins_metric == 'crossentropy':
            self.euclidian = False
            self.soft_before_euclidian = False 
        elif self.config.bins_metric == 'jsd_standard':
            self.euclidian = False
            self.jsd = 'standard'
        else:
            if self.config.bins_metric == 'euclidian':
                self.euclidian = True
            else: 
                if self.config.bins_metric == 'jsd':
                    self.euclidian = False
                    self.jsd = True
                else:
                    raise ValueError(f"bins_metric should be 'crossentropy', 'euclidian', 'jsd' or 'jsd_standard' but got {config.bins_metric}")
               
        # Feature extractor / Wav2Vec2 CNN
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.wav2vec2_path)   
             
        self.dropout = nn.Dropout(config.wav2vec2_output_dropout)
                
        if config.final_layers == 'linear':    
            self.lm_head = nn.Linear(self.wav2vec2.config.hidden_size, self.config.vocab_size)
            self.lm_weight = nn.Linear(self.wav2vec2.config.hidden_size, 1)    
        elif config.final_layers == 'mlp':
            self.lm_head = nn.Sequential(
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, self.config.vocab_size)
            )
            self.lm_weight = nn.Sequential(
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, 1)
            )
        elif config.final_layers == 'mlp2':
            self.lm_head = nn.Sequential(
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, self.config.vocab_size)
            )
            self.lm_weight = nn.Sequential(
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.wav2vec2.config.hidden_size, 1)
            )
        else: 
            raise ValueError(f"final_layer_logits should be 'linear' or 'mlp' but got {config.final_layer_logits}")
                                

        self.wave2vec2_config = self.wav2vec2.config
        self.config = config

        self.step = -1
        self.intermediate_step = -1
        self.alignement_training_time = False

        self.oracle_weights_model_bool = False
        self.oracle_weights_model = get_oracle_weights_model(config.oracle_weights_model_name) if config.oracle_weights_model_name is not None else None
        #disable gradient for oracle weights model
        if self.oracle_weights_model is not None:
            for param in self.oracle_weights_model.parameters():
                param.requires_grad = False
            self.oracle_weights_model_bool = True
        
        self.forward_type = config.forward_type
        if self.forward_type  == 'alternate_scheme' or self.forward_type == 'alternate_scheme2' or self.forward_type == 'regfrozen':
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False

        if self.forward_type == 'regfrozen' and self.config.frozen_steps_for_regfrozen == -1:
            raise ValueError(f"... frozen_epochs_for_regfrozen should be set for forward_type = 'regfrozen'")
        self.frozen_steps_for_regfrozen = config.frozen_steps_for_regfrozen

        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        self.wav2vec2.gradient_checkpointing_enable()
                    
    @torch.no_grad()
    def get_mask_from_lengths(self, x: torch.Tensor, x_len: torch.Tensor):
        ''' Creates a mask from lengths of sequences. '''
        padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
        padding_mask[
            (torch.arange(padding_mask.shape[0], device=padding_mask.device), x_len - 1)
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return padding_mask
    
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
        """ Sets the gradient checkpointing of modules """
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

    def regular_forward(        
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ):
        ''' Regular forward pass of the model, optimizing both the prediction and the alignment weights. '''
        hidden_states = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )[0]

        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(self.wave2vec2_config, attention_mask.sum(-1)).to(torch.long)
        output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
        weights_logits = F.softmax( self.lm_weight(hidden_states).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )

        return logits, weights_logits, output_mask, hidden_states

    def alternate_scheme_forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        inital_phase_step: Optional[int] = 13400,
        alterned_phase_step: Optional[int] = 200,
        ):
        """ Alternates between training on prediction and alignment weights after a certain number of steps. """
        if self.training:
            self.step = self.step + 1
            if self.step >= inital_phase_step:
                self.intermediate_step = self.intermediate_step + 1

        if self.step == inital_phase_step:
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False

        if self.intermediate_step == alterned_phase_step:
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False
            self.intermediate_step = 0
            self.alignement_training_time = not self.alignement_training_time

        hidden_states = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )[0]

        hidden_states = self.dropout(hidden_states)

        if self.step >= inital_phase_step:
            with torch.no_grad():
                hidden_states2 = self.wav2vec2_freeze(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )[0]

                hidden_states2 = self.dropout(hidden_states2)

        if self.step < inital_phase_step or not self.alignement_training_time:
            logits = self.lm_head(hidden_states)
        else:
            with torch.no_grad():
                logits = self.lm_head(hidden_states2)

        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(self.wave2vec2_config, attention_mask.sum(-1)).to(torch.long)
        output_mask = self.create_attention_mask_from_input_sequence(input_lengths)        

        if self.step < inital_phase_step or self.alignement_training_time:
            weights_logits = F.softmax( self.lm_weight(hidden_states).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )
        else:
            with torch.no_grad():
                weights_logits = F.softmax( self.lm_weight(hidden_states2).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )

        return logits, weights_logits, output_mask

    def alternate_scheme2_forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        inital_phase_step: Optional[int] = 4400,
        alterned_phase_step: Optional[int] = 200,
        ):
        ''' Alternates between freezing the alignment head and training the alignment weights after a certain number of steps. '''

        if self.training:
            self.step = self.step + 1
            if self.step >= alterned_phase_step :
                self.intermediate_step = self.intermediate_step + 1

        if self.step == inital_phase_step :
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False

        if self.intermediate_step == alterned_phase_step:
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False
            self.intermediate_step = 0
            self.alignement_training_time = not self.alignement_training_time

        hidden_states = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )[0]

        hidden_states = self.dropout(hidden_states)

        if self.step >= inital_phase_step :
            with torch.no_grad():
                hidden_states2 = self.wav2vec2_freeze(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )[0]

                hidden_states2 = self.dropout(hidden_states2)

        logits = self.lm_head(hidden_states)

        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(self.wave2vec2_config, attention_mask.sum(-1)).to(torch.long)
        output_mask = self.create_attention_mask_from_input_sequence(input_lengths)        

        if self.step < inital_phase_step or self.alignement_training_time:
            weights_logits = F.softmax( self.lm_weight(hidden_states).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )
        else:
            with torch.no_grad():
                weights_logits = F.softmax( self.lm_weight(hidden_states2).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )

        return logits, weights_logits, output_mask   

    def regular_forward_and_frozen(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        stop_steps: Optional[int] = 2000,
        ):

        ''' Regular forward pass of the model, optimizing both the prediction and the alignment weights until stop_steps,
        then the alignment head is frozen, and only the prediction is optimized.'''

        if self.training:
            self.step = self.step + 1

        if self.step == stop_steps - 1:
        # copy and freeze whole self.wav2vec2 moel
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2)
            self.wav2vec2_freeze.requires_grad = False

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if  self.step >= stop_steps - 1:
            with torch.no_grad():
                outputs2 = self.wav2vec2_freeze(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )

                hidden_states2 = outputs2[0]
                # hidden_states2 = self.dropout(hidden_states2)
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(self.wave2vec2_config, attention_mask.sum(-1)).to(torch.long)
        output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
        if self.step >= stop_steps - 1:
            with torch.no_grad():
                weights_logits = F.softmax( self.lm_weight(hidden_states2).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )
        else:
            weights_logits = F.softmax( self.lm_weight(hidden_states).squeeze(-1).masked_fill_(output_mask == 0, - torch.inf ) )


        return logits, weights_logits, output_mask

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Main forward pass of the model, orchestrating different forward pass types and loss computation.

        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """   

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.forward_type == 'regular':
            logits, weights_logits, output_mask, hidden_st = self.regular_forward(input_values, attention_mask, return_dict, labels)
        elif self.forward_type == 'alternate_scheme':
            logits, weights_logits, output_mask = self.alternate_scheme_forward(input_values, attention_mask, return_dict, labels)
        elif self.forward_type == 'alternate_scheme2':
            logits, weights_logits, output_mask = self.alternate_scheme2_forward(input_values, attention_mask, return_dict, labels)
        elif self.forward_type == 'regfrozen':
            logits, weights_logits, output_mask = self.regular_forward_and_frozen(input_values, attention_mask, return_dict, stop_steps = self.frozen_steps_for_regfrozen)
        else:
            raise ValueError(f"forward_type {self.forward_type} not recognized")
        
        # Inference must depend only on audio. Transport is needed only for loss.
        if labels is None:
            if not return_dict:
                return (logits,)
            return CausalLMOutput(logits=logits,
                                  hidden_states=(weights_logits, output_mask))
        labels = labels.clone()
        labels_mask = (labels >= 0).float()
        labels[labels == -100] = 0 # value choosen not important since it will be masked
        one_hot_labels = F.one_hot(labels, num_classes=self.config.vocab_size)

        # if another model is used to predict weights
        if self.oracle_weights_model_bool:
            with torch.no_grad():
                oracle_out = self.oracle_weights_model(
                input_values,
                attention_mask,
                None,
                None,
                return_dict,
                labels)
            oracle_out_pred = oracle_out.logits.argmax(-1)
            FA = [ torchaudio.functional.forced_align(log_probs=oracle_out.logits[k:k+1,:output_mask.sum(axis=-1)[k].int()],
                                            targets=labels[k:k+1,:labels_mask.sum(axis=-1)[k].int()],
                                            input_lengths=output_mask.sum(axis=-1)[k:k+1],
                                            target_lengths=labels_mask.sum(axis=-1)[k:k+1])[0]  for k in range(oracle_out_pred.shape[0])]
            shrink_labels =  [ input_tensor[0][input_tensor[0].diff(append = torch.Tensor([100]).int().to('cuda')) != 0] for input_tensor in FA]
            counts_labels = [torch.unique(torch.cumsum( input_tensor[0].diff(prepend = torch.Tensor([input_tensor[0][0]]).int().to('cuda')) != 0 , axis = 0) , return_counts=True)[1] for input_tensor in FA]
            weights_labels = [c / c.sum() for c in counts_labels]
            weights_labels =  torch.nn.utils.rnn.pad_sequence(weights_labels, batch_first = True)
            labels = torch.nn.utils.rnn.pad_sequence(shrink_labels, batch_first = True)
            labels_mask = (labels >= 0).float()
            one_hot_labels = F.one_hot(labels, num_classes=self.config.vocab_size)
        else:
            weights_labels =   labels_mask /  labels_mask.sum(axis = 1)[:,None] #weights_logits.sum(axis = 1)[:,None] *         
       

        if labels is not None:
            self.count += 1
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
            
            loss, index_logits, index_labels, wtransport = batched_ottc_loss_bucketized(logits, one_hot_labels, weights_logits, weights_labels, amask = None, bmask = labels_mask, euclidian = self.euclidian, jsd = self.jsd)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states = (weights_logits, output_mask, weights_labels) , attentions= (index_logits, index_labels, wtransport)
        ) 
