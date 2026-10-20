"""Preserve the original single-head OTTC implementation over a real WavLM encoder.

The original attribute name `wav2vec2` remains a checkpoint compatibility slot.
Both that slot and the regfrozen snapshot contain WavLMModel instances.
"""
import bootstrap
from contextlib import contextmanager
from pathlib import Path
import torch
from transformers import WavLMModel
from transformers.models.wavlm.modeling_wavlm import WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder
import ottc.models.sequence.wav2vec2_ottc as original


class WavLMOTTCConfig(original.wvOTTCConfig):
    model_type='wavlm-ottc'


class EncoderLoader:
    @staticmethod
    def from_pretrained(path):
        # The official 2021 checkpoint uses legacy weight_norm names. Torch 2.6
        # exposes the same two tensors through parametrizations; map names only.
        state=torch.load(Path(path)/'pytorch_model.bin',map_location='cpu',mmap=True,weights_only=False)
        state={key.replace('.weight_g','.parametrizations.weight.original0')
                   .replace('.weight_v','.parametrizations.weight.original1'):value for key,value in state.items()}
        model,info=WavLMModel.from_pretrained(path,state_dict=state,local_files_only=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('WavLM pretrained loading mismatch: '+repr(info))
        return model


@contextmanager
def wavlm_loader():
    # Scoped to construction in this independent single-process entry.
    # No repository source or checkpoint is changed.
    previous=original.Wav2Vec2Model
    original.Wav2Vec2Model=EncoderLoader
    try:yield
    finally:original.Wav2Vec2Model=previous


class WavLMForOTTC(original.wvForOTTC):
    config_class=WavLMOTTCConfig

    def __init__(self,config):
        with wavlm_loader():super().__init__(config)
        assert isinstance(self.wav2vec2,WavLMModel)
        if hasattr(self,'wav2vec2_freeze'):assert isinstance(self.wav2vec2_freeze,WavLMModel)

    def _set_gradient_checkpointing(self,module,value=False):
        if isinstance(module,(WavLMEncoder,WavLMEncoderStableLayerNorm,WavLMFeatureEncoder)):
            module.gradient_checkpointing=value
