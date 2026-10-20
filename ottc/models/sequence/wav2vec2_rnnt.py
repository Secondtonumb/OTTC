"""Existing OTTC Wav2Vec2 encoder with an independent phone RNN-T head."""
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model

from .rnnt import RNNTPhoneHead


class wvRNNTConfig(PretrainedConfig):
    model_type = 'wav2vec2-rnnt'

    def __init__(self, encoder_config=None, vocab_size=42, embedding_dim=256,
                 predictor_dim=256, predictor_layers=2, joint_dim=256, dropout=0.1,
                 blank_id=0, ottc_weight=0., max_symbols_per_frame=10, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config, self.vocab_size = encoder_config, vocab_size
        for key, value in dict(embedding_dim=embedding_dim, predictor_dim=predictor_dim,
                predictor_layers=predictor_layers, joint_dim=joint_dim, dropout=dropout,
                blank_id=blank_id, ottc_weight=ottc_weight,
                max_symbols_per_frame=max_symbols_per_frame).items():
            setattr(self, key, value)
        self.loss_reduction = 'RNNT sum / target phones + weight * upstream OTTC sum / utterances'
        self.target_convention = 'phone occurrences; blank=0 also starts predictor; no EOS; -100 padding'
        self.joint = 'SpeechBrain Transducer_joint(sum, Tanh), separate vocabulary projection'


class wvForRNNT(PreTrainedModel):
    config_class = wvRNNTConfig
    base_model_prefix = 'wav2vec2'

    def __init__(self, config):
        super().__init__(config)
        if config.encoder_config is None:
            raise ValueError('encoder_config required; use from_encoder_pretrained for a new run')
        self.wav2vec2 = Wav2Vec2Model(Wav2Vec2Config(**config.encoder_config))
        self.dropout = nn.Dropout(config.dropout)
        self.phone_head = RNNTPhoneHead(self.wav2vec2.config.hidden_size, config.vocab_size,
            **{k: getattr(config, k) for k in ('embedding_dim', 'predictor_dim', 'predictor_layers',
                'joint_dim', 'dropout', 'blank_id', 'ottc_weight', 'max_symbols_per_frame')})

    @classmethod
    def from_encoder_pretrained(cls, path, **kwargs):
        encoder = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
        net = cls(wvRNNTConfig(encoder_config=encoder.config.to_dict(), **kwargs))
        net.wav2vec2 = encoder
        return net

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def encode(self, input_values, attention_mask):
        hidden = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True).last_hidden_state
        valid = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        return self.dropout(hidden), valid

    def forward(self, input_values, attention_mask, labels):
        hidden, valid = self.encode(input_values, attention_mask)
        return self.phone_head(hidden, valid, labels)

    @torch.no_grad()
    def recognize(self, input_values, attention_mask):
        hidden, valid = self.encode(input_values, attention_mask)
        return self.phone_head.recognize(hidden, valid)
