"""Wav2Vec2 + blank-free OTTC transducer with a separately learned count."""
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model

from .ottc_transducer import OTTCTransducerHead


class wvOTTCTransducerConfig(PretrainedConfig):
    model_type = 'wav2vec2-ottc-transducer-blankfree'

    def __init__(self, encoder_config=None, phone_ids=None, external_vocab_size=42,
                 embedding_dim=256, predictor_dim=256, predictor_layers=2, joint_dim=256,
                 dropout=.1, length_weight=1., initial_phone_rate=.2, max_output_tokens=512, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.phone_ids = list(range(1,41)) if phone_ids is None else list(phone_ids)
        for name,value in dict(external_vocab_size=external_vocab_size, embedding_dim=embedding_dim,
            predictor_dim=predictor_dim, predictor_layers=predictor_layers, joint_dim=joint_dim,
            dropout=dropout, length_weight=length_weight, initial_phone_rate=initial_phone_rate,
            max_output_tokens=max_output_tokens).items():
            setattr(self,name,value)
        self.vocab_size = len(self.phone_ids)
        self.blank_free = True
        self.loss_reduction = '(original upstream JSD joint-OT sum + length_weight * log-count Huber sum) / utterances'
        self.decode = 'predict count from audio; uniform target bins; original transport; greedy minimum expected upstream JSD; no collapse'


class wvForOTTCTransducer(PreTrainedModel):
    config_class = wvOTTCTransducerConfig
    base_model_prefix = 'wav2vec2'

    def __init__(self, config):
        super().__init__(config)
        if config.encoder_config is None:
            raise ValueError('encoder_config required; use from_encoder_pretrained for a fresh run')
        self.wav2vec2 = Wav2Vec2Model(Wav2Vec2Config(**config.encoder_config))
        self.dropout = nn.Dropout(config.dropout)
        self.phone_head = OTTCTransducerHead(self.wav2vec2.config.hidden_size, config.phone_ids,
            **{k:getattr(config,k) for k in ('external_vocab_size','embedding_dim','predictor_dim',
                'predictor_layers','joint_dim','dropout','length_weight','initial_phone_rate','max_output_tokens')})

    @classmethod
    def from_encoder_pretrained(cls, path, **kwargs):
        encoder = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
        net = cls(wvOTTCTransducerConfig(encoder_config=encoder.config.to_dict(), **kwargs))
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
