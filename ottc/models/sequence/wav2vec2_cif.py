"""Resumable Wav2Vec2 + CIF phone model; optional auxiliary CTC branch."""
from transformers import PretrainedConfig, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model
import torch
from torch import nn

from .cif import CIFPhoneHead


class wvCIFConfig(PretrainedConfig):
    model_type = 'wav2vec2-cif'

    def __init__(self, encoder_config=None, vocab_size=42, decoder_dim=256,
                 decoder_layers=2, decoder_heads=4, decoder_ffn=2048, dropout=0.1,
                 quantity_weight=1.0, ctc_weight=0.0, ottc_weight=0.0, label_smoothing=0.1, blank_id=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.vocab_size = vocab_size
        for name, value in dict(decoder_dim=decoder_dim, decoder_layers=decoder_layers,
                                decoder_heads=decoder_heads, decoder_ffn=decoder_ffn,
                                dropout=dropout, quantity_weight=quantity_weight,
                                ctc_weight=ctc_weight, ottc_weight=ottc_weight, label_smoothing=label_smoothing,
                                blank_id=blank_id).items():
            setattr(self, name, value)
        self.target_convention = 'phone occurrences only; no EOS; private BOS input embedding'
        self.loss_reduction = '(sum CE + quantity_weight * sum L1 + ctc_weight * sum CTC) / target phone count + ottc_weight * native upstream OTTC sum / utterances'
        self.ottc_metric = 'original upstream jsd=True, not standard JSD'


class wvForCIF(PreTrainedModel):
    config_class = wvCIFConfig
    base_model_prefix = 'wav2vec2'

    def __init__(self, config):
        super().__init__(config)
        if config.encoder_config is None:
            raise ValueError('encoder_config is required; use from_encoder_pretrained for a new run')
        self.wav2vec2 = Wav2Vec2Model(Wav2Vec2Config(**config.encoder_config))
        self.dropout = nn.Dropout(config.dropout)
        self.phone_head = CIFPhoneHead(self.wav2vec2.config.hidden_size, config.vocab_size,
            **{key: getattr(config, key) for key in ('decoder_dim', 'decoder_layers', 'decoder_heads',
                'decoder_ffn', 'dropout', 'quantity_weight', 'ctc_weight', 'ottc_weight', 'label_smoothing', 'blank_id')})

    @classmethod
    def from_encoder_pretrained(cls, path, **kwargs):
        encoder = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
        net = cls(wvCIFConfig(encoder_config=encoder.config.to_dict(), **kwargs))
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
        tokens, cif = self.phone_head.recognize(hidden, valid)
        return dict(tokens=tokens, cif=cif, frame_lengths=valid.sum(-1))
