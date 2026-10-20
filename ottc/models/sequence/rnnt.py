"""Phone RNN-T with SpeechBrain's joint and TorchAudio's loss/container.

The encoder is supplied by the caller. No CTC collapse, target-length scaling,
EOS, external LM, or reference input is used during recognition.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import rnnt_loss
from torchaudio.models import RNNT
from speechbrain.nnet.transducer.transducer_joint import Transducer_joint


class AcousticProjection(nn.Module):
    def __init__(self, input_dim, joint_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, joint_dim)

    def forward(self, input, lengths):
        return self.projection(input), lengths


class PhonePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers, joint_dim, dropout):
        super().__init__()
        # Blank is also the initial predictor input. It is never a loss target.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.)
        self.projection = nn.Linear(hidden_dim, joint_dim)

    def forward(self, input, lengths, state=None):
        hx = None if state is None else (state[0][0], state[0][1])
        hidden, (h, c) = self.lstm(self.embedding(input.long()), hx)
        return self.projection(hidden), lengths, [[h, c]]


class SpeechBrainJoiner(nn.Module):
    def __init__(self, joint_dim, vocab_size):
        super().__init__()
        self.joint = Transducer_joint(joint='sum', nonlinearity=nn.Tanh)
        self.output = nn.Linear(joint_dim, vocab_size)

    def forward(self, source_encodings, source_lengths, target_encodings, target_lengths):
        # SpeechBrain's joint returns hidden features, not vocabulary logits.
        hidden = self.joint(source_encodings[:, :, None], target_encodings[:, None])
        return self.output(hidden), source_lengths, target_lengths


class RNNTPhoneHead(nn.Module):
    def __init__(self, input_dim, vocab_size, *, embedding_dim=256, predictor_dim=256,
                 predictor_layers=2, joint_dim=256, dropout=0.1, blank_id=0,
                 ottc_weight=0., max_symbols_per_frame=10):
        super().__init__()
        if min(input_dim, vocab_size, embedding_dim, predictor_dim, predictor_layers,
               joint_dim, max_symbols_per_frame) < 1:
            raise ValueError('dimensions and decoding cap must be positive')
        if not 0 <= blank_id < vocab_size or not 0 <= dropout < 1:
            raise ValueError('invalid blank id or dropout')
        if not math.isfinite(ottc_weight) or ottc_weight < 0:
            raise ValueError('OTTC weight must be finite and nonnegative')
        self.vocab_size, self.blank_id = vocab_size, blank_id
        self.ottc_weight, self.max_symbols_per_frame = ottc_weight, max_symbols_per_frame
        self.transducer = RNNT(AcousticProjection(input_dim, joint_dim),
            PhonePredictor(vocab_size, embedding_dim, predictor_dim, predictor_layers, joint_dim, dropout),
            SpeechBrainJoiner(joint_dim, vocab_size))
        self.ottc_head = nn.Linear(input_dim, vocab_size) if ottc_weight else None
        self.ottc_mass_head = nn.Linear(input_dim, 1) if ottc_weight else None

    def _lengths(self, features, valid_mask):
        if features.ndim != 3 or valid_mask.shape != features.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError('expected features [B,T,H] and boolean valid_mask [B,T]')
        if not len(features) or (valid_mask.sum(-1) == 0).any():
            raise ValueError('nonempty batch and positive frame lengths required')
        if ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
            raise ValueError('valid frames must be right-padded prefixes')
        return valid_mask.sum(-1)

    def forward(self, features, valid_mask, labels):
        frame_lengths = self._lengths(features, valid_mask)
        if labels.ndim != 2 or len(labels) != len(features):
            raise ValueError('labels must be [B,U], padded with -100')
        mask = labels != -100
        if ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError('targets must be right-padded prefixes')
        if ((labels[mask] < 0) | (labels[mask] >= self.vocab_size) | (labels[mask] == self.blank_id)).any():
            raise ValueError('targets must be vocabulary phones without blank')
        lengths = mask.sum(-1)
        if (lengths == 0).any():
            raise ValueError('empty phone targets are not supported by this recipe')
        targets = labels[:, :int(lengths.max())].masked_fill(~mask[:, :int(lengths.max())], self.blank_id)
        predictor_input = F.pad(targets, (1, 0), value=self.blank_id)
        logits, output_lengths, _, _ = self.transducer(features, frame_lengths, predictor_input, lengths + 1)
        with torch.autocast(features.device.type, enabled=False):
            # TorchAudio requires B,T,U+1,V contiguous FP32 logits and int32 lengths/targets.
            rnnt_sum = rnnt_loss(logits.float().contiguous(), targets.int().contiguous(),
                output_lengths.int().contiguous(), lengths.int().contiguous(),
                blank=self.blank_id, reduction='sum', fused_log_softmax=True)
            ottc_sum = rnnt_sum.new_zeros(())
            if self.ottc_head is not None:
                if not features.is_cuda:
                    raise ValueError('frozen upstream OTTC requires CUDA; use GPU for the auxiliary objective')
                from reproduction.zapformer_s3t2_native_20261012.archived_ot_loss import batched_ottc_loss_bucketized
                target_mask = torch.arange(targets.shape[1], device=labels.device)[None] < lengths[:, None]
                mass = self.ottc_mass_head(features.float()).squeeze(-1).masked_fill(~valid_mask, -torch.inf).softmax(-1)
                target_mass = target_mask.float() / lengths[:, None]
                one_hot = F.one_hot(targets, self.vocab_size).float()
                ottc_sum = batched_ottc_loss_bucketized(self.ottc_head(features.float()), one_hot,
                    mass, target_mass, amask=valid_mask, bmask=target_mask, jsd=True)[0]
            loss = rnnt_sum / lengths.sum() + self.ottc_weight * ottc_sum / len(features)
        return dict(loss=loss, rnnt_sum=rnnt_sum, ottc_sum=ottc_sum, token_count=lengths.sum())

    @torch.no_grad()
    def recognize(self, features, valid_mask):
        """Greedy RNN-T: blank advances time; a phone updates only predictor state.

        Repeated phones and multiple emissions in one frame are preserved. A
        finite per-frame cap bounds pathological untrained decoders; cap hits
        are returned and reported, never hidden or tuned on the test set.
        """
        if self.training:
            raise ValueError('call eval() before recognition')
        lengths = self._lengths(features, valid_mask)
        acoustic, _ = self.transducer.transcribe(features, lengths)
        tokens, frames, caps = [], [], []
        one = torch.ones(1, dtype=torch.long, device=features.device)
        for b, length in enumerate(lengths.tolist()):
            previous = torch.full((1, 1), self.blank_id, dtype=torch.long, device=features.device)
            predicted, _, state = self.transducer.predict(previous, one, None)
            sequence, times, hits = [], [], 0
            for t in range(length):
                for _ in range(self.max_symbols_per_frame):
                    logits, _, _ = self.transducer.join(acoustic[b:b+1, t:t+1], one, predicted, one)
                    token = int(logits[0, 0, 0].float().argmax())
                    if token == self.blank_id:
                        break
                    sequence.append(token)
                    times.append(t)
                    previous.fill_(token)
                    predicted, _, state = self.transducer.predict(previous, one, state)
                else:
                    hits += 1
            tokens.append(sequence)
            frames.append(times)
            caps.append(hits)
        return dict(tokens=tokens, emission_frames=frames, frame_lengths=lengths, cap_hits=caps)
