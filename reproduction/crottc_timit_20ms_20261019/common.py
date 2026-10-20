"""TIMIT CROTTC with frozen IF-MDD augmentation and CR semantics."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT/'reproduction'),str(ROOT)]
from reproduction.native_stride_ottc_20261018.common import (
    sha,write,atomic_weights,load_data,Collator,scorer,phones_from_ids,normalize)
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC,wvOTTCConfig
from reproduction.crottc_timit_20ms_20261019.vendor.legacy_cr import LegacyCR

SSL=ROOT/'reproduction/models/wav2vec2-large-lv60'
SSL_SHA='d3c26ae7322ea631b59128407c0913b31c0d7d2c2a7a5f4375144a5b2422ef04'


def source_hashes():
    paths=list(HERE.glob('*.py'))+list((HERE/'vendor').glob('*'))+[HERE/'run_one.pbs',
        ROOT/'ottc/models/sequence/wav2vec2_ottc.py',ROOT/'ottc/ops_loss/ot_loss.py',
        ROOT/'reproduction/native_stride_ottc_20261018/common.py',
        ROOT/'reproduction/libri100_singlehead_20261014/train.py',
        ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',
        ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py',
        ROOT/'reproduction/ifmdd_k2_alignment.py']
    return {str(p):sha(p) for p in sorted(paths) if p.is_file()}


def matrix():
    return [f'wav2vec2__timit__20ms__mono__crottc_{cost}' for cost in ['jsd_upstream','ce','jsd_standard']]


def recipe(name):
    assert name in matrix(),name
    cost=name.split('__')[-1].removeprefix('crottc_')
    return dict(name=name,experiment_family=HERE.name,backbone='wav2vec2',corpus='timit',frame_ms=20,head='mono',cost=cost,
        epochs=40,global_batch=64,learning_rate=2e-4,warmup_steps=500,seed=42,weight_decay=.005,
        initialization='original facebook/wav2vec2-large-lv60 SSL encoder; fresh random OTTC heads; no supervised OTTC parent',
        ssl_checkpoint=str(SSL),ssl_weights_sha256=SSL_SHA,ssl_revision='0cde644b64dac88d8416bec1c92a4099b850ba0b',
        native_conv_stride=[5,2,2,2,2,2,2],upsample=False,topology=None,target_mass='phone_uniform',step_offset=0,
        forward_type='regular',freeze_cnn=True,train_transformer=True,train_all_task_heads=True,
        mass_head_training='trainable all40 epochs, matching IF-MDD CROTTC; no regfrozen switch',
        bins_metric={'ce':'crossentropy','jsd_upstream':'jsd','jsd_standard':'jsd_standard'}[cost],cr_loss_weight=.1,
        objective='(0.5*OTTC_sum_both_views + 0.1*legacy_CR_mean_global_batch) / source_utterance_batch_size',
        cr_reduction='exact frozen IF-MDD method; denominator B*T*C - number_of_padding_frames; global logical batch aggregation before optimizer normalization',
        augmentation=dict(shared_speed_percent=[95,100,105],DropFreq=dict(drop_freq_low=1e-14,drop_freq_high=1,drop_freq_count_low=1,drop_freq_count_high=3,drop_freq_width=.10,epsilon=1e-12),
            DropChunk=dict(drop_length_low=1000,drop_length_high=3000,drop_count_low=1,drop_count_high=3),
            views=2,training_only=True,implementation='speechbrain1.0.3; original IF-MDD recipe; no timewarp'),
        checkpoint_policy='atomic weights each completed epoch; retain all40, best-dev and full latest; final5/final10 averages',
        source_sha256=source_hashes())


class SpeechViews:
    """Plain object: augmentation kernels are not acoustic-model state."""
    def __init__(self,device):
        from speechbrain.augment.time_domain import DropFreq,DropChunk,SpeedPerturb
        from speechbrain.augment.augmenter import Augmenter
        self.speed=SpeedPerturb(orig_freq=16000,speeds=[95,100,105],device=str(device))
        self.augment=Augmenter(augmentations=[DropFreq(drop_freq_low=1e-14,drop_freq_high=1,
            drop_freq_count_low=1,drop_freq_count_high=3,drop_freq_width=.10,epsilon=1e-12),
            DropChunk(drop_length_low=1000,drop_length_high=3000,drop_count_low=1,drop_count_high=3)])

    @torch.no_grad()
    def __call__(self,batch):
        audio=batch['input_values'];relative=batch['attention_mask'].sum(-1).float()/audio.shape[1]
        audio=self.speed(audio)
        first,length1=self.augment(audio.clone(),lengths=relative)
        second,length2=self.augment(audio.clone(),lengths=relative)
        assert first.shape==second.shape and torch.equal(length1,length2)
        length=(length1*first.shape[1]).round().long().clamp(max=first.shape[1])
        mask=torch.arange(first.shape[1],device=audio.device)[None,:]<length[:,None]
        # Filters may leak into padding; remove it before the encoder.
        first=first.masked_fill(~mask,0.);second=second.masked_fill(~mask,0.)
        return dict(input_values=torch.cat([first,second]),attention_mask=mask.long().repeat(2,1),
            labels=batch['labels'].repeat(2,1),cr_relative_lengths=length1),int(self.speed.speeds[self.speed.samp_index])


def pair_chunks(model,batch,augmenter,micro):
    """Augment once per logical batch; split paired views on one padded grid."""
    views,speed=augmenter(batch);size=len(batch['labels'])
    frames=int(model._get_feat_extract_output_lengths(model.wave2vec2_config,torch.tensor(views['input_values'].shape[1])))
    valid=(views['cr_relative_lengths']*frames).long()
    denominator=size*frames*model.config.vocab_size-(frames-valid).sum()
    chunks=[]
    for start in range(0,size,micro):
        stop=min(start+micro,size)
        part={k:torch.cat([v[start:stop],v[size+start:size+stop]]) for k,v in views.items() if k!='cr_relative_lengths'}
        part.update(cr_views=True,cr_relative_lengths=views['cr_relative_lengths'][start:stop],cr_global_denominator=denominator)
        chunks.append(part)
    return chunks,dict(speed_percent=speed,source_utterances=size,encoder_utterances=2*size,frames=frames,
        cr_denominator=float(denominator),microbatch=micro)


class CROTTC(wvForOTTC):
    def create_attention_mask_from_input_sequence(self,lengths):
        return (torch.arange(1,int(lengths.max())+1,device=lengths.device)[None,:]<=lengths[:,None]).float()

    def regular_forward(self,input_values,attention_mask=None,return_dict=True,labels=None):
        # Preserve the complete logical batch's padded grid across paired chunks.
        hidden=self.wav2vec2(input_values,attention_mask=attention_mask,
            output_hidden_states=True,return_dict=True)[0]
        hidden=self.dropout(hidden)
        logits=self.lm_head(hidden).float();mass_logits=self.lm_weight(hidden).squeeze(-1).float()
        if attention_mask is None:attention_mask=torch.ones_like(input_values,dtype=torch.long)
        lengths=self._get_feat_extract_output_lengths(self.wave2vec2_config,attention_mask.sum(-1)).long()
        mask=(torch.arange(hidden.shape[1],device=hidden.device)[None,:]<lengths[:,None]).float()
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            mass=mass_logits.masked_fill(mask==0,-torch.inf).softmax(-1)
        return logits,mass,mask,hidden

    def forward(self,input_values,attention_mask=None,return_dict=None,labels=None,
                cr_views=False,cr_relative_lengths=None,cr_global_denominator=None):
        if not cr_views:
            return super().forward(input_values,attention_mask,return_dict,labels)
        assert self.training and labels is not None and len(labels)%2==0
        output=super().forward(input_values,attention_mask,True,labels)
        # Transport and KL use FP32 even when the encoder uses BF16.
        with torch.autocast(device_type=input_values.device.type,enabled=False):
            log1,log2=output.logits.float().log_softmax(-1).chunk(2)
            legacy=LegacyCR()._compute_cr_loss(log1,log2,cr_relative_lengths)
            B,T,C=log1.shape
            local_denominator=B*T*C-(T-(cr_relative_lengths*T).long()).sum()
            cr=legacy*local_denominator/cr_global_denominator
            transport=.5*output.loss
            output.loss=transport+self.config.cr_loss_weight*cr
        self.last_loss_components=dict(ottc_sum=transport.detach(),cr_global_contribution=cr.detach(),
            weighted_cr_global_contribution=(self.config.cr_loss_weight*cr).detach(),total_sum=output.loss.detach())
        return output


def build_model(config,vocab,steps_per_epoch):
    model=CROTTC(wvOTTCConfig(wav2vec2_path=str(SSL),vocab_size=42,bins_metric=config['bins_metric'],
        forward_type='regular',wav2vec2_output_dropout=.1))
    model.config.cr_loss_weight=config['cr_loss_weight']
    model.freeze_feature_encoder()
    model.config.native_frame_ms=20;model.config.native_conv_stride=config['native_conv_stride'];model.config.upsample=False
    model.step=-1
    assert list(model.wave2vec2_config.conv_stride)==config['native_conv_stride']
    assert math.prod(model.wave2vec2_config.conv_stride)==320
    assert not hasattr(model,'wav2vec2_freeze')
    return model,None


def parameter_manifest(model):
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        cnn_all_frozen=all(not p.requires_grad for p in model.wav2vec2.feature_extractor.parameters()),
        transformer_all_trainable=all(p.requires_grad for p in model.wav2vec2.encoder.parameters()),
        heads_all_trainable=all(p.requires_grad for h in [model.lm_head,model.lm_weight] for p in h.parameters()),
        native_stride=list(model.wave2vec2_config.conv_stride))
