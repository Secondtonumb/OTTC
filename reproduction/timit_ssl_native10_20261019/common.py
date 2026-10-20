"""Fresh OTTC task training from original SSL, with a trainable 10ms CNN tail."""
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT/'reproduction'),str(ROOT)]
from reproduction.native_stride_ottc_20261018.common import (
    sha,write,atomic_weights,load_data,Collator,topology,scorer,phones_from_ids,normalize)
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC,wvOTTCConfig

SSL=ROOT/'reproduction/models/wav2vec2-large-lv60'
SSL_SHA='d3c26ae7322ea631b59128407c0913b31c0d7d2c2a7a5f4375144a5b2422ef04'


class ScratchOTTC(wvForOTTC):
    def create_attention_mask_from_input_sequence(self,lengths):
        # Same mask as upstream; choose the input device rather than hard-code CUDA.
        return (torch.arange(1,int(lengths.max())+1,device=lengths.device)[None,:] <= lengths[:,None]).float()


def source_hashes():
    paths=[HERE/name for name in ['common.py','train.py','evaluate.py','graphs.py','profile_recipe.py','validate.py','run_one.pbs']]+[ROOT/'ottc/models/sequence/wav2vec2_ottc.py',
        ROOT/'ottc/ops_loss/ot_loss.py',ROOT/'reproduction/native_stride_ottc_20261018/common.py',
        ROOT/'reproduction/zapformer_ifmdd_topology_ottc_20261013/vendor/utils/subphonetic_topology.py',
        ROOT/'reproduction/libri100_singlehead_20261014/train.py',ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',
        ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py',
        ROOT/'reproduction/ifmdd_k2_alignment.py']
    return {str(p):sha(p) for p in sorted(paths)}


def matrix():
    return [f'wav2vec2__timit__10ms__{head}__{cost}' for head in ['mono','s3t2ss']
            for cost in ['jsd_upstream','ce','jsd_standard']]


def recipe(name):
    assert name in matrix()
    head,cost=name.split('__')[-2:]
    return dict(name=name,experiment_family=HERE.name,backbone='wav2vec2',corpus='timit',frame_ms=10,head=head,cost=cost,
        epochs=40,global_batch=64,learning_rate=2e-4,warmup_steps=500,seed=42,weight_decay=.005,
        initialization='original facebook/wav2vec2-large-lv60 SSL encoder; fresh random OTTC heads; no supervised OTTC parent',
        from_scratch_definition='User definition: SSL-initialized encoder, native10ms trainable last CNN layer and newly initialized trainable task heads',
        ssl_checkpoint=str(SSL),ssl_weights_sha256=SSL_SHA,ssl_revision='0cde644b64dac88d8416bec1c92a4099b850ba0b',
        upsample=False,native_conv_stride=[5,2,2,2,2,2,1],step_offset=0,
        forward_type='regular',mass_head_training='trainable for all40 epochs; no regfrozen schedule',
        bins_metric={'ce':'crossentropy','jsd_upstream':'jsd','jsd_standard':'jsd_standard'}[cost],
        objective='unchanged upstream OTTC transport with selected CE/upstream-JSD/standard-JSD cost',
        topology='s3_t2_star_star' if head=='s3t2ss' else None,
        target_mass='state_uniform' if head=='s3t2ss' else 'phone_uniform',
        freeze_cnn_first_six=True,train_cnn_last_layer=True,train_transformer=True,train_all_task_heads=True,
        checkpoint_policy='atomic independently named weights every epoch; retain all40; full latest resume plus best-dev; final5/final10 averages',
        source_sha256=source_hashes())


def configure_model(model,config,vocab):
    topo=topology(vocab) if config['head']=='s3t2ss' else None
    if topo:
        # New task, new state head. Do not copy a supervised phone head.
        head=torch.nn.Linear(model.lm_head.in_features,len(topo['labels']))
        model._init_weights(head)
        model.lm_head=head;model.config.vocab_size=len(topo['labels'])
    cnn=model.wav2vec2.feature_extractor
    assert list(model.wav2vec2.config.conv_stride)==[5,2,2,2,2,2,2]
    model.wav2vec2.config.conv_stride=list(config['native_conv_stride'])
    cnn.conv_layers[-1].conv.stride=(1,)
    model.freeze_feature_encoder()
    cnn.conv_layers[-1].requires_grad_(True)
    # Earlier layers need neither weight nor input gradients; last layer's
    # trainable parameters still create a differentiable path to Transformer.
    cnn._requires_grad=False
    model.lm_head.requires_grad_(True);model.lm_weight.requires_grad_(True)
    model.wave2vec2_config=model.wav2vec2.config
    model.config.native_frame_ms=10;model.config.native_conv_stride=list(config['native_conv_stride'])
    model.config.upsample=False;model.config.topology=config['topology']
    model.step=-1
    assert not hasattr(model,'wav2vec2_freeze')
    assert math.prod(model.wave2vec2_config.conv_stride)==160
    assert all(not p.requires_grad for layer in cnn.conv_layers[:-1] for p in layer.parameters())
    assert all(p.requires_grad for p in cnn.conv_layers[-1].parameters())
    assert all(p.requires_grad for p in model.wav2vec2.encoder.parameters())
    assert all(p.requires_grad for head in [model.lm_head,model.lm_weight] for p in head.parameters())
    return model,topo


def build_model(config,vocab,steps_per_epoch):
    model=ScratchOTTC(wvOTTCConfig(wav2vec2_path=str(SSL),vocab_size=42,
        bins_metric=config['bins_metric'],forward_type='regular',wav2vec2_output_dropout=.1))
    return configure_model(model,config,vocab)


def parameter_manifest(model):
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                cnn_layers=[dict(layer=i,stride=list(layer.conv.stride),
                    parameters={n:dict(shape=list(p.shape),trainable=p.requires_grad) for n,p in layer.named_parameters()})
                    for i,layer in enumerate(model.wav2vec2.feature_extractor.conv_layers)],
                heads={name:{n:dict(shape=list(p.shape),trainable=p.requires_grad) for n,p in getattr(model,name).named_parameters()} for name in ['lm_head','lm_weight']})
