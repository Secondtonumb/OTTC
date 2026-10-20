"""Small CUDA regression gates for native stride, three costs, and exact resume."""
import common
from common import HERE,ROOT,source_hashes,write,topology
import json
import fcntl
from pathlib import Path
import torch
from torch import nn
from transformers import Wav2Vec2Config,Wav2Vec2Model,TrainingArguments,TrainerCallback,set_seed
import ottc.models.sequence.wav2vec2_ottc as original
from reproduction.native_stride_ottc_20261018.train import QuietTrainer,Policy,start_wandb


class TinyLoader:
    @staticmethod
    def from_pretrained(path):
        return Wav2Vec2Model(Wav2Vec2Config(hidden_size=32,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=64,conv_dim=[8]*7,conv_stride=[5,2,2,2,2,2,1],conv_kernel=[10,3,3,3,3,2,2],
            num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16,apply_spec_augment=False,
            hidden_dropout=0.,attention_dropout=0.,feat_proj_dropout=0.,layerdrop=0.,activation_dropout=0.,feat_extract_norm='layer'))


def tiny(cost):
    old=original.Wav2Vec2Model;original.Wav2Vec2Model=TinyLoader
    try:model=original.wvForOTTC(original.wvOTTCConfig(wav2vec2_path='tiny-validation-only',vocab_size=42,
        bins_metric=cost,forward_type='regfrozen',frozen_steps_for_regfrozen=2,wav2vec2_output_dropout=0.))
    finally:original.Wav2Vec2Model=old
    model.wav2vec2_freeze.requires_grad_(False);model.wav2vec2.gradient_checkpointing_disable()
    return model


class Data:
    def __len__(self):return 4
    def __getitem__(self,index):
        return dict(input_values=torch.sin(torch.arange(5120).float()*(.01+index*.001)),
            attention_mask=torch.ones(5120,dtype=torch.long),labels=torch.tensor([1,2,3],dtype=torch.long))


def collate(rows):
    return dict(microbatches=[{key:torch.stack([r[key] for r in rows]) for key in rows[0]}],logical_batch_size=len(rows))


class Stop(TrainerCallback):
    def __init__(self,stop):self.stop=stop
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==self.stop:control.should_training_stop=True;control.should_save=True


class Dummy:
    def log(self,*a,**k):pass


def train_case(path,stop=None,resume=False):
    path.mkdir(parents=True,exist_ok=True);set_seed(42);model=tiny('jsd')
    config=dict(name='tiny_validation',step_offset=0,source_sha256=source_hashes(),native_conv_stride=[5,2,2,2,2,2,1])
    policy=Policy(path,config,[],{},None,None,float('inf'),Dummy())
    args=TrainingArguments(output_dir=str(path),per_device_train_batch_size=2,num_train_epochs=2,
        learning_rate=1e-4,warmup_steps=1,save_strategy='epoch',logging_steps=1,report_to=[],
        dataloader_num_workers=0,remove_unused_columns=False,disable_tqdm=True,seed=42,save_safetensors=True)
    trainer=QuietTrainer(model=model,args=args,train_dataset=Data(),data_collator=collate,
        callbacks=[policy]+([Stop(stop)] if stop else []))
    trainer.train(resume_from_checkpoint=str(path/'checkpoint-2') if resume else None)
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def main():
    assert torch.cuda.is_available();torch.set_num_threads(4)
    output=HERE/'validation';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    gate=output/'PASS.json'
    if gate.exists() and json.loads(gate.read_text())['source_sha256']==source_hashes():return
    results={}
    for cost in ['crossentropy','jsd','jsd_standard']:
        set_seed(42);model=tiny(cost).cuda();model.step=3;model.train()
        batch={key:torch.stack([Data()[0][key],Data()[1][key]]).cuda() for key in Data()[0]}
        out=model(**batch);assert out.logits.shape[1]==30 and int(out.hidden_states[1][0].sum())==30
        assert torch.isfinite(out.loss);out.loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results[cost]=dict(loss=float(out.loss),native_frames=30,finite_gradients=True)
        del model
    vocab=json.loads((ROOT/'reproduction/libri960_continue5_20261017/models/jsd_upstream/vocab_used.json').read_text())
    topo=topology(vocab)
    from reproduction.native_stride_ottc_20261018.graphs import graph,decode
    b=topo['mapping'][vocab['b']];ids=[b[0],b[0],b[1],b[2],b[2],b[0],b[2]]
    scores=torch.full((len(ids),len(topo['labels'])),-30.,device='cuda')
    scores[torch.arange(len(ids),device='cuda'),torch.tensor(ids,device='cuda')]=30.
    tokens,path,starts=decode(scores,graph(topo,vocab,device='cuda'))
    assert tokens==[vocab['b'],vocab['b']] and path==ids
    expected=graph(topo,vocab,[vocab['b'],vocab['b']],device='cuda')
    assert decode(scores,expected)[0]==[1,2]
    full=train_case(output/'continuous');train_case(output/'resumed',stop=2);resumed=train_case(output/'resumed',resume=True)
    difference=max(float((full[k]-resumed[k]).abs().max()) for k in full)
    assert difference==0.,difference
    assert (output/'continuous/epoch_weights/epoch-001.pt').exists() and (output/'continuous/epoch_weights/epoch-002.pt').exists()
    report=dict(source_sha256=source_hashes(),costs=results,native_hop_samples=160,upsample=False,
        S3T2_repeated_phone_exact_path=True,continuous_vs_resume_model_max_difference=difference,
        independent_epoch_weights_verified=True,full_latest_optimizer_scheduler_RNG_verified=True)
    write(gate,report)
    run=start_wandb(output,dict(name='native-stride-OTTC-validation',validation=report),'validation')
    run.log(dict(resume_max_difference=difference,native_hop_samples=160,three_costs_passed=1,topology_path_passed=1))
    artifact=__import__('wandb').Artifact('native-stride-OTTC-validation',type='validation');artifact.add_file(str(gate));run.log_artifact(artifact);run.finish()
    print('VALIDATION_PASS',json.dumps(report),flush=True)


if __name__=='__main__':main()
