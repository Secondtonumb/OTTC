"""CPU/GPU gates for SSL identity, trainability, six costs and exact resume."""
import argparse
from contextlib import contextmanager
import gc
import fcntl
import inspect
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from transformers import Wav2Vec2Model,Wav2Vec2Config,TrainingArguments,TrainerCallback,set_seed
from common import HERE,ROOT,SSL,SSL_SHA,sha,recipe,matrix,build_model,configure_model,write,source_hashes,parameter_manifest,ScratchOTTC,load_data,topology
import ottc.models.sequence.wav2vec2_ottc as upstream


@contextmanager
def cpu_loss_allocations():
    # Test-only allocation adaptation, never used by production/GPU training.
    # The original function body/formula is reused byte-for-byte except for
    # its five literal CUDA allocations. The frozen source file is unchanged.
    original=upstream.batched_ottc_loss_bucketized
    source=inspect.getsource(original)
    namespace=dict(original.__globals__)
    exec(compile(source.replace(".to('cuda')",".to('cpu')"),'cpu_test_allocation_only','exec'),namespace)
    upstream.batched_ottc_loss_bucketized=namespace[original.__name__]
    try:yield
    finally:upstream.batched_ottc_loss_bucketized=original


def tested_forward(model,data,device):
    if device=='cpu':
        with cpu_loss_allocations():return model(**data)
    return model(**data)


def verify_ssl_encoder(model):
    source=torch.load(SSL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    checked=0
    for key,value in model.wav2vec2.state_dict().items():
        source_key=key if key in source else 'wav2vec2.'+key
        assert source_key in source,source_key
        assert torch.equal(value.detach().cpu(),source[source_key]),key
        checked+=1
    del source;gc.collect()
    return dict(tensors_checked=checked,all_equal=True,checkpoint='facebook/wav2vec2-large-lv60',expected_weights_sha256=SSL_SHA)


class TinyLoader:
    @staticmethod
    def from_pretrained(path):
        return Wav2Vec2Model(Wav2Vec2Config(hidden_size=32,num_hidden_layers=1,num_attention_heads=2,intermediate_size=64,
            conv_dim=[8]*7,conv_stride=[5,2,2,2,2,2,2],conv_kernel=[10,3,3,3,3,2,2],
            num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16,apply_spec_augment=False,
            hidden_dropout=0.,attention_dropout=0.,feat_proj_dropout=0.,layerdrop=0.,activation_dropout=0.,feat_extract_norm='layer'))


def tiny(config,vocab):
    original=upstream.Wav2Vec2Model;upstream.Wav2Vec2Model=TinyLoader
    try:model,topo=build_model(config,vocab,2)
    finally:upstream.Wav2Vec2Model=original
    model.dropout.p=0.;model.wav2vec2.gradient_checkpointing_disable()
    return model,topo


class Data:
    def __len__(self):return 4
    def __getitem__(self,i):
        return dict(input_values=torch.sin(torch.arange(5120).float()*(.01+i*.001)),
            attention_mask=torch.ones(5120,dtype=torch.long),labels=torch.tensor([1,2,3]))


def collate(rows):
    return dict(microbatches=[{k:torch.stack([r[k] for r in rows]) for k in rows[0]}],logical_batch_size=len(rows))


class Stop(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==2:control.should_training_stop=True;control.should_save=True


def train_case(path,config,vocab,resume=False,stop=False):
    from reproduction.timit_ssl_native10_20261019.train import QuietTrainer,Policy
    class Dummy:
        def log(self,*a,**k):pass
    set_seed(42);model,_=tiny(config,vocab)
    path.mkdir(parents=True,exist_ok=True)
    policy=Policy(path,config,[],vocab,None,None,float('inf'),Dummy())
    args=TrainingArguments(output_dir=str(path),per_device_train_batch_size=2,num_train_epochs=2,learning_rate=1e-4,
        warmup_steps=1,save_strategy='epoch',logging_steps=1,report_to=[],dataloader_num_workers=0,
        remove_unused_columns=False,disable_tqdm=True,seed=42,save_safetensors=True,use_cpu=True)
    trainer=QuietTrainer(model=model,args=args,train_dataset=Data(),data_collator=collate,callbacks=[policy]+([Stop()] if stop else []))
    with cpu_loss_allocations():trainer.train(resume_from_checkpoint=str(path/'checkpoint-2') if resume else None)
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--full-ssl',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    output=HERE/'validation';output.mkdir(exist_ok=True)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    gate=output/(args.device.upper()+('_FULL_SSL_PASS.json' if args.full_ssl else '_PASS.json'))
    if gate.exists() and json.loads(gate.read_text())['source_sha256']==source_hashes():
        print('VALIDATION_CACHE',gate,flush=True);return
    vocab=json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json').read_text())
    checks={}
    from reproduction.timit_ssl_native10_20261019.profile_recipe import gradient_checks
    from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
    assert evaluate.__module__=='reproduction.timit_ssl_native10_20261019.evaluate'
    for name in matrix():
        config=recipe(name);set_seed(42);model,topo=tiny(config,vocab);model.to(args.device).train()
        data=collate([Data()[0],Data()[1]])['microbatches'][0]
        if topo:data['labels']=torch.tensor([[s for token in row for s in topo['mapping'][int(token)]] for row in data['labels']])
        data={k:v.to(args.device) for k,v in data.items()}
        first=model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.detach().clone()
        last=model.wav2vec2.feature_extractor.conv_layers[-1].conv.weight.detach().clone()
        out=tested_forward(model,data,args.device);assert out.logits.shape[1]==30 and torch.isfinite(out.loss)
        out.loss.backward();norms=gradient_checks(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3);optimizer.step()
        assert torch.equal(first,model.wav2vec2.feature_extractor.conv_layers[0].conv.weight)
        assert not torch.equal(last,model.wav2vec2.feature_extractor.conv_layers[-1].conv.weight)
        checks[name]=dict(raw_loss=float(out.loss),frames=30,finite_nonzero_gradient_groups=norms,first_CNN_unchanged=True,last_CNN_updated=True)
        del model,optimizer,out;gc.collect()
    # Graph creation must work inside inference mode; repeats remain distinct.
    from reproduction.timit_ssl_native10_20261019.graphs import graph,decode
    topo=topology(vocab);states=topo['mapping'][vocab['b']];ids=[states[0],states[0],states[1],states[2],states[0],states[2]]
    scores=torch.full((len(ids),len(topo['labels'])),-30.,device=args.device)
    scores[torch.arange(len(ids),device=args.device),torch.tensor(ids,device=args.device)]=30.
    with torch.inference_mode():
        vector=graph(topo,vocab,device=args.device);supplied=graph(topo,vocab,[vocab['b'],vocab['b']],device=args.device)
        assert decode(scores,vector)[0]==[vocab['b'],vocab['b']]
        assert decode(scores,supplied)[0]==[1,2]
    config=recipe(matrix()[0]);full=train_case(output/'continuous',config,vocab)
    train_case(output/'resumed',config,vocab,stop=True);resumed=train_case(output/'resumed',config,vocab,resume=True)
    delta=max(float((full[k]-resumed[k]).abs().max()) for k in full);assert delta==0.,delta
    assert (output/'continuous/epoch_weights/epoch-001.pt').exists() and (output/'continuous/epoch_weights/epoch-002.pt').exists()
    report=dict(source_sha256=source_hashes(),six_recipes=checks,exact_resume_model_max_difference=delta,
                CPU_test_only='original loss with literal CUDA allocation device remapped to CPU; production/GPU core unchanged',
                S3_repeated_phone_graph_inside_inference_mode=True,independent_epoch_checkpoints=True,device=args.device)
    if args.full_ssl:
        assert sha(SSL/'pytorch_model.bin')==SSL_SHA
        model,_=build_model(config,vocab,73)
        report['original_SSL_identity']=verify_ssl_encoder(model);report['parameters']=parameter_manifest(model)
        del model;gc.collect()
        _,rows,_,_,audit=load_data(config);report['data']=audit
        report['longest_train_seconds']=max(r['duration'] for r in rows['train'])
    write(gate,report);print('VALIDATION_PASS',gate,flush=True)


if __name__=='__main__':main()
