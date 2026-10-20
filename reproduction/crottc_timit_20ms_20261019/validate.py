"""Verify real augmentation, legacy CR, padding, microbatch scaling and resume."""
import argparse
from contextlib import contextmanager
import gc
import fcntl
import inspect
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from transformers import Wav2Vec2Model,Wav2Vec2Config,TrainingArguments,TrainerCallback,set_seed
from common import HERE,ROOT,SSL,SSL_SHA,sha,recipe,matrix,build_model,write,source_hashes,parameter_manifest,SpeechViews,pair_chunks
import ottc.models.sequence.wav2vec2_ottc as upstream
from reproduction.crottc_timit_20ms_20261019.vendor.legacy_cr import LegacyCR


@contextmanager
def cpu_loss_allocations(device='cpu'):
    original=upstream.batched_ottc_loss_bucketized
    if device=='cpu':
        namespace=dict(original.__globals__)
        exec(compile(inspect.getsource(original).replace(".to('cuda')",".to('cpu')"),'cpu_allocation_test_only','exec'),namespace)
        upstream.batched_ottc_loss_bucketized=namespace[original.__name__]
    try:yield
    finally:upstream.batched_ottc_loss_bucketized=original


def verify_ssl_encoder(model):
    source=torch.load(SSL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True);checked=0
    for key,value in model.wav2vec2.state_dict().items():
        source_key=key if key in source else 'wav2vec2.'+key
        assert source_key in source and torch.equal(value.detach().cpu(),source[source_key]),key
        checked+=1
    del source;gc.collect()
    return dict(tensors_checked=checked,all_equal=True,weights_sha256=SSL_SHA)


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
        audio=torch.sin(torch.arange(8000).float()*(.01+i*.001))
        length=8000-i*640;audio[length:]=0
        return dict(input_values=audio,attention_mask=(torch.arange(8000)<length).long(),labels=torch.tensor([1,2,3]))


def stack(rows):return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}


class TinyCollator:
    micro=1
    def __call__(self,rows):return dict(microbatches=[stack(rows)],logical_batch_size=len(rows))


class Stop(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==2:control.should_training_stop=True;control.should_save=True


def train_case(path,config,vocab,resume=False,stop=False):
    from reproduction.crottc_timit_20ms_20261019.train import QuietTrainer,Policy
    class Dummy:
        def log(self,*a,**k):pass
    set_seed(42);model,_=tiny(config,vocab);path.mkdir(parents=True,exist_ok=True)
    policy=Policy(path,config,[],vocab,None,None,float('inf'),Dummy())
    args=TrainingArguments(output_dir=str(path),per_device_train_batch_size=2,num_train_epochs=2,learning_rate=1e-4,
        warmup_steps=1,save_strategy='epoch',logging_steps=1,report_to=[],dataloader_num_workers=0,
        remove_unused_columns=False,disable_tqdm=True,seed=42,save_safetensors=True,use_cpu=True)
    trainer=QuietTrainer(model=model,args=args,train_dataset=Data(),data_collator=TinyCollator(),callbacks=[policy]+([Stop()] if stop else []))
    with cpu_loss_allocations():trainer.train(resume_from_checkpoint=str(path/'checkpoint-2') if resume else None)
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--full-ssl',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    # Compare FP32 full/microbatch algebra at FP32 precision. GH200 TF32
    # convolutions select different kernels for batch8 versus batch2; the
    # separate numeric probe demonstrates that this alone exceeds the gate.
    # These flags affect only this validation process, not BF16 training.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=HERE/'validation';output.mkdir(exist_ok=True)
    lock=(output/'.validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    gate=output/(args.device.upper()+('_FULL_SSL_PASS.json' if args.full_ssl else '_PASS.json'))
    if gate.exists() and json.loads(gate.read_text())['source_sha256']==source_hashes():print('VALIDATION_CACHE',gate);return
    provenance=json.loads((HERE/'vendor/provenance.json').read_text())
    for path,expected in provenance['source_files'].items():assert sha(path)==expected,('External source drift',path)
    vocab=json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json').read_text())
    from reproduction.crottc_timit_20ms_20261019.profile_recipe import gradient_checks
    checks={};device=args.device
    for name in matrix():
        config=recipe(name);set_seed(42);model,_=tiny(config,vocab);model.to(device).train()
        base={k:v.to(device) for k,v in stack([Data()[i] for i in range(4)]).items()};augmentation=SpeechViews(device)
        # Lazy SpeechBrain imports can change PyTorch's global precision flags.
        import_precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        rng=torch.get_rng_state();python_rng=__import__('random').getstate()
        chunks,aug=pair_chunks(model,base,augmentation,4)
        assert chunks[0]['input_values'].shape[0]==8
        assert not torch.equal(chunks[0]['input_values'][:4],chunks[0]['input_values'][4:])
        mask=chunks[0]['attention_mask'];assert torch.equal(mask[:4],mask[4:])
        assert not chunks[0]['input_values'][mask==0].count_nonzero()
        frozen=[p.detach().clone() for p in model.wav2vec2.feature_extractor.parameters()]
        with cpu_loss_allocations(device):
            out=model(**chunks[0]);full_loss=out.loss;full_loss.backward()
            norms=gradient_checks(model);full_grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
            full_components={k:float(v) for k,v in model.last_loss_components.items()}
            model.zero_grad(set_to_none=True)
            # Same prepared views, split by pairing index. No second random augmentation.
            c=chunks[0];sum_loss=0.
            for i in range(4):
                part={k:(torch.cat([v[i:i+1],v[4+i:5+i]]) if k in ['input_values','attention_mask','labels'] else
                    v[i:i+1] if k=='cr_relative_lengths' else v) for k,v in c.items()}
                small=model(**part);small.loss.backward();sum_loss+=float(small.loss)
            delta=max(float((p.grad-full_grads[k]).abs().max()) for k,p in model.named_parameters() if k in full_grads)
            assert abs(float(full_loss)-sum_loss)<2e-5,(float(full_loss),sum_loss)
            assert delta<1e-4,delta
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3);optimizer.step()
        assert all(torch.equal(a,b) for a,b in zip(frozen,model.wav2vec2.feature_extractor.parameters()))
        model.eval()
        clean=model(input_values=base['input_values'],attention_mask=base['attention_mask'])
        assert clean.logits.shape[0]==4 and clean.logits.shape[1]==24
        checks[name]=dict(loss=float(full_loss),gradient_norms=norms,microbatch_gradient_max_difference=delta,
            microbatch_loss_absolute_difference=abs(float(full_loss)-sum_loss),
            tf32_flags_after_augmentation_import=import_precision,
            loss_components=full_components,augmentation=aug,CNN_unchanged=True,inference_single_view_native20ms=True)
        del model,optimizer,out,small,clean;gc.collect()
    # Exact historical padding and stop-gradient behavior, compared to an independent autograd expression.
    set_seed(7);left=torch.randn(2,8,42,requires_grad=True);right=torch.randn(2,8,42,requires_grad=True)
    lengths=torch.tensor([1.,.5]);log1=left.log_softmax(-1);log2=right.log_softmax(-1)
    legacy=LegacyCR()._compute_cr_loss(log1,log2,lengths);legacy.backward()
    assert left.grad[1,4:].count_nonzero()==0 and right.grad[1,4:].count_nonzero()==0
    with torch.no_grad():
        valid=(torch.arange(8)[None,:]<(lengths*8).long()[:,None]).unsqueeze(-1)
        expected_left=(log1.exp()-log2.exp())*valid/(2*8*42-4)
        assert torch.allclose(left.grad,expected_left,atol=1e-8)
        assert torch.allclose(right.grad,-expected_left,atol=1e-8)
        reference=((log2.exp()*(log2-log1)+log1.exp()*(log1-log2)).sum(-1)*
            (torch.arange(8)[None,:]<(lengths*8).long()[:,None])).sum()/(2*8*42-4)
        assert torch.allclose(legacy,reference,atol=1e-7)
    config=recipe(matrix()[0])
    # Isolated validation directories permit safe revalidation after source edits.
    with tempfile.TemporaryDirectory(prefix='resume_',dir=output) as case:
        case=Path(case);full=train_case(case/'continuous',config,vocab)
        train_case(case/'resumed',config,vocab,stop=True);resumed=train_case(case/'resumed',config,vocab,resume=True)
        delta=max(float((full[k]-resumed[k]).abs().max()) for k in full);assert delta==0.,delta
        snapshots=list((case/'continuous/epoch_weights').glob('epoch-*.pt'));assert len(snapshots)==2
        # Validation files contain tiny models only; retain evidence alongside the gate.
        saved=torch.load(snapshots[0],map_location='cpu',weights_only=False)
        metadata={k:v for k,v in saved.items() if k!='model'}
    report=dict(source_sha256=source_hashes(),three_recipes=checks,exact_resume_model_max_difference=delta,
        independent_epoch_checkpoints=True,epoch_snapshot_metadata=metadata,legacy_CR_formula_verified=True,
        masked_padding_zero_gradient=True,bidirectional_teacher_stop_gradient_verified=True,
        device=device,validation_precision='FP32; CUDA matmul and cuDNN TF32 disabled; original tolerances retained',
        CPU_test_only='CUDA allocation literals remapped only in test; core source unchanged')
    if args.full_ssl:
        assert sha(SSL/'pytorch_model.bin')==SSL_SHA
        model,_=build_model(config,vocab,73);report['original_SSL_identity']=verify_ssl_encoder(model)
        report['parameters']=parameter_manifest(model);del model;gc.collect()
        from common import load_data
        _,rows,_,_,audit=load_data(config);report['data']=audit;report['longest_train_seconds']=max(r['duration'] for r in rows['train'])
    write(gate,report);print('VALIDATION_PASS',gate,flush=True)


if __name__=='__main__':main()
