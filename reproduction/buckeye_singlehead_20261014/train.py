"""Matched Buckeye single-head OTTC CE / original upstream JSD training."""
import bootstrap
from bootstrap import ROOT, HERE
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

import datasets
import jiwer
import numpy as np
import torch
from transformers import TrainerCallback, Wav2Vec2Processor
import run_exp
from segmentation_metrics import r_value_percent

OBJECTIVES = {'ce':'crossentropy', 'jsd_upstream':'jsd'}


def model_directory(objective, smoke=False):
    suffix = ('smoke_' if smoke else 'runexp_')+'buckeye_singlehead_'+objective+'_20261014'
    return ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/suffix


def hashes():
    files = [HERE/'train.py', HERE/'prepare.py', HERE/'bootstrap.py', ROOT/'run_exp.py',
        ROOT/'ottc/models/sequence/wav2vec2_ottc.py', ROOT/'ottc/ops_loss/ot_loss.py']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def boundary_matches(ref, pred, tolerance):
    i=j=n=0
    while i<len(ref) and j<len(pred):
        if abs(ref[i]-pred[j])<=tolerance+1e-9:
            n+=1;i+=1;j+=1
        elif pred[j]<ref[i]:j+=1
        else:i+=1
    return n


def preprocess(logits, labels):
    ids = logits[0].argmax(-1) if isinstance(logits,tuple) else logits.argmax(-1)
    if isinstance(logits,tuple):
        ids=ids.masked_fill(logits[1][1]==0,0)
    return ids.masked_fill(ids==41,0)


def recognition_metrics(processor):
    def metrics(pred):
        ids=np.asarray(pred.predictions).copy();ids[ids<0]=0
        labels=np.asarray(pred.label_ids).copy();labels[labels<0]=0
        hyps=processor.batch_decode(ids)
        refs=processor.batch_decode(labels,group_tokens=False)
        def nosil(text):
            return ' '.join('d' if p=='dx' else p for p in text.split() if p not in ('sil','q'))
        return {'wer':jiwer.wer(refs,hyps), 'nosil_PER_percent':100*jiwer.wer(list(map(nosil,refs)),list(map(nosil,hyps)))}
    return metrics


class DevelopmentSegmentation(TrainerCallback):
    def __init__(self,processor,dataset,output,smoke):
        self.processor=processor;self.dataset=dataset;self.output=output;self.last=None
        self.rows=[json.loads(l) for l in (HERE/'data/dev.jsonl').read_text().splitlines()]
        if smoke:self.rows=self.rows[:len(dataset)]

    def on_evaluate(self,args,state,control,model=None,**kwargs):
        if self.last==state.global_step:return
        self.last=state.global_step
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        training=model.training;model.eval();counts=Counter();begin=time.monotonic()
        try:
            with torch.inference_mode():
                for start in range(0,len(self.rows),8):
                    rows=self.rows[start:start+8]
                    features=[{'input_values':self.dataset[i]['input_values']} for i in range(start,start+len(rows))]
                    batch=self.processor.pad(features,return_tensors='pt').to('cuda')
                    out=model(**batch)  # Segmentation logits depend only on audio.
                    ids=out.logits.argmax(-1).cpu().numpy();lengths=out.hidden_states[1].sum(-1).cpu().tolist()
                    for row,path,length in zip(rows,ids,lengths):
                        path=path[:int(length)];path[path==41]=0
                        rb=sorted({float(t) for _,s,e in row['reference'] for t in (s,e) if .02<t<row['duration']-.02})
                        pb=[int(i)*.02 for i in np.flatnonzero(path[1:]!=path[:-1])+1 if .02<int(i)*.02<row['duration']-.02]
                        counts['matches']+=boundary_matches(rb,pb,.02)
                        counts['reference']+=len(rb);counts['predicted']+=len(pb)
        finally:
            model.train(training);random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
        m,r,p=counts['matches'],counts['reference'],counts['predicted']
        precision=100*m/p if p else 0.;recall=100*m/r if r else 0.
        record=dict(step=state.global_step,epoch=state.epoch,utterances=len(self.rows),
            text_free_F1_20ms=200*m/(r+p) if r+p else 0.,text_free_R_value_20ms=r_value_percent(precision,recall),
            precision_20ms=precision,recall_20ms=recall,counts=dict(counts),seconds=time.monotonic()-begin)
        with (self.output/'development_segmentation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print('DEV_SEGMENTATION',json.dumps(record),flush=True)
        import wandb
        if wandb.run:wandb.run.log({'train/global_step':state.global_step,**{'development/'+k:v for k,v in record.items() if isinstance(v,(float,int))}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=list(OBJECTIVES),required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(),'Run on a PBS GPU allocation'
    audit=json.loads((HERE/'data/PREPARE_COMPLETE.json').read_text())
    output=model_directory(args.objective,args.smoke)
    prior_gate=HERE/'validation'/args.objective/'PASS.json'
    if args.smoke and prior_gate.exists():
        previous=json.loads(prior_gate.read_text())
        if previous['source_sha256']==hashes():
            print('GPU validation already passed; reuse',flush=True);return
    if (output/'TRAIN_COMPLETE.json').exists():
        print('Completed checkpoint exists; reuse it',flush=True);return
    if not args.smoke:
        gate=json.loads((HERE/'validation'/args.objective/'PASS.json').read_text())
        assert gate['source_sha256']==hashes(),'Training sources changed after GPU validation'
    output.mkdir(parents=True,exist_ok=True)
    provenance=dict(objective=args.objective,dataset='buckeye_original_partial19',
        initialization='facebook/wav2vec2-large-lv60 self-supervised pretrained backbone; random phone/mass heads; no TIMIT supervised weights',
        frame_ms=20,single_head=True,epochs=40,batch_size=64,learning_rate=2e-4,seed=42,
        precision='BF16 autocast; FP32 master weights; common CE/JSD setting',
        freeze_cnn=True,transformer_finetuned=True,loss='original upstream OTTC transport '+OBJECTIVES[args.objective],
        framewise_CE=False,dev='epoch diagnostics only',selection='fixed final epoch40; never select on test',
        data=audit,source_sha256=hashes(),job=os.environ.get('PBS_JOBID'),host=platform.node(),smoke=args.smoke)
    (output/'experiment.json').write_text(json.dumps(provenance,indent=2)+'\n')
    data=datasets.DatasetDict({s:datasets.load_from_disk(HERE/'data'/('cache_'+s)) for s in ['train','dev']})
    vocab=json.loads((HERE/'data/vocab.json').read_text())
    processor=Wav2Vec2Processor.from_pretrained(HERE/'data/processor')
    if args.smoke:
        lengths=data['train']['length']
        longest=sorted(range(len(lengths)),key=lengths.__getitem__,reverse=True)[:64]
        data['train']=data['train'].select(longest)
        data['dev']=data['dev'].select(range(16))
    run_exp.DATASET_LOADERS['buckeye_original_partial19']=run_exp.DatasetConfig(
        name='buckeye_original_partial19',loader=lambda *a,**k:(data,vocab,processor),transcript_column='labels')
    run_exp.build_compute_metrics=lambda **kwargs:recognition_metrics(processor)
    original_arguments=run_exp.TrainingArguments
    def bf16_arguments(*a,**kwargs):
        kwargs.update(fp16=False,bf16=True)
        return original_arguments(*a,**kwargs)
    run_exp.TrainingArguments=bf16_arguments
    original_config=run_exp.load_config_from_args
    def bf16_config(parsed):
        config=original_config(parsed);config['precision']='bf16'
        return config
    run_exp.load_config_from_args=bf16_config
    original_load=run_exp.load_model
    def checked_model(config,vocab_size):
        resume=run_exp.latest_checkpoint(output)
        model=original_load(config,vocab_size)
        assert model.config.vocab_size==42 and model.forward_type=='regfrozen'
        if resume is None:
            pretrained=torch.load(ROOT/'reproduction/models/wav2vec2-large-lv60/pytorch_model.bin',map_location='cpu',weights_only=False)
            for key in ['feature_projection.projection.weight','encoder.layers.0.attention.q_proj.weight','encoder.layers.23.attention.q_proj.weight']:
                source_key=key if key in pretrained else 'wav2vec2.'+key
                assert torch.equal(model.wav2vec2.state_dict()[key].cpu(),pretrained[source_key]),key
            del pretrained
        if args.smoke:model.frozen_steps_for_regfrozen=2
        return model
    run_exp.load_model=checked_model
    original_trainer=run_exp.create_trainer
    timings=[]
    def checked_trainer(model,dataset,processor,config,metrics_fn,model_output_dir):
        trainer=original_trainer(model,dataset,processor,config,metrics_fn,model_output_dir)
        trainer.args.disable_tqdm=True
        trainer.preprocess_logits_for_metrics=preprocess
        trainer.add_callback(DevelopmentSegmentation(processor,dataset['dev'],model_output_dir,args.smoke))
        if args.smoke:
            trainer.args.max_steps=2
            original_step=trainer.training_step
            def training_step(active,batch):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
                loss=original_step(active,batch)
                torch.cuda.synchronize();assert torch.isfinite(loss)
                raw=trainer.accelerator.unwrap_model(active)
                assert raw.lm_head.weight.grad is not None and torch.isfinite(raw.lm_head.weight.grad).all()
                q=raw.wav2vec2.encoder.layers[0].attention.q_proj.weight
                assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
                assert all(p.grad is None for p in raw.wav2vec2.feature_extractor.parameters())
                record=dict(model_step=raw.step,loss=float(loss),batch_size=len(batch['input_values']),
                    padded_duration_seconds=batch['input_values'].shape[1]/16000,
                    forward_backward_seconds=time.monotonic()-begin,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    transformer_gradient_verified=True,cnn_frozen_verified=True)
                timings.append(record);print('GPU_VALIDATION',json.dumps(record),flush=True)
                return loss
            trainer.training_step=training_step
        return trainer
    run_exp.create_trainer=checked_trainer
    os.environ.pop('WANDB_DISABLED',None)
    sys.argv=['run_exp.py','--dataset','buckeye_original_partial19','--train_name','train','--test_name','dev',
        '--loss_type','ottc','--encoder_name','wav2vec2-large-lv60','--batch_size','64',
        '--learning_rate','2e-4','--num_train_epochs','40','--suffix',output.name,
        '--forward_type','regfrozen','--frozen_epochs_for_regfrozen','30',
        '--bins_metric',OBJECTIVES[args.objective],'--logging_steps','50','--dataloader_num_workers','4']
    run_exp.main()
    state=json.loads((output/'trainer_state.json').read_text())
    checkpoint=run_exp.latest_checkpoint(output)
    if args.smoke:
        assert state['global_step']==2 and len(timings)==2
        gate=dict(objective=args.objective,source_sha256=hashes(),batch64_longest_inputs_safe=True,
            pretrained_samples_identical=True,optimizer_steps=2,freeze_transition_tested=True,batches=timings,
            job=os.environ.get('PBS_JOBID'),host=platform.node())
        directory=HERE/'validation'/args.objective;directory.mkdir(parents=True,exist_ok=True)
        (directory/'PASS.json').write_text(json.dumps(gate,indent=2)+'\n')
    else:
        expected=40*math.ceil(len(data['train'])/64)
        assert state['epoch']==40 and state['global_step']==expected,state
        (output/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(epoch=40,global_step=expected,
            checkpoint=str(checkpoint),objective=args.objective,job=os.environ.get('PBS_JOBID')),indent=2)+'\n')
    print('TRAIN_COMPLETE',args.objective,'smoke' if args.smoke else 'full',checkpoint,flush=True)


if __name__=='__main__':main()
