"""Diagnostic dev400 scoring without changing training RNG or selection."""
import bootstrap
from bootstrap import ROOT
from collections import Counter
import json,random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import TrainerCallback
import reproduction.evaluate as legacy
from segmentation_metrics import r_value_percent


class DevelopmentCallback(TrainerCallback):
    def __init__(self,processor,output):
        self.processor=processor;self.output=Path(output);self.last_step=None
        self.manifest=ROOT.parent/'IF-MDD/data/timit_alignments_39phone_unrounded/json/dev.json'
        self.rows=sorted(json.loads(self.manifest.read_text()).items())
        assert len(self.rows)==400

    def on_evaluate(self,args,state,control,model=None,**kwargs):
        if not state.is_world_process_zero or state.global_step==self.last_step:return
        self.last_step=state.global_step
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        training=model.training;model.eval();counts=Counter();bounds=[0,0,0]
        try:
            with torch.inference_mode():
                for key,row in self.rows:
                    wav=row['wav'].replace('/home/m64000/work/','/work/gm64/m64000/')
                    audio,sr=sf.read(wav,dtype='float32');assert sr==16000 and audio.ndim==1
                    batch=self.processor(audio,sampling_rate=sr,return_tensors='pt',return_attention_mask=True).to('cuda')
                    reference=[r['phoneme'] for r in row['phonemes']]
                    labels=self.processor.tokenizer(reference).input_ids
                    labels=torch.tensor([[i for seq in labels for i in seq]],device='cuda')
                    out=model(**batch,labels=labels)
                    ids=out.logits[0].argmax(-1).cpu().numpy()
                    alpha=out.hidden_states[0].cpu().numpy().reshape(-1)
                    ids[alpha==0]=self.processor.tokenizer.pad_token_id
                    ids[ids==self.processor.tokenizer.unk_token_id]=self.processor.tokenizer.pad_token_id
                    hyp=self.processor.decode(ids).split()
                    counts.update(errors=legacy.edit_alignment(reference,hyp)[0],phones=len(reference),
                        loss=float(out.loss),utterances=1)
                    duration=len(audio)/sr
                    rb=sorted({float(t) for r in row['phonemes'] for t in (r['start'],r['end']) if .02<float(t)<duration-.02})
                    pb=[int(i)*.02 for i in np.flatnonzero(ids[1:]!=ids[:-1])+1 if .02<int(i)*.02<duration-.02]
                    bounds[0]+=legacy.matches(rb,pb,.02);bounds[1]+=len(rb);bounds[2]+=len(pb)
        finally:
            model.train(training);random.setstate(saved[0]);np.random.set_state(saved[1])
            torch.set_rng_state(saved[2]);torch.cuda.set_rng_state(saved[3])
        p=100*bounds[0]/bounds[2] if bounds[2] else 0.;r=100*bounds[0]/bounds[1] if bounds[1] else 0.
        record=dict(step=state.global_step,epoch=state.epoch,utterances=counts['utterances'],
            raw_PER_percent=100*counts['errors']/counts['phones'],loss_per_utterance=counts['loss']/counts['utterances'],
            text_free_precision_20ms=p,text_free_recall_20ms=r,text_free_F1_20ms=2*p*r/(p+r) if p+r else 0.,
            text_free_R_value_20ms=r_value_percent(p,r),boundary_counts=bounds,
            protocol='original single-head argmax changes including blank; 20ms grid, edges excluded 20ms; one-to-one matching; frozen IF-MDD R-value')
        with (self.output/'development_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print('DEVELOPMENT',json.dumps(record),flush=True)
        import wandb
        if wandb.run:wandb.run.log({'train/global_step':state.global_step,**{'development/'+k:v for k,v in record.items() if isinstance(v,(int,float))}})
