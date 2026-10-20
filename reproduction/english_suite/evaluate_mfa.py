"""Fixed-checkpoint external dev/test evaluation for the original MFA English models."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SUITE=Path(__file__).resolve().parent
BASE=Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa')
sys.path[:0]=[str(BASE/'experiments/mfa_zapformer'),str(SUITE)]
import torch
from model_10ms import MFAZapformer
from film_k4_10ms import FiLMK4
from mfa_metrics import evaluate,DATA


def main():
    from reproduction.wandb_policy import init_evaluation
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--final',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    progress=state['progress'];step=progress['global_step']
    if args.final:assert progress['epoch']>=10
    model=FiLMK4() if args.arm=='film_k4' else MFAZapformer(args.arm)
    model.load_state_dict(state['model'],strict=True);del state
    model=model.cuda().eval()
    # Prove the target-free acoustic implementation reproduces the original forward.
    import numpy as np
    from mfa_metrics import audio_forward
    row=json.loads(next((DATA/'development.jsonl').open()))
    data=np.memmap(row['feature_file'],mode='r',dtype=np.float32)
    x=torch.from_numpy(np.array(data[row['feature_offset']:row['feature_offset']+80*row['frames']]).reshape(1,-1,80)).cuda()
    lengths=torch.tensor([row['frames']],device='cuda');targets=torch.tensor(row['tokens'],device='cuda');target_lengths=torch.tensor([len(row['tokens'])],device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(x,lengths,targets,target_lengths)
        expected=audio_forward(model,args.arm,x,lengths)[0]
    torch.testing.assert_close(actual['logits'] if args.arm=='film_k4' else actual[1],expected,rtol=0,atol=0)
    del actual,expected,x,data
    output=SUITE/'mfa_evaluation'/args.arm/('step-'+str(step))
    if args.limit:output=output/'validation'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'EVAL_COMPLETE.json').exists():
        print('Already evaluated',output,flush=True)
        return
    result={'arm':args.arm,'checkpoint':str(args.checkpoint),'progress':progress,'final':args.final,'intermediate_diagnostic':not args.final,'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'metrics_source_sha256':hashlib.sha256((SUITE/'mfa_metrics.py').read_bytes()).hexdigest(),'data_provenance':json.loads((DATA/'PREPARE_COMPLETE.json').read_text()),'splits':{}}
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    result['checkpoint_sha256']=digest.hexdigest()
    for split in ['development','test']:
        result['splits'][split]=evaluate(model,args.arm,split,forced=split=='test',limit=args.limit,output=output)
        (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    import wandb
    run=init_evaluation(project='ottc-mfa-english',entity='kevingenghaopeng-',job_type='evaluation',name=args.arm+'-step'+str(step)+'-dev-test',group='mfa_english_external_timit',dir=str(output),config={'arm':args.arm,'checkpoint_step':step,'final':args.final,'limit':args.limit,'test_for_checkpoint_selection':False},settings=wandb.Settings(init_timeout=45))
    for split,modes in result['splits'].items():
        for mode in ['text_free','forced_k2']:
            if mode in modes:run.summary.update({split+'/'+mode+'/'+k:v for k,v in modes[mode].items() if isinstance(v,(int,float))})
    artifact=wandb.Artifact(args.arm+'-step'+str(step)+'-external-timit',type='evaluation')
    artifact.add_file(str(output/'metrics.json'));run.log_artifact(artifact)
    (output/'wandb_run.json').write_text(json.dumps({'id':run.id,'url':run.url})+'\n');run.finish()
    (output/'EVAL_COMPLETE.json').write_text(json.dumps({'arm':args.arm,'step':step,'final':args.final,'limit':args.limit,'test':True})+'\n')
    print(json.dumps(result['splits']),flush=True)
    from summarize_results import main as summarize_results
    summarize_results()

if __name__=='__main__':main()
