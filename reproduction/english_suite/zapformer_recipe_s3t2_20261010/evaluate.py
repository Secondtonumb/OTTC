"""Evaluate last model and cumulative averaged model without test selection."""
import runtime
import json,sys
from pathlib import Path
import torch
from model_s3t2 import MFAZapformer
from tracking import Tracker
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
from mfa_metrics import evaluate

def main():
    out=HERE/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if (out/'EVAL_COMPLETE.json').exists():return
    path=max((HERE/'training').glob('step-*.pt'))
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['progress']['epoch']>=10 and state['contract']['recipe']=='zapformer_archived_tricks_v1'
    tracker=Tracker(HERE/'training',state['contract'],0)
    results=dict(checkpoint=str(path),progress=state['progress'],contract=state['contract'],models={})
    for label,key in [('last','model'),('cumulative_avg','model_avg')]:
        net=MFAZapformer();net.load_state_dict(state[key],strict=True);net.cuda().eval()
        model_results={}
        for split in ['development','test']:
            target=out/label;target.mkdir(exist_ok=True)
            model_results[split]=evaluate(net,'spottc',split,forced=split=='test',output=target)
            if tracker.run:
                for kind in ['text_free','forced_k2']:
                    tracker.run.summary.update({'final/'+label+'/'+split+'/'+kind+'/'+k:v for k,v in model_results[split].get(kind,{}).items() if isinstance(v,(int,float))})
        results['models'][label]=model_results
        (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
        del net;torch.cuda.empty_cache()
    (out/'EVAL_COMPLETE.json').write_text(json.dumps(dict(final=True,step=state['progress']['global_step'],test_selection=False))+'\n')
    if tracker.run:tracker.run.summary['training/complete']=True
    tracker.finish()

if __name__=='__main__':main()
