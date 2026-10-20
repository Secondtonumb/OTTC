"""Exercise native epoch completion, boundary restart and final average scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'zapformer_s3t2ss_native_20261012'
sys.path.insert(0,str(NATIVE))
import runtime
import numpy as np
import torch
from data import CE_ROOT, jsd_rows, plan, read_rows
from validate import source_identity


def exact(x,y,path=''):
    if torch.is_tensor(x):
        assert x.dtype==y.dtype and torch.equal(x,y),path
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for key in x:exact(x[key],y[key],path+'/'+str(key))
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)):exact(u,v,path+'/'+str(i))
    else:assert x==y,(path,x,y)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'results')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    identity=source_identity();results={}
    for objective in ('jsd',):
        rows=jsd_rows(1024) if objective=='jsd' else read_rows(CE_ROOT/'data/training.jsonl')[:1024]
        count=len(plan(rows,500,0,1,0));assert count>2
        base=args.output/objective
        def run(directory,stop=0):
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(NATIVE/'train_s3t2.py'),
                '--objective',objective,'--topology','s3_t2_star_star','--output',str(directory),'--tiny','--smoke','--epochs','2',
                '--max-duration','500','--save-every','1000','--eval-every','1000','--average-period','2','--eval-limit','4']
            if stop:command+=['--stop-after',str(stop)]
            subprocess.run(command,check=True)
        run(base/'continuous')
        run(base/'resumed',count)
        assert not (base/'resumed/TRAIN_COMPLETE.json').exists()
        run(base/'resumed')
        states=[]
        for directory in (base/'continuous',base/'resumed'):
            assert (directory/'TRAIN_COMPLETE.json').exists()
            state=torch.load(sorted(directory.glob('step-*.pt'))[-1],map_location='cpu',weights_only=False)
            assert state['progress']['epoch']==2
            metrics=[json.loads(line) for line in (directory/'timit_metrics.jsonl').read_text().splitlines()]
            averaged=[m for m in metrics if m.get('checkpoint')=='native_cumulative_float64_average']
            assert {m['split'] for m in averaged}=={'development','test'}
            states.append(state)
        for key in ('model','model_avg','optimizer','scheduler','rank_states','contract','progress'):
            exact(states[0][key],states[1][key],key)
        logs=[[json.loads(line)['loss'] for line in (directory/'updates.jsonl').read_text().splitlines()] for directory in (base/'continuous',base/'resumed')]
        assert logs[0]==logs[1]
        results[objective]=dict(epoch_boundary_restart_step=count,final_optimizer_updates=states[0]['progress']['optimizer_updates'],
            all_saved_states_and_raw_losses='bitwise equal',completed_epochs=2,current_and_average_scoring=True)
        del states
    result=dict(source_identity=identity,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results)
    (args.output/'EPOCH_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
