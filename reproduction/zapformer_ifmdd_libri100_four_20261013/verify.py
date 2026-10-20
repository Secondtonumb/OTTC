"""Bound full eight-rank checkpoint resume and throughput acceptance to frozen sources."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ARMS=('jsd_fs','jsd_only','ce_fs','ce_only')

def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}

def exact(x,y,path):
    if torch.is_tensor(x):assert x.dtype==y.dtype and torch.equal(x,y),path
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:exact(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y),path
        for i,(a,b) in enumerate(zip(x,y)):exact(a,b,path+'/'+str(i))
    else:assert x==y,(path,x,y)

def verify(arm):
    cfg=json.loads((HERE/'configs'/f'{arm}.json').read_text());selection=json.loads((HERE/'BATCH_SELECTION.json').read_text())
    assert selection['source_sha256']==hashes()
    for name in ('CRITERION_GPU_PASS.json','PARITY_GPU_PASS.json'):
        assert json.loads((HERE/name).read_text())['status']=='passed'
    assert json.loads((HERE/'CRITERION_GPU_PASS.json').read_text())['source_sha256']==hashes()
    states=[];logs=[];perf=[]
    for run in ('uninterrupted','resumed'):
        output=HERE/'validation'/arm/run
        state=torch.load(output/'step-000000006.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['world_size']==len(state['rank_states'])==8
        assert state['progress']['global_step']==state['progress']['optimizer_updates']==6
        assert state['contract']['epochs']==50 and state['contract']['aux_loss_scale']==.05
        assert state['contract']['max_duration']==selection['max_duration']
        assert state['contract']['loss_config']==cfg and not state['contract']['smoke'] and not state['contract']['tiny']
        assert state['contract']['adapter_sha256']==hashes()
        rows=[json.loads(v) for v in (output/'updates.jsonl').read_text().splitlines()]
        assert len(rows)==6 and [r['optimizer_step'] for r in rows]==list(range(1,7))
        for r in rows:
            assert math.isfinite(r['loss']) and r['utterances']>8
            assert abs(r['loss']-r['weighted_ottc_sum']-r['weighted_forward_sum'])<.005
            if not cfg['forward_sum_weight']:assert r['forward_sum_raw_nll_sum']==r['weighted_forward_sum']==0
        states.append(state);logs.append(rows)
        perf.append([json.loads(v) for v in (output/'throughput.jsonl').read_text().splitlines()])
    for key in ('model','model_avg','optimizer','scheduler','rank_states','contract','progress'):exact(states[0][key],states[1][key],key)
    exact(logs[0],logs[1],'raw_loss_logs')
    assert all(r['seconds']>0 and r['audio_seconds_per_second']>0 for rows in perf for r in rows)
    memory=json.loads((HERE/'throughput'/arm/str(int(selection['max_duration']))/'SUMMARY.json').read_text())['total_memory_GiB']
    assert max(r['peak_reserved_GiB']/memory for rows in perf for r in rows)<.9,'DDP memory headroom insufficient'
    return dict(status='passed',arm=arm,source_sha256=hashes(),world_size=8,initialization_sha256=states[0]['contract']['initialization_sha256'],loss_config=cfg,full_model_resume='bitwise model/float64 average/optimizer/scheduler/scaler/RNG/sampler/statistics and raw logs',global_utterances_per_update=[r['utterances'] for r in logs[0]],losses=[r['loss'] for r in logs[0]],DDP_throughput=perf[0],max_duration=selection['max_duration'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--write-marker',action='store_true');a=ap.parse_args()
    marker=HERE/f'{a.arm.upper()}_8GPU_PASS.json'
    if a.write_marker:
        result=verify(a.arm);marker.write_text(json.dumps(result,indent=2)+'\n')
    else:
        result=json.loads(marker.read_text());assert result['status']=='passed' and result['source_sha256']==hashes()
        assert result['max_duration']==json.loads((HERE/'BATCH_SELECTION.json').read_text())['max_duration']
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
