"""Integration checks: native features/RNG, real task updates and process resumes."""
import runtime
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import train as native
from augmentation import augment

HERE=Path(__file__).resolve().parent


def source_identity():
    files=list(HERE.glob('*.py'))+[HERE/'environment.sh',HERE/'validate_ddp_worker.sh',HERE/'vendor_manifest.json']
    native_hashes=json.loads((HERE/'vendor_manifest.json').read_text())['sha256']
    for name,digest in native_hashes.items():
        assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==digest,name
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def check_acceptance(single_only=False):
    paths=[HERE/'validation/SMOKE_PASS.json']
    if not single_only:paths.append(HERE/'validation_ddp/DDP_PASS.json')
    for path in paths:
        result=json.loads(path.read_text())
        assert result['source_identity']==source_identity(),f'Stale acceptance marker: {path}'
        assert set(result['objectives'])=={'jsd','framece'},path


def batch_evidence(directory, require_multiple=True):
    records=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
    if require_multiple:assert all(r['utterances']>1 for r in records),directory
    return dict(utterances_per_update=[r['utterances'] for r in records],loss_per_update=[r['loss'] for r in records])


def compare(left,right):
    a=torch.load(left,map_location='cpu',weights_only=False)
    b=torch.load(right,map_location='cpu',weights_only=False)
    maxima={}
    def visit(x,y,path):
        if torch.is_tensor(x):
            assert x.dtype==y.dtype and x.shape==y.shape,path
            if x.is_floating_point():
                delta=float((x-y).abs().max()) if x.numel() else 0.
                maxima[path.split('/')[0]]=max(delta,maxima.get(path.split('/')[0],0.))
                torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6,msg=path)
            else:assert torch.equal(x,y),path
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for key in x:visit(x[key],y[key],path+'/'+str(key))
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y),path
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'/'+str(i))
        elif isinstance(x,np.ndarray):
            assert np.array_equal(x,y),path
        else:assert x==y,(path,x,y)
    for key in ('model','model_avg','optimizer','scheduler'):
        visit(a[key],b[key],key)
    assert a['progress']['optimizer_updates']==b['progress']['optimizer_updates']==2
    assert len(a['rank_states'])==len(b['rank_states'])==a['world_size']==b['world_size']
    visit(a['contract'],b['contract'],'contract')
    for key in ('global_step','epoch','step_in_epoch'):
        visit(a['progress'][key],b['progress'][key],'progress/'+key)
    for x,y in zip(a['rank_states'],b['rank_states']):
        visit(x['sampler'],y['sampler'],'sampler')
        visit(x['scaler'],y['scaler'],'scaler')
        for key in ('python','numpy','cpu','cuda','specaug_seed','specaug_generators'):
            visit(x['random'][key],y['random'][key],'rng/'+key)
    return dict(next_update_state_comparison='rtol1e-5 atol1e-6; integer/RNG exact',max_absolute_difference=maxima)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--compare-root',type=Path)
    ap.add_argument('--output',type=Path,default=HERE/'validation')
    ap.add_argument('--objectives',nargs='+',choices=('jsd','framece'),default=['jsd','framece'])
    ap.add_argument('--check-acceptance',choices=('single','all'))
    args=ap.parse_args()
    if args.check_acceptance:
        check_acceptance(args.check_acceptance=='single');print('Source-bound acceptance verified');return
    identity=source_identity()
    if args.compare_root:
        results={objective:compare(args.compare_root/objective/'uninterrupted/step-000000002.pt',args.compare_root/objective/'resumed/step-000000002.pt') for objective in args.objectives}
        for objective in args.objectives:
            results[objective]['batches']=batch_evidence(args.compare_root/objective/'uninterrupted')
        results.update(source_identity=identity,objectives=args.objectives)
        marker='DDP_PASS.json' if set(args.objectives)=={'jsd','framece'} else 'DDP_PARTIAL_PASS.json'
        (args.compare_root/marker).write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results));return
    out=args.output;out.mkdir(exist_ok=True)
    for name,digest in json.loads((HERE/'vendor_manifest.json').read_text())['sha256'].items():
        assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==digest
    torch.manual_seed(42)
    features=torch.randn(4,350,80,device='cuda');lengths=torch.tensor([350,310,270,180],device='cuda')
    gold=torch.full((4,350),-100,device='cuda',dtype=torch.long)
    for i,n in enumerate(lengths):gold[i,:int(n)]=torch.arange(int(n),device='cuda')//25+3
    original=native.AlternatingSpecAugment(seed=92)
    adapted=native.AlternatingSpecAugment(seed=92)
    expected=original(features.clone(),lengths)
    actual,labels=augment(adapted,features.clone(),lengths,gold)
    assert torch.equal(actual,expected),'Feature augmentation was changed'
    for device in original.device_to_generator:
        assert torch.equal(original.device_to_generator[device].get_state(),adapted.device_to_generator[device].get_state())
    assert not torch.equal(labels,gold),'No time-warp boundary moved in fixture'
    for i,n in enumerate(lengths):assert bool((labels[i,int(n):]==-100).all())
    results={'source_identity':identity,'objectives':args.objectives,'native_source_hashes':'all unchanged','native_augmentation_features_and_rng':'exact match','ce_label_boundaries':'moved with time warp; padding preserved'}
    def run(objective,directory,steps,tiny=True,init=None):
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(HERE/'train_s3t2.py'),
            '--objective',objective,'--output',str(directory),'--smoke','--stop-after',str(steps),
            '--max-duration','100','--save-every','1','--eval-every','1','--average-period','2','--eval-limit','4']
        if tiny:command+=['--tiny']
        if init:command+=['--init-from',str(init)]
        subprocess.run(command,check=True)
    for objective in args.objectives:
        base=out/objective
        run(objective,base/'uninterrupted',2)
        run(objective,base/'resumed',1)
        run(objective,base/'resumed',2)
        results[objective]=compare(base/'uninterrupted/step-000000002.pt',base/'resumed/step-000000002.pt')
        results[objective]['batches']=batch_evidence(base/'uninterrupted')
        continuous=[json.loads(line)['loss'] for line in (base/'uninterrupted/updates.jsonl').read_text().splitlines()]
        restarted=[json.loads(line)['loss'] for line in (base/'resumed/updates.jsonl').read_text().splitlines()]
        assert continuous==restarted,(objective,continuous,restarted)
    for objective in args.objectives:
        initial=Path('/work/gm64/m64000/fa-bench/data/mfa_spottc_jsd_s3t2_10ms_20261009/training/step-000068868.pt') if objective=='framece' else None
        run(objective,out/('full_'+objective),1,tiny=False,init=initial)
        state=torch.load(out/('full_'+objective)/'step-000000001.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['progress']['optimizer_updates']==1
        results['full_'+objective]=dict(status='native loop completed one optimizer update; TIMIT loss/recognition/segmentation hooks ran',batches=batch_evidence(out/('full_'+objective)))
        del state
    marker='SMOKE_PASS.json' if set(args.objectives)=={'jsd','framece'} else 'PARTIAL_PASS.json'
    (out/marker).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
