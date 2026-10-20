"""Verify full-model eight-rank epoch extension and deterministic restart."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from continue_jsd import NATIVE, sha, source_identity, torch
sys.path.insert(0,str(HERE.parent/'zapformer_s3t2ss_acceptance_20261012'))
from validate_epochs import exact
from validate import check_acceptance


def script_hashes():
    return {p.name:sha(p) for suffix in ('*.py','*.sh','*.pbs') for p in sorted(HERE.glob(suffix))}


def verify():
    check_acceptance()
    identity=source_identity()
    original=json.loads((HERE.parent/'zapformer_libri100_s3t2ss_native_20261012/EIGHT_GPU_PASS.json').read_text())
    assert original['source_identity']==identity and original['world_size']==8
    spec=json.loads((HERE/'training/continuation_spec.json').read_text())
    assert spec['source_identity']==identity and spec['scheduler_policy']=='native_replanned_total' and spec['target_epochs']==50
    assert sha(spec['origin_checkpoint'])==spec['origin_checkpoint_sha256']
    states=[]; logs=[]
    for name in ('uninterrupted','resumed'):
        output=HERE/'validation'/name
        state=torch.load(output/'step-000001062.pt',map_location='cpu',weights_only=False)
        assert state['world_size']==len(state['rank_states'])==8
        assert state['progress']['optimizer_updates']==state['progress']['global_step']==1062
        assert state['progress']['epoch']==10
        assert state['contract']['epochs']==50 and state['contract']['objective']=='jsd'
        assert state['contract']['continuation']['wrapper_sha256']==sha(HERE/'continue_jsd.py')
        assert state['contract']['continuation']['origin_checkpoint_sha256']==spec['origin_checkpoint_sha256']
        states.append(state)
        rows=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines()]
        assert len(rows)==1062 and [r['optimizer_step'] for r in rows[-2:]]==[1061,1062]
        assert all(r['epoch']==11 and r['lr']>.01 and r['utterances']>8 for r in rows[-2:])
        logs.append(rows[-2:])
    for key in ('model','model_avg','optimizer','scheduler','rank_states','contract','progress'):
        exact(states[0][key],states[1][key],key)
    exact(logs[0],logs[1],'update_logs')
    return dict(source_identity=identity,script_sha256=script_hashes(),world_size=8,
                origin_checkpoint_sha256=spec['origin_checkpoint_sha256'],target_total_epochs=50,
                restored_origin_step=1060,validation_final_step=1062,
                restored_optimizer_scaler_sampler_rng_and_average=True,
                full_model_states_and_raw_losses='bitwise equal after restart',
                new_raw_losses=[r['loss'] for r in logs[0]],new_learning_rates=[r['lr'] for r in logs[0]],
                global_utterances_per_update=[r['utterances'] for r in logs[0]])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-marker',action='store_true')
    args=parser.parse_args(); marker=HERE/'CONTINUATION_8GPU_PASS.json'
    result=verify()
    if args.write_marker:marker.write_text(json.dumps(result,indent=2)+'\n')
    else:assert result==json.loads(marker.read_text()),'Stale continuation acceptance'
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_identity','script_sha256')}))
