"""Bind the 8-rank resume and full-size throughput evidence to frozen source."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'zapformer_s3t2ss_native_20261012'
sys.path.insert(0,str(NATIVE))
import runtime
import torch
from validate import check_acceptance, source_identity


def verify(job):
    check_acceptance()
    identity=source_identity()
    epoch=HERE.parent/'zapformer_s3t2ss_acceptance_20261012/results/EPOCH_PASS.json'
    assert json.loads(epoch.read_text())['source_identity']==identity
    root=HERE/'validation_8gpu'
    result=json.loads((root/'STAR_DDP_PASS.json').read_text())
    assert result['source_identity']==identity and result['topology']=='s3_t2_star_star'
    for mode in ('uninterrupted','resumed'):
        checkpoint=root/'jsd'/mode/'step-000000002.pt'
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert state['world_size']==len(state['rank_states'])==8
        assert state['progress']['optimizer_updates']==2
    losses=[]
    for mode in ('uninterrupted','resumed'):
        rows=[json.loads(line) for line in (root/'jsd'/mode/'updates.jsonl').read_text().splitlines()]
        losses.append([r['loss'] for r in rows])
    assert losses[0]==losses[1]
    profile_root=HERE.parent/'zapformer_s3t2ss_gpu_profile_20261012'/f'ddp_8_640_{job}'
    files=sorted(profile_root.glob('profile-rank-*.json'))
    profiles=[json.loads(path.read_text()) for path in files]
    assert len(profiles)==8 and {r['rank'] for r in profiles}==set(range(8))
    assert all(r['world_size']==8 and r['actual_updates']==r['requested_updates']==16 and r['error'] is None and r['source_identity']==identity for r in profiles)
    measured=sum(max(r['records'][i]['seconds'] for r in profiles) for i in range(4,16))
    audio=sum(r['audio_seconds'] for r in profiles)
    return dict(source_identity=identity,world_size=8,topology='s3_t2_star_star',objective='jsd_upstream',
        profile_job=job,profile_root=str(profile_root),max_duration_per_rank=640,
        resume_comparison=result['jsd'],raw_losses_exact=True,all_rank_states_checked=True,
        full_model_updates_per_rank=16,audio_seconds_per_second=audio/measured,
        audio_seconds_per_node_second=audio/measured/8,
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in profiles),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in profiles))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile-job');ap.add_argument('--write-marker',action='store_true')
    args=ap.parse_args();marker=HERE/'EIGHT_GPU_PASS.json'
    saved=json.loads(marker.read_text()) if not args.write_marker else None
    result=verify(args.profile_job if args.profile_job else saved['profile_job'])
    if args.write_marker:marker.write_text(json.dumps(result,indent=2)+'\n')
    else:assert result==saved,'8-GPU evidence changed'
    print(json.dumps({k:v for k,v in result.items() if k!='source_identity'}))


if __name__=='__main__':main()
