"""Instrument the accepted native phone-task loop; keep its actual updates."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'zapformer_s3t2ss_native_20261012'
sys.path.insert(0,str(NATIVE))
import runtime
import torch
import train_s3t2 as entry
from validate import check_acceptance, source_identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-duration',type=float,required=True)
    ap.add_argument('--steps',type=int,default=16);ap.add_argument('--warmup',type=int,default=4)
    args=ap.parse_args();assert args.steps>args.warmup>=1
    check_acceptance()
    epoch_marker=HERE.parent/'zapformer_s3t2ss_acceptance_20261012/results/EPOCH_PASS.json'
    assert json.loads(epoch_marker.read_text())['source_identity']==source_identity()
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if rank==0:
        assert not args.output.exists(),f'Use a new output for each independent profile: {args.output}'
        args.output.mkdir(parents=True)
    else:
        deadline=time.monotonic()+60
        while not args.output.exists():
            if time.monotonic()>deadline:raise TimeoutError('Rank 0 did not create the profile output')
            time.sleep(.1)
    records=[];current={}
    original_batch=entry.Inputs.batch
    def batch(inputs,ids):
        training='tokens' in inputs.rows[ids[0]] and inputs.rows[ids[0]].get('gradient_training_allowed',True)
        if training:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            current.clear();current['started']=time.perf_counter();current['started_wall']=time.time()
        value=original_batch(inputs,ids)
        if training:
            current.update(frames=int(value['supervisions']['num_frames'].sum()),
                padded_frames=value['inputs'].shape[0]*value['inputs'].shape[1],utterances=len(ids))
        return value
    entry.Inputs.batch=batch
    def optimizer_done(*unused):
        torch.cuda.synchronize()
        records.append(dict(update=len(records)+1,seconds=time.perf_counter()-current['started'],
            started_wall=current['started_wall'],ended_wall=time.time(),
            frames=current['frames'],padded_frames=current['padded_frames'],utterances=current['utterances'],
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    original_optimizer=entry.native.BatchedRubik
    def optimizer(*pos,**kw):
        value=original_optimizer(*pos,**kw);value.register_step_post_hook(optimizer_done);return value
    entry.native.BatchedRubik=optimizer
    # These are disposable performance probes. Avoid writing multi-GB weights;
    # this affects persistence only, after the real optimizer update completes.
    def discard_checkpoint(*pos,**kw):return None
    entry.persistence.save=discard_checkpoint
    stream=(args.output/f'gpu-rank-{rank}.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi',f'--id={local}',
        '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits','--loop-ms=500'],stdout=stream,stderr=subprocess.DEVNULL)
    sys.argv=[str(NATIVE/'train_s3t2.py'),'--objective','jsd','--topology','s3_t2_star_star','--output',str(args.output),
        '--epochs','1','--stop-after',str(args.steps),'--max-duration',str(args.max_duration),
        '--save-every','100000','--eval-every','100000','--average-period','200','--eval-limit','4']
    error=None
    try:entry.main()
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        monitor.terminate();monitor.wait(timeout=10);stream.close()
        measured=records[args.warmup:]
        summary=dict(rank=rank,world_size=world,max_duration=args.max_duration,
            warmup_updates=args.warmup,requested_updates=args.steps,actual_updates=len(records),
            scope='full Libri100 manifest; full model; native actual JSD/auxiliary/BF16/Rubik updates; no weight persistence',
            timing='input load through optimizer post hook with CUDA synchronization; evaluation/checkpoint overhead excluded',
            source_identity=source_identity(),profile_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            records=records,error=error)
        if measured:
            seconds=sum(r['seconds'] for r in measured);audio=sum(r['frames'] for r in measured)/100.
            summary.update(measured_seconds=seconds,audio_seconds=audio,audio_seconds_per_second=audio/seconds,
                median_update_seconds=statistics.median(r['seconds'] for r in measured),
                padding_fraction=1-sum(r['frames'] for r in measured)/sum(r['padded_frames'] for r in measured),
                peak_allocated_gib=max(r['gpu_peak_allocated_bytes'] for r in measured)/2**30,
                peak_reserved_gib=max(r['gpu_peak_reserved_bytes'] for r in measured)/2**30)
            samples=[]
            for row in csv.reader((args.output/f'gpu-rank-{rank}.csv').read_text().splitlines()):
                try:
                    timestamp=datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
                    if any(r['started_wall']<=timestamp<=r['ended_wall'] for r in measured):samples.append(float(row[1]))
                except (ValueError,IndexError):continue
            summary.update(gpu_utilization_samples=len(samples),gpu_utilization_percent_mean=statistics.mean(samples) if samples else None)
        (args.output/f'profile-rank-{rank}.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k not in ('records','source_identity')}),flush=True)


if __name__=='__main__':main()
