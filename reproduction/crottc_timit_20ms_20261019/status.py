"""Artifact-first status; no scheduler state alone establishes completion."""
from datetime import datetime,timezone
import json
import subprocess
from common import HERE,matrix,write


def main():
    dispatch=json.loads((HERE/'dispatch.json').read_text()) if (HERE/'dispatch.json').exists() else {'experiments':{}}
    report={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'experiments':{}}
    for name in matrix():
        entry=dispatch['experiments'].get(name,{});output=HERE/'models'/name;profile=HERE/'profiles'/name
        job=entry.get('job');continuation=output/'continuation_job.txt'
        if continuation.exists():job=continuation.read_text().strip() or job
        item=dict(job=job,epoch_checkpoints=len(list((output/'epoch_weights').glob('epoch-*.pt'))),
            profile_pass=(profile/'PASS.json').exists(),training_complete=(output/'TRAIN_COMPLETE.json').exists(),
            benchmark_complete=(output/'BENCHMARK_COMPLETE.json').exists(),log=entry.get('log'))
        if (output/'LATEST_STATE.json').exists():item['latest']=json.loads((output/'LATEST_STATE.json').read_text())
        history=output/'history.jsonl'
        if history.exists():
            last=history.read_text().splitlines()[-1:]
            if last:item['last_history']=json.loads(last[0])
        item['phase']='benchmark_complete' if item['benchmark_complete'] else 'training_complete_benchmark_pending' if item['training_complete'] else 'production_training' if history.exists() else 'profile_pass_training_pending' if item['profile_pass'] else 'profiling' if (profile/'batches.jsonl').exists() else 'queued_or_validation_pending'
        if job:
            try:
                result=subprocess.run(['qstat','-f',job],capture_output=True,text=True)
                item['qstat_returncode']=result.returncode
                item['scheduler_fields']=[line.strip() for line in result.stdout.splitlines() if any(k in line for k in ['job_state =','exec_host =','resources_used.walltime =','comment =','estimated.start_time ='])]
                if result.stderr:item['scheduler_error']=result.stderr.strip()
            except PermissionError as e:item['scheduler_error']=str(e)
        report['experiments'][name]=item
    write(HERE/'STATUS.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
