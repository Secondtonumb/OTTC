"""Artifact-first status of the authorized training matrix; optional PBS refresh."""
from common import HERE,write
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scheduler',action='store_true');args=parser.parse_args()
    dispatch=json.loads((HERE/'dispatch.json').read_text());report={}
    for name,record in dispatch['experiments'].items():
        output=HERE/'models'/name;profile=HERE/'profiles'/name
        info=dict(submission=record['status'],job=record.get('job'),recipe=name)
        if record['status']=='held_until_TIMIT_and_Buckeye_results_reviewed':
            info.update(status=record['status'],gate=record['gate']);report[name]=info;continue
        if record['status']=='reused_completed_training':
            info.update(status='reused_completed_training',checkpoint=record['checkpoint']);report[name]=info;continue
        job=record.get('job')
        if (output/'continuation_job.txt').exists():job=(output/'continuation_job.txt').read_text().strip();info['job']=job
        if args.scheduler and job:
            result=subprocess.run([shutil.which('qstat') or '/usr/local/bin/qstat','-f',job],capture_output=True,text=True)
            fields={}
            for key in ['job_state','exec_host','Resource_List.walltime','resources_used.walltime','queue','comment']:
                match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(.*)$',result.stdout,re.M)
                if match:fields[key]=match.group(1)
            info['scheduler']=fields if fields else dict(query_error=result.stderr.strip())
        info['profile_passed']=(profile/'PASS.json').exists()
        if (output/'BENCHMARK_COMPLETE.json').exists():info['status']='training_and_local_benchmarks_complete'
        elif (output/'TRAIN_COMPLETE.json').exists():info['status']='training_complete_benchmarks_pending'
        elif (output/'history.jsonl').exists():info['status']='training_started_not_complete'
        elif (output/'run_config.json').exists():info['status']='training_initialization'
        elif info['profile_passed']:info['status']='profile_passed_training_pending'
        elif (profile/'batches.jsonl').exists():info['status']='profiling_not_yet_passed'
        else:info['status']='submitted_or_prepared_no_training_artifacts'
        if (output/'history.jsonl').exists():
            lines=(output/'history.jsonl').read_text().splitlines()
            if lines:info['latest_logged_training']=json.loads(lines[-1])
        if (output/'LATEST_STATE.json').exists():info['latest_saved_training']=json.loads((output/'LATEST_STATE.json').read_text())
        if (output/'wandb_run.json').exists():info['wandb']=json.loads((output/'wandb_run.json').read_text())
        report[name]=info
    value=dict(updated_at_UTC=datetime.now(timezone.utc).isoformat(),experiments=report)
    write(HERE/'STATUS.json',value)
    from collections import Counter
    print(json.dumps(dict(Counter(r['status'] for r in report.values())),indent=2))


if __name__=='__main__':main()
