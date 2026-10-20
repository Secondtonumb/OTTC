"""Submit exactly the three user-authorized TIMIT20ms mono CROTTC runs."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
from common import HERE,ROOT,recipe,matrix,write,source_hashes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    path=HERE/'dispatch.json';state=json.loads(path.read_text()) if path.exists() else dict(experiments={})
    state.update(updated_at_UTC=datetime.now(timezone.utc).isoformat(),authorized_experiments=3,
        walltime='02:00:00',GPUs_per_job=1,epochs=40,purpose='TIMIT20ms mono dual-view CROTTC: upstream-JSD / CE / standard-JSD')
    if args.submit:
        gate=json.loads((HERE/'validation/CPU_FULL_SSL_PASS.json').read_text())
        assert gate['source_sha256']==source_hashes(),'CPU/source validation stale'
        assert gate['original_SSL_identity']['all_equal'] and gate['exact_resume_model_max_difference']==0
    for name in matrix():
        record=state['experiments'].setdefault(name,dict(status='prepared',recipe=recipe(name)))
        assert record['recipe']==recipe(name),'Source/config changed after preparation'
        if args.submit and not record.get('job'):
            log=HERE/(name+'.log')
            result=subprocess.run(['/opt/pbs/bin/qsub','-v','CROTTC_RECIPE='+name,'-o',str(log),str(HERE/'run_one.pbs')],capture_output=True,text=True)
            if result.returncode:
                record.update(status='submission_failed',error=result.stderr.strip());write(path,state);raise RuntimeError(result.stderr)
            record.update(status='submitted_validation_then_profile_then_training',job=result.stdout.strip(),log=str(log))
        write(path,state)
    print(json.dumps({k:dict(status=v['status'],job=v.get('job')) for k,v in state['experiments'].items()},indent=2))


if __name__=='__main__':main()
