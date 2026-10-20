"""Submit only the six explicitly authorized fresh-SSL TIMIT experiments."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from common import HERE,ROOT,recipe,matrix,write,source_hashes,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    path=HERE/'dispatch.json';state=json.loads(path.read_text()) if path.exists() else dict(experiments={})
    state.update(updated_at_UTC=datetime.now(timezone.utc).isoformat(),authorized_experiments=6,
        walltime='02:00:00',GPUs_per_job=1,total_independent_jobs=6,epochs=40,
        purpose='User-defined from-scratch: original SSL encoder; trainable native10ms final CNN and fresh OTTC heads',
        profile_gate='Each exact recipe must pass full real-data forward/backward/optimizer/longest-input/K2 checks before production',
        no_Buckeye_or_Libri_submission=True)
    if args.submit:
        gate=json.loads((HERE/'validation/CPU_FULL_SSL_PASS.json').read_text())
        assert gate['source_sha256']==source_hashes(),'CPU/source gate stale'
        assert gate['original_SSL_identity']['all_equal'] and gate['exact_resume_model_max_difference']==0
    # Reuse is inappropriate when the actual initialization/trainability differs.
    previous=ROOT/'reproduction/native_stride_ottc_20261018/models'
    audit={}
    for name in matrix():
        old=previous/name/'run_config.json'
        cfg=json.loads(old.read_text())
        assert 'Libri960' in cfg['initialization'] and cfg['freeze_cnn'] is True
        audit[name]=dict(previous_config=str(old),sha256=sha(old),matching=False,
            reason='previous10ms inherits supervised Libri960 OTTC and freezes all CNN; new original-SSL task has trainable last CNN and all heads')
        record=state['experiments'].setdefault(name,dict(status='prepared',recipe=recipe(name)))
        assert record['recipe']==recipe(name),'Submitted source/config contract changed'
        if args.submit and not record.get('job'):
            log=HERE/(name+('.attempt2.log' if record.get('previous_jobs') else '.log'))
            result=subprocess.run(['/opt/pbs/bin/qsub','-l','walltime=02:00:00','-v','SSL_TIMIT_RECIPE='+name,
                '-o',str(log),str(HERE/'run_one.pbs')],capture_output=True,text=True)
            if result.returncode:
                record.update(status='submission_failed',error=result.stderr.strip());write(path,state)
                raise RuntimeError(result.stderr)
            record.update(status='submitted_GPU_validation_then_profile_then_training',job=result.stdout.strip(),log=str(log))
        write(path,state)
    write(HERE/'REUSE_AUDIT.json',audit)
    print(json.dumps({k:dict(status=v['status'],job=v.get('job')) for k,v in state['experiments'].items()},indent=2))


if __name__=='__main__':main()
