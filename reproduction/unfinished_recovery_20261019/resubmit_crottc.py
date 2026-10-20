"""Resubmit the three failed pre-production jobs after fresh source validation."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'reproduction/crottc_timit_20ms_20261019'
sys.path.insert(0,str(HERE))
from common import matrix,recipe,source_hashes,write

gate=json.loads((HERE/'validation/CPU_FULL_SSL_PASS.json').read_text())
assert gate['source_sha256']==source_hashes(),'Revalidate changed source before submission'
assert gate['original_SSL_identity']['all_equal'] and gate['exact_resume_model_max_difference']==0
assert gate['validation_precision'].startswith('FP32;')
path=HERE/'dispatch.json';state=json.loads(path.read_text())
for name in matrix():
    record=state['experiments'][name]
    if record.get('recovery_attempt')==3:
        print('ALREADY_SUBMITTED',name,record['job'],flush=True);continue
    old=record['recipe'];new=recipe(name)
    assert {k:v for k,v in old.items() if k!='source_sha256'}=={k:v for k,v in new.items() if k!='source_sha256'}
    changed=[p for p,h in old['source_sha256'].items() if new['source_sha256'].get(p)!=h]
    assert set(changed)=={str(HERE/'validate.py'),str(HERE/'run_one.pbs')},changed
    assert not (HERE/'models'/name/'run_config.json').exists(),'Production already started'
    record.setdefault('previous_jobs',[]).append(dict(job=record['job'],log=record['log'],
        status='cancelled_while_queued_user_changed_walltime_to_1h',source_sha256=old['source_sha256']))
    log=HERE/(name+'.attempt3_1h.log')
    result=subprocess.run(['/opt/pbs/bin/qsub','-v','CROTTC_RECIPE='+name,'-o',str(log),str(HERE/'run_one.pbs')],capture_output=True,text=True,check=True)
    record.update(recipe=new,job=result.stdout.strip(),log=str(log),status='submitted_FP32_validation_then_exact_profile_then_training',recovery_attempt=3,walltime='01:00:00')
    state.update(updated_at_UTC=datetime.now(timezone.utc).isoformat(),walltime='01:00:00');write(path,state)
    print(name,record['job'],flush=True)
