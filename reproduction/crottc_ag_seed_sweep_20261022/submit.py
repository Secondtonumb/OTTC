"""Submit only this authorized four-seed sweep, preserving partial submissions."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
gate=json.loads((HERE/'validation/CPU_PASS.json').read_text())
preflight=json.loads((HERE/'PREFLIGHT.json').read_text())
assert len(gate['four_seed_recipes'])==4 and gate['exact_resume_model_max_difference']==0
for source,expected in gate['source_sha256'].items():
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,source
record_path=HERE/'SUBMISSIONS.json'
record=json.loads(record_path.read_text()) if record_path.exists() else dict(jobs={},purpose='TIMIT matched10ms CROTTC+AG seeds42-45,40epoch,batch64',profile_gate='required before formal training')
(HERE/'jobs').mkdir(exist_ok=True)
for config in preflight['configs']:
    name=config['name']
    if name in record['jobs']:
        print('ALREADY_SUBMITTED',name,record['jobs'][name]['job'],flush=True)
        continue
    assert config['source_sha256']==gate['source_sha256']
    command=['qsub','-N',f"croAG_s{config['seed']}",'-v',f"CROTTC_SEED={config['seed']}",'-o',str(HERE/'jobs'),str(HERE/'run.pbs')]
    result=subprocess.run(command,env=dict(os.environ,TMPDIR='/tmp'),text=True,capture_output=True,check=True)
    job=result.stdout.strip()
    assert job.endswith('.opbs') and job.split('.')[0].isdigit(),job
    record['jobs'][name]=dict(job=job,seed=config['seed'],command=command,
        submitted_at=datetime.now(timezone.utc).isoformat(),status='submitted',training_complete=False)
    temporary=record_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(record_path)
    print(name,job,flush=True)
