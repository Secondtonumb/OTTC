"""Submit the two quota-interrupted S3 recipes with the user's 1h limit."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'timit_ssl_native10_20261019'
path=ORIGINAL/'dispatch.json';state=json.loads(path.read_text())
for cost in ['ce','jsd_standard']:
    name='wav2vec2__timit__10ms__s3t2ss__'+cost;record=state['experiments'][name]
    if record.get('recovery_attempt')==4:
        print('ALREADY_SUBMITTED',name,record['job'],flush=True);continue
    cfg=json.loads((ORIGINAL/'models'/name/'run_config.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['source_sha256'].items())
    record.setdefault('previous_jobs',[]).append(dict(job=record['job'],log=record['log'],status='cancelled_while_queued_user_changed_walltime_to_1h'))
    log=HERE/('s3_'+cost+'.attempt4_1h.log')
    result=subprocess.run(['/opt/pbs/bin/qsub','-v','SSL_TIMIT_RECIPE='+name,'-o',str(log),str(HERE/'recover_s3.pbs')],capture_output=True,text=True,check=True)
    record.update(job=result.stdout.strip(),log=str(log),status='submitted_first_epoch_reconstruction_then_original_resume',recovery_attempt=4,walltime='01:00:00')
    state.update(updated_at_UTC=datetime.now(timezone.utc).isoformat(),recovery_walltime='01:00:00');path.write_text(json.dumps(state,indent=2)+'\n')
    print(name,record['job'],flush=True)
