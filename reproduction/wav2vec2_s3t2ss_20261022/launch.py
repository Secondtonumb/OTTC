"""Submit two independent experiments after the CPU/source gates pass."""
import argparse
import datetime
import json
import os
import subprocess
from common import HERE, matrix, source_hashes, write

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    hashes=source_hashes()
    for target in ('TOPOLOGY_PASS.json','validation/CPU_FULL_SSL_PASS.json'):
        assert json.loads((HERE/target).read_text())['source_sha256']==hashes,target
    assert json.loads((HERE/'SOURCE_CONTRACT.json').read_text())==hashes
    dispatch=HERE/'dispatch.json'
    records=json.loads(dispatch.read_text()) if dispatch.exists() else {}
    for name in matrix():
        if name in records:continue
        command=['/opt/pbs/bin/qsub','-N','s3ss_'+name.split('__')[-1],
            '-v','SSL_TIMIT_RECIPE='+name,'-o',str(HERE/(name+'.log')),str(HERE/'run_one.pbs')]
        print(command,flush=True)
        if args.submit:
            result=subprocess.run(command,text=True,capture_output=True,env=dict(os.environ,TMPDIR='/tmp'))
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            job=result.stdout.strip();assert job.endswith('.opbs'),result.stdout
            records[name]=dict(job_id=job,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                state='submitted; GPU validation/profile and training not yet verified',command=command,
                source_sha256=hashes,evaluation_uploads=False)
            write(dispatch,records);print(name,job,flush=True)
