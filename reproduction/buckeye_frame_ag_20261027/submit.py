"""Idempotent single-node submission after the full CPU gate."""
import fcntl
import json
import os
import re
import subprocess
from reproduction.buckeye_frame_ag_20261027 import common


def main():
    with (common.HERE/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        gate=json.loads((common.HERE/'CPU_PASS.json').read_text())
        assert gate['passed'] and gate['source_sha256']==common.source_hashes()
        receipt=common.HERE/'SUBMISSION.json'
        if receipt.exists():
            print(receipt.read_text())
            return
        config=common.recipe(common.NAME)
        common.write(common.HERE/'planned_config.json',config)
        command=['qsub','-o',str(common.HERE/'allocation.log'),str(common.HERE/'run.pbs')]
        result=subprocess.run(command,env=dict(os.environ,TMPDIR='/tmp'),capture_output=True,text=True)
        record=dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    recipe=common.NAME,source_sha256=common.source_hashes())
        if result.returncode:
            common.write(common.HERE/'SUBMISSION_FAILED.json',record)
            raise RuntimeError(result.stderr)
        matches=re.findall(r'\b\d+\.opbs\b',result.stdout)
        assert len(matches)==1,record
        record['job_id']=matches[0]
        common.write(receipt,record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
