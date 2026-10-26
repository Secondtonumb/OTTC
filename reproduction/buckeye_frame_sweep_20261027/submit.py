"""Idempotent grouped submissions, after validated replacement of the queued job."""
import fcntl
import json
import os
import re
import subprocess
from reproduction.buckeye_frame_sweep_20261027 import common


def main():
    for marker in ('CPU_PASS.json','LAUNCHER_PASS.json'):
        checked=json.loads((common.HERE/marker).read_text())
        assert checked['passed'] and checked['source_sha256']==common.source_hashes(),marker
    cancelled=json.loads((common.HERE/'REPLACED_SINGLE_JOB.json').read_text())
    assert cancelled['job_id']=='3715358.opbs' and cancelled['cancel_returncode']==0
    with (common.HERE/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=common.HERE/'SUBMISSIONS_GROUPED.json'
        state=json.loads(receipt.read_text()) if receipt.exists() else dict(
            groups={},supersedes='3715358.opbs',independent_experiments=12,
            grouped_jobs=4,GPUs_per_job=3,evaluation_uploads=False)
        for cost in common.COSTS:
            names=[f'frame_{cost}__seed{seed}' for seed in common.SEEDS]
            for name in names:
                common.write(common.HERE/'planned_configs'/f'{name}.json',common.recipe(name))
            if cost in state['groups']:
                print(cost,state['groups'][cost]['job_id'],'already submitted',flush=True)
                continue
            label={'ce':'ce','jsd_standard':'jsd','ag':'ag','jdiv':'jdiv'}[cost]
            log=common.HERE/f'group_{cost}.log'
            command=['qsub','-N',f'buck_frame_{label}3','-v',f'BUCKEYE_FRAME_COST={cost}',
                     '-o',str(log),str(common.HERE/'run.pbs')]
            result=subprocess.run(command,env=dict(os.environ,TMPDIR='/tmp'),text=True,
                                  capture_output=True,timeout=60)
            common.write(common.HERE/f'QSUB_{cost}.json',dict(command=command,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr))
            if result.returncode:
                raise RuntimeError(f'{cost}: {result.stderr} {result.stdout}')
            matches=re.findall(r'\b\d+\.opbs\b',result.stdout)
            assert len(matches)==1,result.stdout
            state['groups'][cost]=dict(job_id=matches[0],cost=cost,seeds=list(common.SEEDS),
                recipes=names,GPUs=3,nodes=3,status='submitted',command=command,log=str(log))
            common.write(receipt,state)
            print(cost,matches[0],'seeds42/43/44;3GPUs;independent world_size=1',flush=True)
    print('SUBMITTED 4 groups /12 independent experiments',flush=True)


if __name__=='__main__':
    main()
