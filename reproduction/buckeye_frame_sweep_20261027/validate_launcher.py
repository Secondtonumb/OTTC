"""Check all twelve actual worker dispatches and contaminated MPI environments."""
import os
import subprocess
from reproduction.buckeye_frame_sweep_20261027 import common


def main():
    records=[]
    for cost in common.COSTS:
        for rank,seed in enumerate(common.SEEDS):
            poisoned=dict(os.environ,BUCKEYE_FRAME_COST=cost,OMPI_COMM_WORLD_RANK=str(rank),
                OMPI_COMM_WORLD_SIZE='3',OMPI_COMM_WORLD_LOCAL_RANK='0',RANK=str(rank),WORLD_SIZE='3',
                LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',MASTER_ADDR='unused',MASTER_PORT='12345',
                PMI_RANK=str(rank),PMI_SIZE='3',PMIX_RANK=str(rank))
            checked=subprocess.run(['bash',str(common.HERE/'worker.sh'),'--dry-run'],
                env=poisoned,text=True,capture_output=True,check=True)
            fields=dict(word.split('=',1) for word in checked.stdout.strip().split())
            assert fields['recipe']==f'frame_{cost}__seed{seed}'
            assert fields['world_size']==fields['GPUs']=='1'
            assert fields['batch']=='64' and fields['epochs']=='40'
            records.append(fields)
    assert sorted(r['recipe'] for r in records)==sorted(common.matrix())
    for rank,cost in [('3','ag'),('-1','ag'),('0','unknown')]:
        checked=subprocess.run(['bash',str(common.HERE/'worker.sh'),'--dry-run'],
            env=dict(os.environ,OMPI_COMM_WORLD_RANK=rank,BUCKEYE_FRAME_COST=cost),
            text=True,capture_output=True)
        assert checked.returncode!=0
    for name in ('run.pbs','worker.sh'):
        subprocess.run(['bash','-n',str(common.HERE/name)],check=True)
    common.write(common.HERE/'LAUNCHER_PASS.json',dict(passed=True,source_sha256=common.source_hashes(),
        assignments=records,poisoned_MPI_environment_removed=True,invalid_rank_and_cost_rejected=True,
        grouped_jobs=4,GPUs_per_job=3,independent_seeds_per_job=3,GPU_execution_tested=False))
    print('LAUNCHER_PASS:4jobs x3GPUs;12 independent seeds;world_size=1')


if __name__=='__main__':
    main()
