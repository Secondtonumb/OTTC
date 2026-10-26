"""Submit the smaller-batch continuation, retaining the original resume point."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

from reproduction.buckeye_full40_retry48_20261026.runtime import HERE, MICRO, hashes
from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.checkpoint_retention import verify_saved_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--continue-job')
    args = parser.parse_args()
    with (HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = HERE / 'SUBMISSION.json'
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        generation = 0
        if args.continue_job:
            assert previous and previous['job_id'] == args.continue_job
            records = [json.loads(p.read_text()) for p in (HERE / 'jobs' / args.continue_job).glob('*/train_EXIT.json')]
            assert len(records) == 12 and all(r['exit_code'] in (0, 75) for r in records), records
            if not any(r['exit_code'] == 75 for r in records):
                assert json.loads((common.HERE / 'COMPARISON.json').read_text())['complete']
                return
            generation = previous['generation'] + 1
            assert generation <= 10
        elif previous:
            print(json.dumps(previous, indent=2))
            return
        pending = HERE / 'SUBMISSION_PENDING.json'
        assert not pending.exists(), 'Uncertain previous submission; inspect before retry'
        _, _, contracts, _ = common.checked_rows()
        experiments = []
        for cost in common.COSTS:
            for seed in common.SEEDS:
                name = f'{cost}__seed{seed}'
                output = common.HERE / 'models' / name
                config = json.loads((output / 'run_config.json').read_text())
                assert config == common.recipe(name, contracts), name
                # Preserve the original step 285 without duplicating tensor storage.
                checkpoint = HERE / 'initial_checkpoints' / name / 'checkpoint-285'
                if not checkpoint.exists():
                    source = output / 'checkpoint-285'
                    verify_saved_checkpoint(output, 285)
                    checkpoint.mkdir(parents=True)
                    for member in source.iterdir():
                        assert member.is_file() and not member.is_symlink()
                        os.link(member, checkpoint / member.name)
                verify_saved_checkpoint(checkpoint.parent, 285)
                state = json.loads((checkpoint / 'trainer_state.json').read_text())
                assert state['global_step'] == 285 and state['epoch'] == 1.
                experiments.append(dict(rank=len(experiments), recipe=name, preserved_resume=str(checkpoint)))
        command = ['qsub', '-o', str(HERE / f'allocation_{generation}.log'), str(HERE / 'run.pbs')]
        plan = dict(command=command, generation=generation, physical_batch=MICRO,
                    logical_batch=64, validation_batch=MICRO, independent_world_size=1,
                    experiments=experiments, runtime_sha256=hashes(), evaluation_uploads=False,
                    resume_from_job='3706138.opbs', gate_required=True)
        common.base.write(HERE / f'PLAN_{generation}.json', plan)
        if not (args.submit or args.continue_job):
            print(json.dumps(plan, indent=2))
            return
        common.base.write(pending, plan)
        result = subprocess.run(command, text=True, capture_output=True)
        common.base.write(HERE / f'QSUB_RESPONSE_{generation}.json',
                          dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, result.stderr
        match = re.search(r'\b(\d+\.opbs)\b', result.stdout)
        assert match, result.stdout
        if previous:
            common.base.write(HERE / f"SUBMISSION_{previous['job_id']}.json", previous)
        common.base.write(receipt, dict(**plan, job_id=match.group(1), status='submitted_not_completed'))
        pending.unlink()
        print(match.group(1))


if __name__ == '__main__':
    main()
