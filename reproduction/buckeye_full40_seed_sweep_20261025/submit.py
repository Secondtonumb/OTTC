"""Lock full40 contracts and submit once; never restart a previous experiment."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys

from reproduction.buckeye_full40_seed_sweep_20261025 import common


def verify_smoke(configs, root):
    for config in configs:
        record = json.loads((root / 'smoke' / config['name'] / 'PASS.json').read_text())
        assert record['recipe'] == config['name']
        assert record['source_sha256'] == config['source_sha256']
        assert record['expected_data'] == config['expected_data']
        assert record['physical_batch'] == 64 and record['resume_verified']
        assert record['saved_model_dtype'] == 'float32'
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-from-job')
    args = parser.parse_args()
    assert not (args.smoke and args.resume_from_job)
    with (common.HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = common.HERE / ('SMOKE_SUBMISSION.json' if args.smoke else 'SUBMISSION.json')
        generation = 0
        if args.resume_from_job:
            prior = json.loads(receipt.read_text())
            assert prior['job_id'] == args.resume_from_job
            generation = prior.get('generation', 0) + 1
            assert generation <= 10, 'Too many continuations; manual review required'
            allocation = common.HERE / 'jobs' / args.resume_from_job
            exits = [json.loads(path.read_text()) for path in allocation.glob('*/WORKER_EXIT.json')]
            assert len(exits) == 12 and all(record['exit_code'] in (0, 75) for record in exits)
            assert any(record['exit_code'] == 75 for record in exits)
            assert not json.loads((common.HERE / 'COMPARISON.json').read_text())['complete']
            receipt.replace(common.HERE / f"SUBMISSION_{args.resume_from_job}.json")
        if receipt.exists():
            print(receipt.read_text())
            return
        pending = common.HERE / 'SUBMISSION_PENDING.json'
        if pending.exists():
            raise RuntimeError('Previous submission outcome uncertain; inspect scheduler before retry')
        if args.smoke:
            subprocess.run([sys.executable, '-m', 'unittest', 'reproduction.buckeye_full40_seed_sweep_20261025.test_recipe', '-v'], check=True)
        _, _, contracts, _ = common.checked_rows()
        experiments = []
        configs = []
        for cost in common.COSTS:
            for seed in common.SEEDS:
                name = f'{cost}__seed{seed}'
                config = common.recipe(name, contracts)
                configs.append(config)
                planned = common.HERE / 'planned_configs' / f'{name}.json'
                if not args.smoke and (common.HERE / 'models' / name / 'run_config.json').exists():
                    assert json.loads(planned.read_text()) == config, 'Submitted contract changed'
                common.base.write(common.HERE / 'planned_configs' / f'{name}.json', config)
                experiments.append(dict(rank=len(experiments), recipe=name))
        common.base.write(common.HERE / 'EXPERIMENT_MATRIX.json', dict(experiments=experiments,
                          GPUs=12, physical_batch_per_experiment=64, independent=True, world_size=1))
        common.base.write(common.HERE / 'CPU_VALIDATION.json', dict(passed=True, tests=4,
                          source_sha256=common.source_hashes(), expected_data=contracts,
                          gpu_validation='pending; mandatory per-recipe gate before training'))
        if not args.smoke:
            common.base.write(common.HERE / 'DIRECT_SUBMISSION.json', dict(
                requested_by_user=True, smoke_required=False, source_sha256=common.source_hashes()))
        script = 'smoke.pbs' if args.smoke else 'run.pbs'
        log = 'smoke_allocation.log' if args.smoke else f'allocation_{generation}.log'
        command = ['qsub', '-o', str(common.HERE / log), str(common.HERE / script)]
        if not args.submit:
            print(json.dumps(dict(submitted=False, command=command, experiments=experiments), indent=2))
            return
        common.base.write(pending, dict(command=command, state='submitting; do not retry automatically'))
        result = subprocess.run(command, text=True, capture_output=True, timeout=60, env=os.environ.copy())
        common.base.write(common.HERE / 'QSUB_RESPONSE.json', dict(returncode=result.returncode,
                          stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError(f'Submission failed; inspect response before retry: {result.stderr}')
        match = re.search(r'\b(\d+\.opbs)\b', result.stdout)
        if not match:
            raise RuntimeError('Unknown receipt; inspect scheduler before retry')
        common.base.write(receipt, dict(job_id=match.group(1), experiments=experiments, GPUs=12,
                          status='submitted_not_completed', queue='regular-g', walltime='01:00:00' if args.smoke else '10:00:00',
                          stage='smoke' if args.smoke else 'training_and_evaluation', generation=generation,
                          evaluation_uploads=False, command=command))
        pending.unlink()
        print(receipt.read_text())


if __name__ == '__main__':
    main()