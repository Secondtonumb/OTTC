#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == '--_batch-worker' ]]; then
    shift
    source "$script_dir/../libri960_continue5_20261017/environment.sh"
    exec "$LIBRI_PYTHON" -u - "$script_dir/run_interactive.sh" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone

launcher = Path(sys.argv[1]).resolve()
target = sys.argv[2]
requested = int(os.environ['BPE_RUNTIME_BATCH_SIZE']) if os.environ.get('BPE_RUNTIME_BATCH_SIZE') else None
assert requested is None or 1 <= requested <= 64
sys.path.insert(0, str(launcher.parents[2]))
from reproduction.timit_bpe500_20261020 import train as recipe


def set_batch(collator, logical_batch, requested_batch):
    actual = min(logical_batch, requested_batch)
    assert actual >= 1
    collator.micro = actual
    return actual


if target == '--self-test':
    for logical, selected, count in [(64, 64, 1), (64, 16, 4), (4, 64, 1)]:
        collator = recipe.base.ChunkedCollator(lambda rows: rows, 2)
        assert set_batch(collator, logical, selected) == min(logical, selected)
        batch = collator(list(range(logical)))
        assert batch['logical_batch_size'] == logical
        assert len(batch['microbatches']) == count
        assert [item for chunk in batch['microbatches'] for item in chunk] == list(range(logical))
        assert len(collator([0, 1, 2])['microbatches']) == 1
    print('PASS: physical batch 64, batch 16, diagnostic batch 4 and short-tail handling')
    sys.exit(0)

if '--check' not in sys.argv[3:]:
    from transformers import Trainer
    from reproduction.checkpoint_retention import CheckpointRetentionCallback
    original_init = Trainer.__init__

    def initialize(self, *args, **kwargs):
        settings = kwargs.get('args')
        collator = kwargs.get('data_collator')
        if settings is None or collator is None or not hasattr(collator, 'micro'):
            raise ValueError('Unexpected Trainer interface; batch override was not applied')
        actual = set_batch(collator, settings.per_device_train_batch_size,
                   requested if requested is not None else collator.micro)
        original_init(self, *args, **kwargs)
        output = Path(settings.output_dir)
        retention = CheckpointRetentionCallback(output)
        self.add_callback(retention)
        content = launcher.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        snapshots = output / 'runtime_batching'
        snapshots.mkdir(parents=True, exist_ok=True)
        snapshot = snapshots / (digest + '.sh')
        if not snapshot.exists():
            snapshot.write_bytes(content)
        record = dict(requested_batch_size=requested, actual_train_microbatch=actual,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            launcher_sha256=digest, timestamp_utc=datetime.now(timezone.utc).isoformat(),
            protocol='Optional runtime microbatch override; frozen run_config retains recipe defaults. Logical batch, loss reduction and optimizer schedule unchanged.')
        with (output / 'runtime_batching.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        wandb = sys.modules.get('wandb')
        if wandb is not None and wandb.run is not None:
            wandb.run.config.update({'runtime_batching': record,
                'runtime_checkpoint_retention': retention.policy}, allow_val_change=True)
        print('Runtime batching: ' + json.dumps(record), flush=True)

    Trainer.__init__ = initialize

sys.argv = [target, *sys.argv[3:]]
runpy.run_path(target, run_name='__main__')
PY
fi
batch_size=''
batch_requested=0
forwarded=()
while (( $# )); do
    case "$1" in
        --batch-size)
            if (( $# < 2 )); then printf '%s\n' '--batch-size requires a value' >&2; exit 2; fi
            batch_size=$2
            batch_requested=1
            shift 2
            ;;
        --batch-size=*) batch_size=${1#*=}; batch_requested=1; shift ;;
        --help|-h)
            printf 'Usage: bash %s [--probe] {jsd|ce|both} [total_seconds] [--batch-size N]\n' "$0"
            printf 'Default: full TIMIT training, 40 epochs per cost, total_seconds=7200.\n'
            printf '%s\n' '--batch-size N: physical training microbatch, 1-64; full logical batch stays 64.'
            printf 'Omit --batch-size to retain the original microbatch setting. Small diagnostic batches are not enlarged.\n'
            printf 'Checkpoint retention: dev-best 3 epochs plus latest 10; preserve best, references and full resume states.\n'
            printf 'Checkpoint resume is automatic; a time budget is not an epoch limit.\n'
            printf 'Optional --probe: legacy 200-step test, total_seconds=2400.\n'
            printf 'Run inside an allocated PBS GPU interactive session.\n'
            exit 0
            ;;
        *) forwarded+=("$1"); shift ;;
    esac
done
if (( batch_requested )); then
    if [[ ! "$batch_size" =~ ^[1-9][0-9]?$ ]] || (( batch_size > 64 )); then
        printf '%s\n' '--batch-size must be an integer between 1 and 64' >&2
        exit 2
    fi
fi
set -- "${forwarded[@]}"
runner="$script_dir/full/run_interactive.sh"
if [[ -f "$script_dir/full/recovery/READY.json" ]]; then
    export BPE_FULL_RECOVERY=1
fi
if [[ "${1:-}" == '--probe' ]]; then
    shift
    runner="$script_dir/run_probe_interactive.sh"
fi
export BPE_RUNTIME_BATCH_SIZE="$batch_size" BPE_BATCH_LAUNCHER="$script_dir/run_interactive.sh"
runner_code=$(<"$runner")
invocation='"$LIBRI_PYTHON" -u "$recipe_root/train.py"'
replacement='bash "$BPE_BATCH_LAUNCHER" --_batch-worker "$recipe_root/train.py"'
if [[ "$runner_code" != *"$invocation"* ]]; then
    printf 'Training launcher changed; cannot install runtime batching and checkpoint retention\n' >&2
    exit 2
fi
runner_code=${runner_code//"$invocation"/"$replacement"}
exec bash -s -- "$@" <<< "$runner_code"