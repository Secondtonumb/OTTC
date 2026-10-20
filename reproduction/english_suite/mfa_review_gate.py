"""Fail closed on short-run restoration/update/profile evidence."""
import json
import math
import statistics
import sys
import subprocess
from pathlib import Path

root = Path(sys.argv[1])
report = {}
for arm, start in [('ctc', 5253), ('ottc', 5253), ('spottc', 3444), ('film_k4', 0)]:
    p = root / arm
    loaded = json.loads((p / 'loaded_state.json').read_text())
    config = json.loads((p / 'run_config.json').read_text())
    events = [json.loads(line) for line in (p / 'updates.jsonl').read_text().splitlines()]
    updates = [r for r in events if r['event'] == 'update']
    saved = [r for r in events if r['event'] == 'checkpoint_state'][-1]
    assert loaded['progress']['global_step'] == start
    assert [r['step'] for r in updates] == list(range(start + 1, start + 31))
    assert config['world_size'] == 8
    assert config['contract']['manifest_sha256'] == 'ed40c32ca9b16119fafce8a1dab3131eb814df4a0f9014516bc767e3b9bec305'
    assert saved['model_digest'] != loaded['model_digest'], 'No parameter update'
    assert saved['optimizer_digest'] != loaded['optimizer_digest'], 'No optimizer update'
    assert (p / f'step-{start+30:09d}.pt').stat().st_size > 1000000
    for r in updates:
        assert all(math.isfinite(r[k]) for k in ['loss', 'step_seconds', 'input_wait_max', 'grad_norm_rank0'])
        assert r['grad_norm_rank0'] > 0 and r['audio_seconds'] > 0
        assert len(r['rank_ids']) == 8 and all(r['rank_ids'])
    steady = updates[3:]
    report[arm] = dict(start=start, end=start+30, updates=30,
        mean_step_seconds=statistics.mean(r['step_seconds'] for r in steady),
        mean_input_wait_max=statistics.mean(r['input_wait_max'] for r in steady),
        audio_seconds_per_wall_second=sum(r['audio_seconds'] for r in steady)/sum(r['step_seconds'] for r in steady),
        peak_allocated_GiB=max(r['peak_allocated_max'] for r in updates)/2**30,
        peak_reserved_GiB=max(r['peak_reserved_max'] for r in updates)/2**30,
        final_loss=updates[-1]['loss'], parameter_update=True, optimizer_update=True,
        manifest_sha256=config['contract']['manifest_sha256'])
    if arm in ('ctc', 'film_k4'):
        reload = json.loads((root / (arm+'_reload') / 'loaded_state.json').read_text())
        for key in ('model_digest', 'optimizer_digest', 'scheduler'):
            assert reload[key] == saved[key], (arm, key)
        assert reload['progress']['global_step'] == start+30
        revents = [json.loads(s) for s in (root/(arm+'_reload')/'updates.jsonl').read_text().splitlines()]
        assert [r['step'] for r in revents if r['event']=='update'] == [start+31]
        report[arm]['saved_state_reload_match'] = True
(root/'GATE_PASS.json').write_text(json.dumps(dict(status='PASS', arms=report, discarded_warmup_steps=3,
    protocol='Real full-data 8-GPU training, conservative microbatches; no evaluation/PER claim'), indent=2)+'\n')
print(json.dumps(report, indent=2))
# User requested a bounded batch-size pilot only; do not submit production jobs.
print('Production submission disabled: batch-size pilot only.', flush=True)
