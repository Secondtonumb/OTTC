"""Publish the fresh four-example MFA evaluation summary, without raw audio."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIZ = ROOT / 'reproduction/segmentation_figures_20261017'


def main():
    from reproduction.wandb_policy import init_evaluation
    os.environ.setdefault('WANDB_CACHE_DIR', str(HERE / 'wandb_cache'))
    os.environ.setdefault('WANDB_DATA_DIR', str(HERE / 'wandb_data'))
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    result = json.loads((HERE / 'RESULTS_COMPLETE.json').read_text())
    examples = json.loads((VIZ / 'examples.json').read_text())['examples']
    columns = ['recording_id', 'split', 'tolerance_ms', 'precision', 'recall', 'F1', 'R_value']
    rows = []
    for rid, ex in examples.items():
        model = ex['models']['MFA 2.2.17 (word FA)']
        assert model['text_free'] is False
        for ms, m in model['metrics'].items():
            rows.append([rid, ex['split'], int(ms), *[m[k] for k in ['precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct']]])
    summary = dict(scope='four visualization examples, not full test sets',
                   protocol='shared figure references, unique interval edges, utterance edge exclusion20ms, one-to-one matching',
                   text_free=False, rows=[dict(zip(columns, row)) for row in rows])
    (HERE / 'FIGURE_BOUNDARY_METRICS.json').write_text(json.dumps(summary, indent=2) + '\n')
    identity_path = HERE / 'wandb_run.json'
    saved = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    run_id = saved.get('id') or wandb.util.generate_id()
    kwargs = dict(id=run_id, entity=defaults['entity'], project=defaults['project'],
        name='Fresh MFA 2.2.17 + official ARPA v3 | 4 visualization examples',
        group='mfa-direct-20261017', job_type='forced-alignment-evaluation',
        tags=['fresh-inference', 'word-transcript', 'four-examples-only'], dir=str(HERE),
        config=dict(logging_origin='live alignment evaluation', scope=summary['scope'],
                    tool_version=result['tool_version'], acoustic_model=result['acoustic_model'],
                    dictionary=result['dictionary'],
                    input='audio plus original word transcript; no reference phones or timestamps',
                    training_performed=False, claimed_paper_mfa3_reproduction=False),
        settings=wandb.Settings(init_timeout=20, disable_code=True, x_disable_stats=True))
    mode = defaults.get('mode', 'online')
    try:
        run = init_evaluation(mode=mode, resume='allow', **kwargs)
    except wandb.errors.Error:
        mode = 'offline'
        run = init_evaluation(mode=mode, **kwargs)
    run.log({'MFA/selected_examples_boundary_metrics': wandb.Table(columns=columns, data=rows)})
    run.summary.update(dict(aligned_utterances=4, scope=summary['scope'], text_free=False,
                            tool_version=result['tool_version']))
    # Upload scalar/table benchmark summaries only; native TextGrids and
    # all derived audio/transcript artifacts remain local.
    identity = dict(id=run_id, entity=defaults['entity'], project=defaults['project'],
                    mode=mode, url=run.url if mode == 'online' else None)
    run.finish()
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
