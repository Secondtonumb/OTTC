"""Render and log derived comparison tables, preserving original training runs."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproduction/all_models_comparison_20261017'


def main():
    from reproduction.wandb_policy import init_evaluation
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
    os.environ.setdefault('WANDB_CACHE_DIR', str(OUT / 'wandb_cache'))
    os.environ.setdefault('WANDB_DATA_DIR', str(OUT / 'wandb_data'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audit = json.loads((OUT / 'metrics.json').read_text())
    groups = [('TIMIT → core-test192', 'TIMIT core test'),
              ('Buckeye → test1840', 'Buckeye test (human boundaries)'),
              ('Libri100 → test-clean2620', 'Libri test-clean (automatic reference)'),
              ('Libri100 → test-other2938', 'Libri test-other (automatic reference)'),
              ('Libri100 → Buckeye zero-shot1840', 'Libri-trained → Buckeye zero-shot'),
              ('WavLM supplementary → core-test192', 'WavLM / TIMIT (different batch sizes)')]
    codes = {'CTC': ('CTC', '#777777'), 'OTTC CE': ('OT-CE', '#277da8'),
             'OTTC upstream-JSD': ('OT-J', '#248543'), 'OTTC standard-JSD': ('OT-SJ', '#8e6bbd'),
             'CIF CE+Quantity': ('CIF', '#d79015'), 'CIF CE+Quantity+CTC': ('CIF-C', '#c84b31'),
             'CIF CE+Quantity+OTTC': ('CIF-J', '#c15d9a'),
             'WavLM OTTC CE (batch128)': ('OT-CE', '#277da8'),
             'WavLM OTTC upstream-JSD (batch64)': ('OT-J', '#248543')}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (group, title) in zip(axes.flat, groups):
        subset = [r for r in audit['rows'] if r['group'] == group]
        for i, r in enumerate(subset):
            label, color = codes[r['model']]
            ax.scatter(r['PER'], r['R_value'], color=color, s=62, zorder=3)
            offset = (6, (i % 3 - 1) * 12 + 5)
            if group.startswith('TIMIT'):
                offset = {'OT-CE': (12, -7), 'OT-J': (7, 17), 'CIF-C': (7, -15)}.get(label, offset)
            if 'test-clean' in group:
                offset = {'CIF': (6, 8), 'CIF-C': (6, -14)}.get(label, offset)
            ax.annotate(label, (r['PER'], r['R_value']), xytext=offset,
                        textcoords='offset points', fontsize=8, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Phone error rate (%) ↓')
        ax.set_ylabel('Text-free R-value (%) ↑')
        ax.grid(alpha=.2)
        ax.margins(x=.25, y=.2)
    fig.suptitle('Recognition vs segmentation: existing single-seed recipes\n'
                 'Upper left is better. Shared scoring within each panel; no fitted offsets.', fontsize=14)
    fig.savefig(OUT / 'recognition_segmentation.png', dpi=180)
    fig.savefig(OUT / 'recognition_segmentation.pdf')
    plt.close(fig)
    print('Plots saved.', flush=True)

    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    identity_path = OUT / 'wandb_run.json'
    saved = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    run_id = saved.get('id') or wandb.util.generate_id()
    identity = dict(id=run_id, project=defaults['project'], entity=defaults['entity'],
                    origin='backfill-derived-comparison-summary', status='initializing')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    kwargs = dict(id=run_id, project=defaults['project'], entity=defaults['entity'],
                  name='CTC vs native OTTC vs CIF | verified comparison 20261017',
                  group='all-models-comparison-20261017', job_type='evaluation-summary',
                  dir=str(OUT), tags=['backfill', 'derived-summary', 'no-new-inference'],
                  config=dict(logging_origin='backfill-derived-summary',
                              original_training_runs_unchanged=True,
                              imported_training_curves=False, missing_points_interpolated=False,
                              evaluation_protocol='fixed20ms, one-to-one matching; IF-MDD R-value',
                              supplementary_protocol='scratch Zapformer retains its own native clock/reference',
                              selected_by='original per-recipe selection; never test-selected',
                              source_sha256=audit['sources_sha256']),
                  settings=wandb.Settings(init_timeout=20, disable_code=True, x_disable_stats=True))
    mode = defaults.get('mode', 'online')
    try:
        run = init_evaluation(mode=mode, resume='allow', **kwargs)
    except wandb.errors.Error as exc:
        print('Online initialization unavailable:', type(exc).__name__, '; retaining offline logs.', flush=True)
        mode = 'offline'
        run = init_evaluation(mode='offline', **kwargs)
    columns = ['group', 'model', 'PER', 'precision', 'recall', 'F1', 'R_value', 'source']
    run.log({'comparison/text_free': wandb.Table(columns=columns,
             data=[[r[c] for c in columns] for r in audit['rows']]),
             'comparison/recognition_vs_segmentation': wandb.Image(str(OUT / 'recognition_segmentation.png'))})
    run.summary.update(dict(logging_origin='backfill-derived-summary',
                            model_split_rows=len(audit['rows']),
                            checkpoint_comparison_rows=len(audit['checkpoint_average_rows']),
                            training_time_axis_imported=False))
    artifact = wandb.Artifact('verified-ctc-ottc-cif-comparison', type='evaluation-summary')
    for name in ['REPORT_ZH.md', 'metrics.json', 'metrics.csv',
                 'recognition_segmentation.png', 'recognition_segmentation.pdf']:
        artifact.add_file(str(OUT / name))
    run.log_artifact(artifact)
    identity.update(mode=mode, url=run.url if mode == 'online' else None,
                    status='finished-online' if mode == 'online' else 'retained-offline')
    run.finish()
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
