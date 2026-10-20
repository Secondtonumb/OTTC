"""Publish the requested comparison as saved-result backfill; never train."""
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    from reproduction.wandb_policy import init_evaluation
    import wandb
    for var,folder in [('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:
        os.environ.setdefault(var,str(HERE/folder))
    defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    identity=HERE/'wandb_run.json'
    saved=json.loads(identity.read_text()) if identity.exists() else {}
    if saved.get('finished') and saved.get('mode')=='online':
        print(json.dumps(saved));return
    data=json.loads((HERE/'metrics.json').read_text())
    assert (HERE/'VALIDATION_PASS.json').exists() and len(data['rows'])==23
    options=dict(id=saved.get('id') or wandb.util.generate_id(),
        entity=os.environ.get('WANDB_ENTITY',defaults['entity']),project=os.environ.get('WANDB_PROJECT',defaults['project']),
        name='TIMIT | topology and historical versions | saved-result comparison',group='topology_comparison_figures_20261019',job_type='evaluation-visualization',
        dir=str(HERE),tags=['backfill','evaluation-only','common-reference','no-new-inference'],
        config=dict(backfill=True,new_training=False,new_inference=False,utterances=192,
                    source_training_dates=['2026-10-13 through 2026-10-18'],import_date_is_training_date=False,
                    boundary_protocol=data['boundary_protocol'],selection=data['current_selection'],historical_warning=data['historical_warning']),
        settings=wandb.Settings(init_timeout=20,disable_code=True,x_disable_stats=True))
    mode=os.environ.get('WANDB_MODE',defaults['mode'])
    identity.write_text(json.dumps(dict(id=options['id'],mode=mode,finished=False))+'\n')
    try:run=init_evaluation(mode=mode,resume='allow' if mode=='online' else None,**options)
    except wandb.errors.Error:
        mode='offline';run=init_evaluation(mode=mode,**options)
    columns=['group','model','mode','selected','P20','Recall20','F1_20','R_value20']
    rows=[]
    for r in data['rows']:
        b=r['boundaries']['20'];rows.append([r['group'],r['model'],r['mode'],r.get('selected','historical'),b['precision_pct'],b['recall_pct'],b['f1_pct'],b['r_value_pct']])
    # No interpolated training curve or fake training step is emitted.
    run.summary['saved_result_comparison']=wandb.Table(columns=columns,data=rows)
    run.summary['backfill']=True
    for name in ['current_six_overview','paired_changes','historical_and_current']:
        run.summary[name]=wandb.Image(str(HERE/'figures'/f'{name}.png'))
    # Only the requested aggregate results leave the workspace. Provenance,
    # source code, paths, validation files, utterance outputs and audio stay local.
    record=dict(id=run.id,mode=mode,url=run.url if mode=='online' else None,backfill=True,uploaded='numeric table and three aggregate PNG figures only',finished=True)
    run.finish();identity.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
