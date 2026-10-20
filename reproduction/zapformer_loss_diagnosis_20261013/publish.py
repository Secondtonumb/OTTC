"""Publish derived diagnostic charts into the finished diagnostic run only."""
import json
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent


def main():
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults=json.loads((HERE.parent/'wandb_config.json').read_text())
    identity=json.loads((HERE/'wandb_run.json').read_text())
    opts=dict(project=defaults['project'],entity=defaults['entity'],id=identity['id'],resume='allow',
              dir=str(HERE),mode=identity['mode'])
    try:run=init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode']='offline';opts.pop('resume',None);run=init_evaluation(**opts)
    report=json.loads((HERE/'log_report.json').read_text())
    run.config.update(dict(derived_metric_backfill=True,interpolation=False,
                           historical_x_axis='Original recorded optimizer step or epoch; upload time is not training time'),allow_val_change=True)
    payload={'diagnosis/raw_vs_normalized':wandb.Image(str(HERE/'loss_diagnosis.png'))}
    for prefix,arms in [('jsd',['jsd_only','jsd_fs']),('ce',['ce_only','ce_fs'])]:
        xs=[];ys=[]
        for arm in arms:
            x=report['arms'][arm]
            records=[r for r in x['epoch_summaries'] if r['complete_epoch']]
            xs.append([r['epoch'] for r in records]);ys.append([r['ottc_per_utterance'] for r in records])
            table=wandb.Table(columns=['original_optimizer_step','original_epoch','ottc_per_utterance','source_run'],
                             data=[[r['step'],r['epoch'],r['ottc_per_utterance'],arm] for r in x['development_loss']])
            payload['diagnosis/'+arm+'_fixed_dev']=wandb.plot.line(table,'original_optimizer_step','ottc_per_utterance',title=arm+' fixed TIMIT dev400 OTTC/utterance (derived backfill)')
        payload['diagnosis/'+prefix+'_epoch_ottc']=wandb.plot.line_series(xs=xs,ys=ys,keys=arms,
           title=prefix.upper()+' completed epoch OTTC/utterance (derived backfill)',xname='original_epoch')
    run.log(payload)
    artifact=wandb.Artifact('zapformer-loss-diagnosis-evidence',type='diagnostic')
    for f in ['log_report.json','gpu_report.json','overfit_report.json','forward_sum_invariance.json',
              'loss_diagnosis.png','analyze_logs.py','invariance.py','overfit.py','REPORT.md']:
        artifact.add_file(str(HERE/f))
    run.log_artifact(artifact)
    run.summary['diagnosis/formal_runs_modified']=False
    run.summary['diagnosis/sp_toy_converges']=True
    run.summary['diagnosis/aux_scale_alone_resolves_overfit']=False
    run.finish()


if __name__=='__main__':main()
