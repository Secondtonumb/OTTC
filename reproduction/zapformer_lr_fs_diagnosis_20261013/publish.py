"""Publish finished evidence to the dedicated diagnostic run, never formal histories."""
import json
import os
from pathlib import Path
import wandb

HERE=Path(__file__).resolve().parent


def main():
    from reproduction.wandb_policy import init_evaluation
    assert (HERE/'COMPLETE.json').exists()
    assert (HERE/'REPORT.md').exists()
    defaults=json.loads((HERE.parent/'wandb_config.json').read_text())
    identity=json.loads((HERE/'gradient_probe/wandb_run.json').read_text())
    opts=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),
              entity=os.environ.get('WANDB_ENTITY',defaults['entity']),id=identity['id'],
              resume='must',dir=str(HERE/'gradient_probe'),
              mode=os.environ.get('WANDB_MODE',defaults.get('mode','online')))
    try:run=init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode']='offline';opts.pop('resume',None);run=init_evaluation(**opts)
    diagnostic=json.loads((HERE/'report.json').read_text())
    rows=[]
    for label,case in diagnostic['cases'].items():
        rows.append([label,case['lr'],case['fs_weight'],case['first']['ottc_sum']/4,
                     case['last']['ottc_sum']/4,case['last']['forward_sum_raw'],case['wandb']['id']])
    formal=json.loads((HERE/'formal_snapshot.json').read_text())
    cols=['arm','original_step','original_epoch','mapped_PER','insertion_percent','deletion_percent']
    cols += [scope+'/'+measure for scope in ['text_free','forced_k2']
             for measure in ['boundary_precision_20ms_percent','boundary_recall_20ms_percent',
                             'boundary_F1_20ms_percent','boundary_R_value_20ms_percent']]
    historic=[]
    for label,value in formal['matched_development_metrics'].items():
        mapped=value['mapped_arpabet']
        row=[label,value['optimizer_step'],value['epoch'],mapped['PER_percent'],
             mapped['insertions_percent'],mapped['deletions_percent']]
        row += [value[scope][measure] for scope in ['text_free','forced_k2']
                for measure in ['boundary_precision_20ms_percent','boundary_recall_20ms_percent',
                                'boundary_F1_20ms_percent','boundary_R_value_20ms_percent']]
        historic.append(row)
    run.log({'diagnostic/four_conditions':wandb.Table(columns=['case','lr','fs_weight','initial_jsd_per_utt',
               'final_jsd_per_utt','final_fs_local_mean','run_id'],data=rows),
             'backfill/matched_TIMIT_dev400':wandb.Table(columns=cols,data=historic),
             'backfill/is_backfill':True,'backfill/interpolation':False,
             'diagnostic/lr_fs_plot':wandb.Image(str(HERE/'lr_fs_diagnosis.png'))})
    artifact=wandb.Artifact('zapformer-lr-fs-diagnostic-evidence',type='diagnostic',
                            metadata=dict(formal_histories_changed=False,backfill=True,
                                          original_steps_retained=True,interpolation=False))
    for name in ['REPORT.md','report.json','formal_snapshot.json','manifest.json',
                 'diagnose.py','diagnose.pbs','snapshot_formal.py','summarize.py','publish.py','lr_fs_diagnosis.png']:
        artifact.add_file(str(HERE/name))
    run.log_artifact(artifact);run.finish()
    (HERE/'PUBLISHED.json').write_text(json.dumps(dict(run_id=run.id,url=run.url,mode=run.settings.mode))+'\n')


if __name__=='__main__':main()
