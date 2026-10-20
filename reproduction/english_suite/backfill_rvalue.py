"""Derive missing R-values from saved P/R without changing original logs.

Historical points retain original epochs/optimizer steps in a separate W&B
backfill run. No weights, source snapshots, trainer states or raw metrics are
overwritten, and no model inference or boundary rescoring is performed.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SUITE=Path(__file__).resolve().parent
REPRO=SUITE.parent
sys.path.insert(0,str(REPRO))
from segmentation_metrics import boundary_r_value,enrich_flat_metrics,SOURCE

OUTPUT=SUITE/'rvalue_backfill'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(value):
    if isinstance(value,list):return [enrich(v) for v in value]
    if not isinstance(value,dict):return value
    result={k:enrich(v) for k,v in value.items()}
    result=enrich_flat_metrics(result)
    if 'precision_percent' in result and 'recall_percent' in result:
        rval=boundary_r_value(result)
        if rval is not None:result.setdefault('r_value_percent',rval)
    return result


def additions(original,enriched,prefix=''):
    rows=[]
    if isinstance(original,dict):
        for k,v in enriched.items():
            key=prefix+'/'+k if prefix else k
            if k not in original:
                if isinstance(v,(int,float)):rows.append({'metric':key,'value':v})
            else:rows+=additions(original[k],v,key)
    elif isinstance(original,list):
        for i,(a,b) in enumerate(zip(original,enriched)):rows+=additions(a,b,prefix+'/'+str(i))
    return rows


def multihead_history(directory):
    points={}
    state=directory/'trainer_state.json'
    if state.exists():
        for r in json.loads(state.read_text()).get('log_history',[]):
            if 'eval_loss' in r:points[(r.get('epoch'),r.get('step'))]={'source':str(state),'record':r}
    log=directory/'train.log'
    if log.exists():
        for line in log.open(errors='replace'):
            if 'eval_boundary_precision_' not in line:continue
            start=line.find('{');end=line.rfind('}')
            if start<0 or end<0:continue
            try:r=ast.literal_eval(line[start:end+1])
            except (ValueError,SyntaxError):continue
            if not isinstance(r,dict) or 'eval_loss' not in r:continue
            # Trainer log lines may omit step; retained trainer_state has it.
            epoch=r.get('epoch')
            old=next((k for k in points if k[0]==epoch),None)
            if old is not None:
                r['step']=points[old]['record'].get('step');points.pop(old)
            points[(epoch,r.get('step'))]={'source':str(log),'record':r}
    return sorted(points.values(),key=lambda x:(x['record'].get('epoch',0),x['record'].get('step') or 0))


def collect():
    OUTPUT.mkdir(exist_ok=True)
    paths=[];curves=[]
    for dataset in ['timit','libri100','libri960']:
        for objective in ['ctc','ce','jsd_upstream']:
            directory=REPRO/f'multihead_{dataset}/models'/objective
            if not directory.exists():continue
            experiment=dataset+'/'+objective
            for p in directory.glob('*_metrics.json'):paths.append(p)
            for point in multihead_history(directory):
                original=point['record'];derived=enrich_flat_metrics(original)
                fields={k:v for k,v in derived.items() if 'boundary_R_value_' in k and k.startswith('eval_')}
                if fields:curves.append({'experiment':experiment,'epoch':original.get('epoch'),
                    'optimizer_step':original.get('step'),'source':point['source'],'R_value_percent':fields})
    paths+=list(REPRO.glob('multihead_*/evaluation_modes/*/metrics.json'))
    paths+=list((SUITE/'mfa_evaluation').glob('*/step-*/metrics.json'))
    paths+=list((REPRO/'checkpoint_comparison').glob('*/results/comparison.json'))
    paths+=list((REPRO/'checkpoint_comparison').glob('*/comparison.json'))
    for directory in (SUITE/'mfa_models').iterdir():
        if not directory.is_dir():continue
        latest=directory/'latest_dev_metrics.json'
        if latest.exists():paths.append(latest)
        history=directory/'dev_metrics.jsonl'
        if not history.exists():continue
        for line in history.open():
            r=json.loads(line);fields={k:v for k,v in enrich_flat_metrics(r['text_free']).items() if 'boundary_R_value_' in k}
            if fields:curves.append({'experiment':'mfa-label-zapformer/'+directory.name,'epoch':r.get('epoch'),
                'optimizer_step':r['optimizer_step'],'source':str(history),'R_value_percent':fields})
    records=[]
    for path in sorted(set(paths)):
        original=json.loads(path.read_text());derived=enrich(original);added=additions(original,derived)
        if not added:continue
        target=OUTPUT/'saved_metrics'/path.relative_to(REPRO);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n')
        records.append({'source':str(path),'source_sha256':sha(path),'enriched_copy':str(target),'derived_fields':added})
    result={'captured_at':datetime.now().astimezone().isoformat(timespec='seconds'),
        'historical_backfill':True,'derivation':'Frozen IF-MDD R-value formula applied to recorded P/R; original boundary protocol unchanged',
        'formula_source':str(SOURCE),'formula_source_sha256':sha(SOURCE),'historical_points_interpolated':False,
        'training_or_inference_rerun':False,'original_metrics_logs_checkpoints_modified':False,
        'saved_metrics':records,'dev_curves':curves}
    (OUTPUT/'backfill.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=[{'experiment':r['experiment'],'epoch':r['epoch'],'optimizer_step':r['optimizer_step'],'metric':k,'R_value_percent':v,'source':r['source']}
          for r in curves for k,v in r['R_value_percent'].items()]
    with (OUTPUT/'dev_rvalue.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['experiment','epoch','optimizer_step','metric','R_value_percent','source']);writer.writeheader();writer.writerows(rows)
    return result


def publish(result):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    identity_path=OUTPUT/'wandb_run.json';identity=json.loads(identity_path.read_text()) if identity_path.exists() else {}
    run=init_evaluation(project='ottc-english-multiphoneset',entity='kevingenghaopeng-',
        name='R-value historical dev-test backfill',job_type='evaluation',group='rvalue_backfill',
        id=identity.get('id'),resume='allow' if identity else None,dir=str(OUTPUT),mode=os.environ.get('WANDB_MODE','online'),
        config={'historical_backfill':True,'no_interpolation':True,'training_updates':0,
            'formula_source_sha256':result['formula_source_sha256'],'boundary_protocol_unchanged':True,
            'paper_MFA_comparison_matched':False})
    identity_path.write_text(json.dumps({'id':run.id,'url':run.url,'historical_backfill':True},indent=2)+'\n')
    columns=['experiment','epoch','original_optimizer_step','metric','R_value_percent','source']
    table=[]
    for record in result['dev_curves']:
        stem='historical/'+record['experiment'];axis=stem+'/original_epoch'
        run.define_metric(stem+'/dev/*',step_metric=axis)
        payload={axis:record['epoch']}
        if record['optimizer_step'] is not None:payload[stem+'/original_optimizer_step']=record['optimizer_step']
        for k,v in record['R_value_percent'].items():
            payload[stem+'/dev/'+k.removeprefix('eval_')]=v
            table.append([record['experiment'],record['epoch'],record['optimizer_step'],k,v,record['source']])
        run.log(payload)
    run.log({'historical_dev_R_value':wandb.Table(columns=columns,data=table)})
    rows=[[r['source'],d['metric'],d['value']] for r in result['saved_metrics'] for d in r['derived_fields']]
    run.log({'saved_test_dev_and_checkpoint_R_value':wandb.Table(columns=['source','metric','R_value_percent'],data=rows)})
    run.summary.update({'historical_dev_records':len(result['dev_curves']),'enriched_metric_files':len(result['saved_metrics'])})
    artifact=wandb.Artifact('english-suite-rvalue-backfill',type='evaluation')
    artifact.add_file(str(OUTPUT/'backfill.json'));artifact.add_file(str(OUTPUT/'dev_rvalue.csv'))
    artifact.add_dir(str(OUTPUT/'saved_metrics'),name='saved_metrics');run.log_artifact(artifact)
    identity_path.write_text(json.dumps({'id':run.id,'url':run.url,'historical_backfill':True},indent=2)+'\n')
    run.finish()
    (OUTPUT/'UPLOAD_COMPLETE.json').write_text(json.dumps({'id':run.id,'url':run.url,
        'dev_records':len(result['dev_curves']),'metric_files':len(result['saved_metrics'])})+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-wandb',action='store_true');args=parser.parse_args()
    result=collect()
    if not args.no_wandb:publish(result)
    print(json.dumps({'dev_records':len(result['dev_curves']),'metric_files':len(result['saved_metrics'])}))


if __name__=='__main__':main()
