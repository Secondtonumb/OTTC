"""Log numeric probe summaries only; keep audio, phones and TextGrids local."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
 from reproduction.wandb_policy import init_evaluation
 import wandb
 defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
 result=json.loads((HERE/'RESULTS.json').read_text())
 os.environ.setdefault('WANDB_CACHE_DIR',str(HERE/'wandb_cache'))
 for job in result['jobs']:
  output=HERE/'runs'/job['job_key'];identity=output/'wandb_run.json'
  saved=json.loads(identity.read_text()) if identity.exists() else {}
  kwargs=dict(id=saved.get('id') or wandb.util.generate_id(),entity=defaults['entity'],project=defaults['project'],
   name=job['job_key']+' | CIF/K2 FA selected-example probe',group='cif_fa_probe_20261017',job_type='alignment-evaluation',
   dir=str(output),tags=['fresh-inference','bounded-probe','not-full-benchmark'],
   config=dict(evaluation_only=True,optimizer_updates=0,dataset=job['dataset'],variant=job['variant'],
    scope='one preselected TIMIT or Buckeye utterance; two preselected Libri utterances',
    selected_epoch=job['selected_epoch'],selected_training_step=job['selected_step'],checkpoint_sha256=job['checkpoint_sha256'],
    methods=list(job['methods']),k2_topology='frozen existing IF-MDD CTC graph',
    conditioning='K2: supplied phone sequence; native CIF: supplied phone count only; no reference times',
    boundary_protocol='shared figure reference, unique edges, exclude20ms utterance edges, one-to-one, frozen IF-MDD R formula'),
   settings=wandb.Settings(init_timeout=20,disable_code=True,x_disable_stats=True))
  mode=os.environ.get('WANDB_MODE',defaults.get('mode','online'))
  try:run=init_evaluation(mode=mode,resume='allow',**kwargs)
  except wandb.errors.Error:
   mode='offline';run=init_evaluation(mode=mode,**kwargs)
  numeric={'utterances':job['utterances'],'alignment_failures':job['alignment_failures'],'optimizer_updates':0}
  for method,by_ms in job['methods'].items():
   for ms,metrics in by_ms.items():
    for name in ['precision_pct','recall_pct','f1_pct','r_value_pct']:
     numeric[f'{method}/{name}_{ms}ms']=metrics[name]
  run.log(numeric);run.summary.update(numeric)
  value=dict(id=run.id,entity=defaults['entity'],project=defaults['project'],mode=mode,url=run.url if mode=='online' else None)
  run.finish();identity.write_text(json.dumps(value,indent=2)+'\n')
  print(job['job_key'],value,flush=True)

if __name__=='__main__':main()
