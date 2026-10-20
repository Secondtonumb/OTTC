"""Publish numeric alpha-decoding summaries only; no audio/text attachments."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
 from reproduction.wandb_policy import init_evaluation
 import wandb
 defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
 results=json.loads((HERE/'RESULTS.json').read_text())
 cases=[json.loads(s) for s in (HERE/'scored_predictions.jsonl').read_text().splitlines()]
 refs=json.loads((HERE/'SCORING_REFERENCES.json').read_text())
 os.environ.setdefault('WANDB_CACHE_DIR',str(HERE/'wandb_cache'))
 identities=[]
 for job in results['jobs']:
  output=HERE/'wandb_runs'/job['dataset'];output.mkdir(parents=True,exist_ok=True)
  identity=output/'wandb_run.json';saved=json.loads(identity.read_text()) if identity.exists() else {}
  kwargs=dict(id=saved.get('id') or wandb.util.generate_id(),entity=defaults['entity'],project=defaults['project'],
   name=job['dataset']+' | upstream JSD audio-only α decoding',group='ottc_alpha_decode_20261017',job_type='segmentation-evaluation',
   dir=str(output),tags=['fresh-inference','audio-only','four-examples-only'],
   config=dict(scope='four preselected visualization examples, not full benchmark',dataset=job['dataset'],
    evaluation_only=True,optimizer_updates=0,objective='original upstream JSD',
    checkpoint_sha256=job['checkpoint_sha256'],original_epoch=job['original_epoch'],original_step=job['original_step'],
    beta='uniform 1/predicted phone count; no reference count or labels',
    methods=['argmax','alpha_grid','alpha_fractional'],clock='same fixed20ms cells; fractional within-cell uniform-mass interpolation',
    reference_used_only_by_scorer=True,recognition_sequence_unchanged=True),
   settings=wandb.Settings(init_timeout=20,disable_code=True,x_disable_stats=True))
  mode=os.environ.get('WANDB_MODE',defaults.get('mode','online'))
  try:run=init_evaluation(mode=mode,resume='allow',**kwargs)
  except wandb.errors.Error:
   mode='offline';run=init_evaluation(mode=mode,**kwargs)
  numeric=dict(utterances=job['utterances'],optimizer_updates=0,recognition_sequence_unchanged=True)
  group=[r for r in cases if r['dataset']==job['dataset']]
  counts=[sum(p not in ('sil','q') for p in refs[r['recording_id']]['reference']['phones']) for r in group]
  numeric['normalized_nosil_PER_percent']=sum(r['normalized_nosil_PER_percent']*n for r,n in zip(group,counts))/sum(counts)
  for method,by_ms in job['methods'].items():
   for ms,metrics in by_ms.items():
    for k in ['precision_pct','recall_pct','f1_pct','r_value_pct']:numeric[f'{method}/{k}_{ms}ms']=metrics[k]
  run.log(numeric);run.summary.update(numeric)
  value=dict(dataset=job['dataset'],id=run.id,entity=defaults['entity'],project=defaults['project'],mode=mode,url=run.url if mode=='online' else None)
  run.finish();identity.write_text(json.dumps(value,indent=2)+'\n');identities.append(value)
 (HERE/'WANDB_RUNS.json').write_text(json.dumps(identities,indent=2)+'\n')
 print(json.dumps(identities),flush=True)

if __name__=='__main__':main()
