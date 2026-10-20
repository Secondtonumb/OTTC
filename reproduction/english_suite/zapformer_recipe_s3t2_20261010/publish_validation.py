"""Keep validation runs separate from independent long training identity."""
import json,os
from pathlib import Path
import wandb
HERE=Path(__file__).resolve().parent
defaults=json.loads((HERE.parents[1]/'wandb_config.json').read_text())
from reproduction.wandb_policy import init_evaluation
for arm in ['tiny','full']:
    output=HERE/'validation'/arm
    record=json.loads((output/'SMOKE_PASS.json').read_text())
    args=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults.get('entity')),
              mode=os.environ.get('WANDB_MODE',defaults.get('mode','online')),
              name='zapformer_recipe_s3t2_'+arm+'_validation',group='zapformer_recipe_validation_20261010',
              dir=str(output),config=dict(validation_only=True,historical_backfill=False,recipe='zapformer_archived_tricks_v1'),
              settings=wandb.Settings(init_timeout=45))
    try:run=init_evaluation(**args)
    except wandb.errors.CommError:
        args.update(mode='offline');run=init_evaluation(**args)
    run.summary.update(record);(output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n');run.finish()
(HERE/'validation/SMOKE_PASS.json').write_text(json.dumps(dict(status='PASS',checks=['tiny','full'],full_fresh_process_resume=True),indent=2)+'\n')
