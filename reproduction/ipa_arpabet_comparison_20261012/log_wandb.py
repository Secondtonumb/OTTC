"""Offline W&B backfill of inventory rescores, preserving original steps."""
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
os.environ['WANDB_MODE'] = 'offline'  # restricted login-node network
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CACHE_DIR'] = str(OUT / '.wandb_cache')
os.environ['WANDB_CONFIG_DIR'] = str(OUT / '.wandb_config')
os.environ['WANDB_DATA_DIR'] = str(OUT / '.wandb_data')
import wandb

cfg = json.loads((ROOT / 'wandb_config.json').read_text())
report = json.loads((OUT / 'results.json').read_text())
from reproduction.wandb_policy import init_evaluation
run = init_evaluation(project=cfg['project'], entity=cfg['entity'], dir=str(OUT),
    name='TIMIT IPA to ARPAbet saved-prediction rescore 20261012',
    job_type='evaluation-backfill', tags=['backfill', 'ipa-to-arpabet', 'text-free', 'word-input-reference'],
    config={'backfill': True, 'no_training': True, 'protocols': report['protocols'],
            'training_timestamp': 'not replaced by import timestamp'})
columns = ['model', 'epoch', 'step', 'split', 'subset', 'protocol', 'input',
           'utterances', 'reference_phones', 'substitutions_percent', 'deletions_percent',
           'insertions_percent', 'PER_percent']
run.log({'comparison': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in report['results']])})
artifact = wandb.Artifact('timit-ipa-arpabet-rescore-20261012', type='evaluation-backfill',
                         metadata={'backfill': True, 'original_epochs_steps_preserved': True})
for name in ['README.md', 'matrix.csv', 'mapping.json', 'results.json', 'source_manifest.json']:
    artifact.add_file(str(OUT / name), name=name)
run.log_artifact(artifact)
(OUT / 'wandb_receipt.json').write_text(json.dumps({'id': run.id, 'project': cfg['project'],
    'entity': cfg['entity'], 'mode': 'offline', 'offline_directory': str(Path(run.dir).parent),
    'sync_pending': True}, indent=2) + '\n')
run.finish()
print('Offline W&B backfill saved')
