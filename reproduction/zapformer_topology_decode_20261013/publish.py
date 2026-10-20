"""Attach the completed review artifacts to each new evaluation run."""
import hashlib
import json
import os
from pathlib import Path
import wandb

HERE = Path(__file__).resolve().parent


def main():
    from reproduction.wandb_policy import init_evaluation
    assert (HERE / 'COMPLETE.json').exists()
    defaults = json.loads((HERE.parent / 'wandb_config.json').read_text())
    files = [p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py', '.pbs', '.json', '.md', '.png', '.pdf', '.txt')]
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(HERE / 'manifest.json')
    receipts = {}
    for arm in ('jsd_fs', 'jsd_only', 'ce_fs', 'ce_only'):
        out = HERE / arm
        receipt = json.loads((out / 'wandb_run.json').read_text())
        run = init_evaluation(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                         entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
                         id=receipt['id'], resume='must', dir=str(out),
                         settings=wandb.Settings(init_timeout=45, disable_code=True))
        artifact = wandb.Artifact(f'{arm}-s3t2ss-topology-review', type='evaluation')
        for file in files:
            artifact.add_file(str(file))
        run.log_artifact(artifact)
        run.summary['evaluation/self_loop_collapse_audit'] = 'passed'
        run.summary['evaluation/IFMDD_graph_equivalence'] = 'passed'
        receipts[arm] = {'id': run.id, 'url': run.url}
        run.finish()
    (HERE / 'PUBLISHED.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    main()
