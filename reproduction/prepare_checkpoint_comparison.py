"""Freeze current candidate lists; test metrics never participate in selection."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from checkpoint_archive import archive_training

REPRO=Path(__file__).resolve().parent
ROOT=REPRO/'checkpoint_comparison'


def prepare(family,objective,dataset=None):
    name=(dataset if family=='multihead' else 'mfa')+'_'+objective
    target=ROOT/name;target.mkdir(parents=True,exist_ok=True)
    if family=='multihead':
        source=REPRO/('multihead_'+dataset)/'models'/objective
        archive_training(source)
        paths=sorted((target/'snapshots').glob('checkpoint-*'),key=lambda p:int(p.name[11:]))
        candidates=[{'step':int(p.name[11:]),'checkpoint':str(p.resolve())} for p in paths if (p/'trainer_state.json').exists()]
    else:
        source=REPRO/'english_suite/mfa_models'/objective
        candidates=[{'step':int(p.stem[5:]),'checkpoint':str(p.resolve())} for p in sorted(source.glob('step-*.pt'))]
    plan={'family':family,'dataset':dataset or 'mfa','objective':objective,'count':3,
          'created_at':datetime.now().astimezone().isoformat(),'candidates':candidates,
          'output':str(target.resolve()),'experiment':str((REPRO/('multihead_'+dataset)).resolve()) if dataset else None,
          'intermediate_diagnostic':not (source/'TRAIN_COMPLETE.json').exists(),
          'selection_metric':'dev PER, original OTTC ARPAbet head for multihead, native MFA phones for MFA',
          'scope':'all retained checkpoints at plan creation; older deleted weights cannot be ranked'}
    path=target/'plan.json'
    if not path.exists():path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['multihead','mfa'],required=True)
    parser.add_argument('--objective',required=True);parser.add_argument('--dataset');a=parser.parse_args()
    print(prepare(a.family,a.objective,a.dataset),flush=True)


if __name__=='__main__':main()
