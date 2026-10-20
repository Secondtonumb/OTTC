"""Real-audio tiny-model validation: exact resume, W&B identity, epoch averages.

All results are smoke checks, not recognition benchmarks or production runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from reproduction.cif_timit_20261016.data import ROOT, sha
from .common import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',WANDB_MODE='offline',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    began = time.monotonic()
    def run(name,cmd):
        with (args.output/(name+'.log')).open('w') as stream:
            result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {args.output/(name+".log")}')
        print(name,'PASS',flush=True)
    run('model_tests',[sys.executable,'-m','unittest','reproduction.ottc_transducer_timit_20261017.test_model','-v'])
    base = [sys.executable,'-u','-m','reproduction.ottc_transducer_timit_20261017.train',
        '--tiny','--epochs','10','--train-limit','4','--eval-limit','1',
        '--batch-size','1','--accumulation-steps','2','--warmup-steps','0',
        '--learning-rate','.001','--save-every','1','--max-wall-seconds','0']
    continuous,resumed = args.output/'continuous',args.output/'resumed'
    run('continuous',base+['--output',str(continuous),'--wandb-mode','disabled'])
    run('pause',base+['--output',str(resumed),'--max-steps','3','--wandb-mode','offline'])
    first_id = json.loads((resumed/'wandb_run.json').read_text())['id']
    paused = torch.load(resumed/'resume.pt',map_location='cpu',weights_only=False)
    assert paused['progress']['step']==3 and paused['progress']['epoch']==1 and paused['progress']['cursor']==1
    del paused
    run('resume',base+['--output',str(resumed),'--wandb-mode','offline'])
    final_id = json.loads((resumed/'wandb_run.json').read_text())['id']
    assert first_id==final_id
    left = torch.load(continuous/'resume.pt',map_location='cpu',weights_only=False)
    right = torch.load(resumed/'resume.pt',map_location='cpu',weights_only=False)
    assert left['progress']==right['progress']
    assert left['contract_id']==right['contract_id']
    assert left['scheduler']==right['scheduler']
    for key in left['model']:
        torch.testing.assert_close(left['model'][key],right['model'][key],rtol=0,atol=0)
    for index,state in left['optimizer']['state'].items():
        for key,value in state.items():
            other = right['optimizer']['state'][index][key]
            if isinstance(value,torch.Tensor): torch.testing.assert_close(value,other,rtol=0,atol=0)
            else: assert value==other
    curves = [[json.loads(x) for x in (directory/'history.jsonl').read_text().splitlines()]
        for directory in (continuous,resumed)]
    losses = [[(r['step'],r['train_loss'],r['learning_rate']) for r in records if 'train_loss' in r] for records in curves]
    assert losses[0]==losses[1] and len(losses[0])==20
    for directory in (continuous,resumed):
        manifest = json.loads((directory/'checkpoint_manifest.json').read_text())
        assert set(manifest)=={str(i) for i in range(1,11)}
        for i,record in manifest.items():
            assert record['sha256']==sha(directory/f'epoch_{int(i):03d}.pt')
            assert record['epoch']==int(i) and record['step']==int(i)*2
        assert (directory/'SMOKE_COMPLETE.json').exists() and not (directory/'TRAIN_COMPLETE.json').exists()
        selection = json.loads((directory/'DEV_SELECTION.json').read_text())
        assert selection['test_used'] is False
        for n in (5,10):
            averaged = torch.load(directory/f'final_{n}_average.pt',map_location='cpu',weights_only=False)
            members = averaged['members']
            assert [m['epoch'] for m in members]==list(range(11-n,11))
            # Independent arithmetic check on a floating classifier tensor.
            key = 'phone_head.output_projection.weight'
            tensors = [torch.load(m['path'],map_location='cpu',weights_only=False)['model'][key] for m in members]
            expected = torch.stack([t.double() for t in tensors]).mean(0).to(tensors[0].dtype)
            torch.testing.assert_close(averaged['model'][key],expected,rtol=0,atol=0)
            assert all(sha(Path(m['path']))==m['sha256'] for m in members)
        for name in ('best_dev','final_5','final_10'):
            for split in ('dev','test'):
                result = json.loads((directory/f'benchmark_{name}_{split}.json').read_text())
                predictions = (directory/f'benchmark_{name}_{split}.predictions.jsonl').read_text().splitlines()
                assert result['partial'] and result['tiny'] and result['audio_only'] and result['blank_free']
                assert not result['forced_alignment'] and result['utterances']==len(predictions)==1
                row = json.loads(predictions[0])
                assert len(row['hypothesis'])==row['predicted_count']
                assert len(row['text_free_boundaries'])==row['predicted_count']-1
    write_json(args.output/'VALIDATION_PASS.json',dict(scope='CPU tiny encoder on real TIMIT audio; not benchmark training',
        exact_resume_model_optimizer_scheduler=True,exact_loss_curve_resume=True,wandb_id=final_id,
        wandb_offline_resume_id_preserved=True,epoch_snapshots=10,final5_final10_arithmetic_verified=True,
        development_only_selection=True,full_training_complete=False,wall_seconds=time.monotonic()-began))
    print('VALIDATION_PASS',args.output,flush=True)


if __name__=='__main__':
    main()
