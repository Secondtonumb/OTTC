#!/usr/bin/env python3
"""Validate unchanged protocol, dev-select retained weights, average and evaluate."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from pfer import HERE, SOURCE, sha, write, score_evaluation, publish
from average_weights import create


def evaluate(objective,checkpoint,output,label,splits,smoke=False):
    command=[sys.executable,'-u',str(HERE/'evaluate_checkpoint.py'),'--objective',objective,
        '--checkpoint',str(checkpoint),'--output',str(output),'--label',label,'--splits',*splits]
    if smoke:command.append('--smoke')
    subprocess.run(command,check=True)
    return json.loads((output/'metrics.json').read_text())


def validate(objective,pool):
    output=HERE/'validation'/objective
    if (output/'PASS.json').exists():return
    metrics=evaluate(objective,Path(pool['original_final_checkpoint']),output,'adapter-validation',['dev_clean'],True)
    def first8(path):
        with path.open() as f:return [json.loads(next(f)) for _ in range(8)]
    aa=first8(output/'dev_clean_predictions.jsonl')
    bb=first8(Path(pool['original_final_evaluation'])/'dev_clean_predictions.jsonl')
    assert aa==bb, 'Parameterized evaluator changed original paths/spans'
    write(output/'PASS.json',dict(original_first8_prediction_records_exactly_equal=True,
        independence=metrics['label_independence_verified'],source_evaluator_sha256=sha(HERE/'source_evaluator_snapshot.py'),
        adapter_sha256=sha(HERE/'evaluate_checkpoint.py')))


def pooled(metrics,pfer):
    splits=['dev_clean','dev_other'];count={k:sum(metrics['splits'][s]['forced_k2']['boundaries']['20'][k] for s in splits) for k in ['matches','reference','predicted']}
    per=100*sum(pfer['splits'][s]['phone_edits'] for s in splits)/sum(pfer['splits'][s]['n_ref_phones'] for s in splits)
    tokens=sum(metrics['splits'][s]['forced_k2'].get('aligned_tokens',0) for s in splits)
    mae=sum(metrics['splits'][s]['forced_k2'].get('aligned_tokens',0)*metrics['splits'][s]['forced_k2'].get('boundary_MAE_ms',0) for s in splits)/tokens
    return dict(nosil_PER_percent=per,fa_f1_20ms=200*count['matches']/(count['reference']+count['predicted']),fa_MAE_ms=mae,
        boundary_counts=count,aligned_tokens=tokens,splits=splits)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--objective',choices=['ce','jsd_upstream','jsd_standard'],required=True)
    args=parser.parse_args();o=args.objective
    lock=(HERE/f'.{o}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'plan.json').read_text());pool=plan['pools'][o]
    assert sha(HERE/'evaluate_checkpoint.py')==plan['adapter_sha256']
    assert sha(SOURCE/'evaluate.py')==plan['source_evaluator_sha256']
    begin=time.monotonic();base=HERE/'results'/o;base.mkdir(parents=True,exist_ok=True)
    create(o,pool['last3'])
    validate(o,pool)
    original=Path(pool['original_final_evaluation'])
    last_pfer=score_evaluation(original,HERE/'pfer'/o/'last',o,'last');publish(last_pfer,HERE/'pfer'/o/'last')
    selection_file=base/'selection.json'
    if selection_file.exists():
        selection=json.loads(selection_file.read_text());assert selection['plan_sha256']==sha(HERE/'plan.json')
    else:
        candidates=[]
        for c in pool['candidates']:
            step=c['step']
            if step==17840:
                metrics=json.loads((original/'metrics.json').read_text());pfer=last_pfer;output=original
            else:
                output=HERE/'candidates'/o/f'step_{step}'
                metrics=evaluate(o,Path(c['checkpoint']),output,f'dev-candidate-{step}',plan['selection_data'])
                pfer=score_evaluation(output,HERE/'pfer'/o/f'dev_candidate_{step}',o,f'dev-candidate-{step}')
            assert metrics['checkpoint']==c['checkpoint']
            assert metrics['protocol']['reference_times_used_by_aligner'] is False
            candidates.append(dict(**c,development=pooled(metrics,pfer),evaluation=str(output),
                evaluation_metrics_sha256=sha(output/'metrics.json'),model_sha256=metrics['checkpoint_sha256']))
        fa=sorted(candidates,key=lambda c:(-c['development']['fa_f1_20ms'],c['development']['fa_MAE_ms'],c['development']['nosil_PER_percent'],-c['step']))
        per=sorted(candidates,key=lambda c:(c['development']['nosil_PER_percent'],-c['development']['fa_f1_20ms'],c['development']['fa_MAE_ms'],-c['step']))
        selection=dict(objective=o,plan_sha256=sha(HERE/'plan.json'),candidate_scope=plan['checkpoint_scope'],candidates=candidates,
            policies={'last3':pool['last3'],'dev_fa_best3':sorted(c['step'] for c in fa[:3]),'dev_per_best3':sorted(c['step'] for c in per[:3])},
            fa_ranking=[c['step'] for c in fa],per_ranking=[c['step'] for c in per],test_and_buckeye_used_for_selection=False,
            pfer_used_for_selection=False,selection_rules=plan['selection_rules'])
        write(selection_file,selection)
    artifacts={}; evaluated={}
    for policy,steps in selection['policies'].items():
        key='_'.join(map(str,sorted(steps)));checkpoint=create(o,steps)
        output=base/'evaluation'/key;pfer_out=HERE/'pfer'/o/key
        provenance=json.loads((checkpoint/'AVERAGE_COMPLETE.json').read_text())
        selected_hashes={c['step']:c['model_sha256'] for c in selection['candidates']}
        assert all(c['model_sha256']==selected_hashes[c['step']] for c in provenance['sources'])
        write(output/'selection.json',selection);write(output/'average_provenance.json',provenance)
        metrics=evaluate(o,checkpoint,output,'avg-'+key,plan['splits'])
        pfer=score_evaluation(output,pfer_out,o,'avg-'+key)
        write(pfer_out/'selection.json',selection);write(pfer_out/'average_provenance.json',provenance)
        publish(pfer,pfer_out)
        artifacts[policy]=dict(steps=steps,checkpoint=str(checkpoint),evaluation=str(output),pfer=str(pfer_out),
            checkpoint_sha256=metrics['checkpoint_sha256'])
        evaluated[key]=dict(evaluation=str(output),pfer=str(pfer_out))
        write(base/'STATUS.json',dict(objective=o,selection=selection,artifacts=artifacts,completed=False))
        subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)
    write(base/'COMPLETE.json',dict(objective=o,selection=selection,artifacts=artifacts,unique_averages=evaluated,
        last_evaluation=str(original),last_pfer=str(HERE/'pfer'/o/'last'),elapsed_seconds=time.monotonic()-begin))
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)
    print('OBJECTIVE_COMPLETE',o,flush=True)


if __name__=='__main__':main()
