#!/usr/bin/env python3
"""Independent <=20-minute units, sharing finished inference by weight-set lock."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from pfer import HERE, SOURCE, sha, write, score_evaluation, publish
from average_weights import create
from worker import validate, pooled
from resume_evaluation import evaluate

TASKS=['last3','dev_a','dev_b','dev_fa_best3','dev_per_best3']


@contextmanager
def locked(name):
    directory=HERE/'short_locks';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(name+'.lock')).open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def selection(o,plan):
    base=HERE/'results'/o;file=base/'selection.json';pool=plan['pools'][o]
    with locked(o+'-selection'):
        if file.exists():
            result=json.loads(file.read_text());assert result['plan_sha256']==sha(HERE/'plan.json')
            return result
        candidates=[]
        for candidate in pool['candidates']:
            step=candidate['step']
            output=Path(pool['original_final_evaluation']) if step==17840 else HERE/'candidates'/o/f'step_{step}'
            pf=HERE/'pfer'/o/('last' if step==17840 else f'dev_candidate_{step}')
            assert (output/'EVAL_COMPLETE.json').exists() and (pf/'PFER_COMPLETE.json').exists(), (o,step)
            metrics=json.loads((output/'metrics.json').read_text());pfer=json.loads((pf/'metrics.json').read_text())
            assert metrics['checkpoint']==candidate['checkpoint']
            assert metrics['protocol']['reference_times_used_by_aligner'] is False
            if step!=17840:assert set(metrics['splits'])==set(plan['selection_data'])
            candidates.append(dict(**candidate,development=pooled(metrics,pfer),evaluation=str(output),
                evaluation_metrics_sha256=sha(output/'metrics.json'),model_sha256=metrics['checkpoint_sha256']))
        fa=sorted(candidates,key=lambda c:(-c['development']['fa_f1_20ms'],c['development']['fa_MAE_ms'],c['development']['nosil_PER_percent'],-c['step']))
        per=sorted(candidates,key=lambda c:(c['development']['nosil_PER_percent'],-c['development']['fa_f1_20ms'],c['development']['fa_MAE_ms'],-c['step']))
        result=dict(objective=o,plan_sha256=sha(HERE/'plan.json'),candidate_scope=plan['checkpoint_scope'],candidates=candidates,
            policies={'last3':pool['last3'],'dev_fa_best3':sorted(c['step'] for c in fa[:3]),'dev_per_best3':sorted(c['step'] for c in per[:3])},
            fa_ranking=[c['step'] for c in fa],per_ranking=[c['step'] for c in per],test_and_buckeye_used_for_selection=False,
            pfer_used_for_selection=False,selection_rules=plan['selection_rules'])
        write(file,result)
        return result


def record(o,task,artifact,selected,plan):
    base=HERE/'results'/o
    with locked(o+'-status'):
        file=base/'STATUS.json'
        state=json.loads(file.read_text()) if file.exists() else dict(objective=o,artifacts={})
        state['artifacts'][task]=artifact
        if (base/'selection.json').exists():state['selection']=json.loads((base/'selection.json').read_text())
        else:state['selection']=selected
        state['completed']=set(state['artifacts'])==set(plan['policies'])
        write(file,state)
        if state['completed']:
            assert all((Path(a['evaluation'])/'EVAL_COMPLETE.json').exists() and
                (Path(a['pfer'])/'PFER_COMPLETE.json').exists() for a in state['artifacts'].values())
            unique={Path(a['evaluation']).name:dict(evaluation=a['evaluation'],pfer=a['pfer']) for a in state['artifacts'].values()}
            write(base/'COMPLETE.json',dict(objective=o,selection=state['selection'],artifacts=state['artifacts'],unique_averages=unique,
                last_evaluation=plan['pools'][o]['original_final_evaluation'],last_pfer=str(HERE/'pfer'/o/'last'),
                dispatch_mode='independent 20-minute task units; sum of task runtimes in task completion markers'))
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)


def run(o,task,plan):
    pool=plan['pools'][o]
    with locked(o+'-validation'):validate(o,pool)
    if task in ['dev_a','dev_b']:
        todo=[c for c in pool['candidates'] if c['step']!=17840]
        todo=todo[:2] if task=='dev_a' else todo[2:]
        assert len(todo)==2
        for candidate in todo:
            step=candidate['step'];output=HERE/'candidates'/o/f'step_{step}'
            metrics=evaluate(o,Path(candidate['checkpoint']),output,f'dev-candidate-{step}',plan['selection_data'])
            assert metrics['checkpoint']==candidate['checkpoint']
            score_evaluation(output,HERE/'pfer'/o/f'dev_candidate_{step}',o,f'dev-candidate-{step}')
        return dict(candidate_steps=[c['step'] for c in todo],splits=plan['selection_data'])
    if task=='last3':
        selected=dict(objective=o,plan_sha256=sha(HERE/'plan.json'),policies={'last3':pool['last3']},
            selection='Predeclared latest three retained weights; independent of dev/test results',test_and_buckeye_used_for_selection=False)
    else:selected=selection(o,plan)
    steps=selected['policies'][task];key='_'.join(map(str,sorted(steps)))
    with locked(o+'-average-evaluation-'+key):
        checkpoint=create(o,steps)
        provenance=json.loads((checkpoint/'AVERAGE_COMPLETE.json').read_text())
        if 'candidates' in selected:
            hashes={c['step']:c['model_sha256'] for c in selected['candidates']}
            assert all(c['model_sha256']==hashes[c['step']] for c in provenance['sources'])
        output=HERE/'results'/o/'evaluation'/key;pfer_out=HERE/'pfer'/o/key
        if not (output/'selection.json').exists():write(output/'selection.json',selected)
        write(output/'average_provenance.json',provenance)
        metrics=evaluate(o,checkpoint,output,'avg-'+key,plan['splits'])
        pfer=score_evaluation(output,pfer_out,o,'avg-'+key)
        if not (pfer_out/'selection.json').exists():write(pfer_out/'selection.json',selected)
        write(pfer_out/'average_provenance.json',provenance);publish(pfer,pfer_out)
    artifact=dict(steps=steps,checkpoint=str(checkpoint),evaluation=str(output),pfer=str(pfer_out),checkpoint_sha256=metrics['checkpoint_sha256'])
    record(o,task,artifact,selected,plan)
    return artifact


def main():
    p=argparse.ArgumentParser();p.add_argument('--objective',choices=['ce','jsd_upstream','jsd_standard'],required=True)
    p.add_argument('--task',choices=TASKS,required=True);args=p.parse_args()
    plan=json.loads((HERE/'plan.json').read_text())
    assert sha(HERE/'evaluate_checkpoint.py')==plan['adapter_sha256']
    assert sha(SOURCE/'evaluate.py')==plan['source_evaluator_sha256']
    started=time.monotonic()
    with locked(args.objective+'-task-'+args.task):
        completed=HERE/'short_tasks'/args.objective/(args.task+'_COMPLETE.json')
        if completed.exists():print('TASK_REUSE',args.objective,args.task,flush=True);return
        result=run(args.objective,args.task,plan)
        write(completed,dict(objective=args.objective,task=args.task,plan_sha256=sha(HERE/'plan.json'),
            result=result,elapsed_seconds=time.monotonic()-started))
    print('SHORT_TASK_COMPLETE',args.objective,args.task,flush=True)


if __name__=='__main__':main()
