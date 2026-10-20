"""Recover complete saved splits after an interactive allocation expires."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import evaluate_checkpoint as scoring
from pfer import HERE, SOURCE, sha, write
from worker import evaluate as original_evaluate


def score_saved(path,split):
    manifest=SOURCE/'data'/f'{split}.jsonl'
    reference_rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    raw=Path(path).read_text().splitlines()
    if len(raw)!=len(reference_rows):return None
    try:records=[json.loads(l) for l in raw]
    except json.JSONDecodeError:return None
    counts={mode:scoring.new_counts() for mode in ['text_free','forced_k2']}
    assert len({r['utt'] for r in records})==len(records)
    for row,refrow in zip(records,reference_rows):
        assert all(row[k]==refrow[k] for k in ['utt','wav','duration','reference'])
        assert row['reference_times_used_by_aligner'] is False
        ref=row['reference'];duration=row['duration'];text=row['text_free'];path=np.asarray(text['frame_token_ids'])
        t=counts['text_free'];t['utterances']+=1
        t['refs'].append(' '.join('d' if p=='dx' else p for p,s,e in ref if p not in ('sil','q')))
        t['hyps'].append(' '.join('d' if p=='dx' else p for p in text['hypothesis'] if p not in ('sil','q')))
        changes=np.flatnonzero(path[1:]!=path[:-1])+1
        scoring.score(t,ref,[int(v)*.02 for v in changes],duration)
        t=counts['forced_k2'];t['utterances']+=1;fa=row['forced_k2']
        if 'failure' in fa:
            t['alignment_failures']+=1;scoring.score(t,ref,[],duration);continue
        spans=fa['predicted_spans']
        assert [p for p,s,e in spans]==[p for p,s,e in ref]
        collapsed=[];previous=None
        for token in fa['frame_token_ids']:
            if token!=previous and token!=0:collapsed.append(token)
            previous=token
        assert collapsed==fa['supplied_target_ids']
        scoring.score(t,ref,[v for p,s,e in spans for v in (s,e)],duration)
        err=t['errors']
        for (rp,rs,re),(pp,ps,pe) in zip(ref,spans):
            err['tokens']+=1;err['start']+=abs(rs-ps);err['end']+=abs(re-pe)
            err['start_hit20']+=abs(rs-ps)<=.02+1e-9;err['end_hit20']+=abs(re-pe)<=.02+1e-9
    return {mode:scoring.summary(t) for mode,t in counts.items()}


def evaluate(objective,checkpoint,output,label,splits,smoke=False):
    checkpoint,output=Path(checkpoint),Path(output)
    if smoke or (output/'EVAL_COMPLETE.json').exists():
        return original_evaluate(objective,checkpoint,output,label,splits,smoke)
    available={};derived={}
    directories=[output,*sorted(output.glob('resume_round_*'))]
    plan=json.loads((HERE/'plan.json').read_text())
    for split in splits:
        for directory in directories:
            path=directory/f'{split}_predictions.jsonl'
            if not path.exists():continue
            snapshot=directory/'evaluator_snapshot.py'
            assert snapshot.exists() and sha(snapshot)==plan['adapter_sha256']
            metrics=score_saved(path,split)
            if metrics is not None:available[split]=path;derived[split]=metrics;break
    if not available:return original_evaluate(objective,checkpoint,output,label,splits,smoke)
    missing=[s for s in splits if s not in available]
    print('RESUME_COMPLETE_SPLITS',objective,sorted(available),'pending',missing,flush=True)
    write(output/'RECOVERY_STATUS.json',dict(completed_splits=sorted(available),pending=missing,optimizer_updated=False))
    if missing:
        stage=output/('resume_round_'+'_'.join(missing))
        template=original_evaluate(objective,checkpoint,stage,label+'-resume-'+'_'.join(missing),missing)
        for split in missing:
            available[split]=stage/f'{split}_predictions.jsonl';derived[split]=template['splits'][split]
    elif (output/'metrics.json').exists():template=json.loads((output/'metrics.json').read_text())
    else:
        stage=output/'resume_protocol_smoke'
        template=original_evaluate(objective,checkpoint,stage,label+'-resume-protocol',['dev_clean'],True)
    result=copy.deepcopy(template)
    assert result['checkpoint']==str(checkpoint.resolve())
    assert result['checkpoint_sha256']==sha(checkpoint/'model.safetensors')
    result.update(artifact_label=label,splits={s:derived[s] for s in splits},
        data_manifest_sha256={s:sha(SOURCE/'data'/f'{s}.jsonl') for s in splits},
        recovered_complete_split_predictions=True,recovery_provenance={s:str(p) for s,p in available.items()},
        elapsed_seconds_scope='Last inference invocation; earlier completed split timings not reconstructed')
    for split,path in available.items():
        destination=output/f'{split}_predictions.jsonl'
        if path.resolve()!=destination.resolve():shutil.copy2(path,destination)
    for name in ['ifmdd_k2_source_snapshot.py','evaluator_snapshot.py']:
        if not (output/name).exists():shutil.copy2(stage/name,output/name)
    write(output/'metrics.json',result)
    scoring.publish(result,output,False)
    write(output/'EVAL_COMPLETE.json',dict(objective=objective,artifact_label=label,checkpoint=str(checkpoint.resolve()),smoke=False,
        utterances={s:m['forced_k2']['utterances'] for s,m in result['splits'].items()},
        failures={s:m['forced_k2']['alignment_failures'] for s,m in result['splits'].items()},recovered=True))
    return result
