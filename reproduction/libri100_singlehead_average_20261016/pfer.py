#!/usr/bin/env python3
"""Backfill validated PanPhon PFER from audio-only saved phone predictions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import jiwer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'reproduction/libri100_singlehead_20261014'
DISTANCE = Path('/work/gm64/m64000/fa-bench/analysis/panphon_pfer')
TABLE = Path('/work/gm64/m64000/seamless_communication/outputs/panphon_libri100/panphon_table.json')
VALIDATION = Path('/work/gm64/m64000/fa-bench/data/panphon_pfer_20261009/distance_validation.json')
sys.path.insert(0, str(DISTANCE))
from distance import feature_distance, token_distance


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def normalize(phones):
    return ['d' if p == 'dx' else p for p in phones if p not in ('sil', 'q')]


def protocol():
    return dict(metric='PanPhon 0.22.2 unweighted feature_edit_distance(hyp,ref) / reference IPA segment count',
        input='Saved text_free.hypothesis from audio only; no supplied-phone FA labels or reference timestamps',
        phone_normalization='Remove sil and q; dx->d, identical to saved no-sil PER; ao->aa and zh->sh already applied in frozen source protocol',
        diphthongs='Expand to native IPA segment_features from frozen PanPhon table; no averaging of conflicting dimensions',
        empty_reference='Keep hypothesis insertion costs in pooled numerator; no utterance division by zero',
        aggregation='Pool feature costs and reference IPA segments, never average per-utterance percentages',
        interpretation='PFER and PER have different costs and denominators; compare models within each metric',
        substitution_diagnostic='Unit-phone minimum edit alignment from jiwer; feature-close if pair PFER/ref IPA segments <=0.25; heuristic alignment, distinct from optimal PFER path',
        bootstrap='1000 paired utterance draws, seed20261016; not speaker-cluster confidence intervals',
        table=str(TABLE), table_sha256=sha(TABLE), distance_source_sha256=sha(DISTANCE/'distance.py'),
        native_distance_sha256=sha(DISTANCE/'feature_distance.so'), validation=json.loads(VALIDATION.read_text()),
        scoring_source_sha256=sha(__file__), historical_metric_backfill=True, optimizer_updated=False)


def score_evaluation(evaluation, output, objective, label):
    evaluation, output = Path(evaluation), Path(output)
    done = json.loads((evaluation/'EVAL_COMPLETE.json').read_text())
    assert not done['smoke'], 'Full results required'
    metrics = json.loads((evaluation/'metrics.json').read_text())
    if (output/'PFER_COMPLETE.json').exists():
        old=json.loads((output/'metrics.json').read_text())
        assert old['source_metrics_sha256']==sha(evaluation/'metrics.json')
        assert old['protocol']==protocol()
        for split, digest in old['predictions_sha256'].items():
            assert sha(evaluation/f'{split}_predictions.jsonl')==digest
        return old
    output.mkdir(parents=True, exist_ok=True)
    table=json.loads(TABLE.read_text())['phones']
    def vectors(seq):
        missing=set(seq)-set(table)
        if missing: raise ValueError('Unmapped phones: '+str(sorted(missing)))
        vv=[v for p in seq for v in table[p]['segment_features']]
        assert all(len(v)==24 and set(v)<={-1,0,1} for v in vv)
        return vv
    result=dict(objective=objective,label=label,checkpoint=metrics['checkpoint'],
        checkpoint_sha256=metrics['checkpoint_sha256'],source_evaluation=str(evaluation),
        source_metrics_sha256=sha(evaluation/'metrics.json'),protocol=protocol(),splits={},predictions_sha256={})
    substitution_cache={}
    for split, modes in metrics['splits'].items():
        totals=Counter(); confusions=Counter(); rows=[]; seen=set()
        prediction=evaluation/f'{split}_predictions.jsonl'
        with prediction.open() as f:
            for line in f:
                r=json.loads(line)
                assert r['utt'] not in seen; seen.add(r['utt'])
                assert r['reference_times_used_by_aligner'] is False
                ref=normalize([p for p,s,e in r['reference']]); hyp=normalize(r['text_free']['hypothesis'])
                rv,hv=vectors(ref),vectors(hyp)
                cost=feature_distance(hv,rv); edits=token_distance(hyp,ref)
                alignment=(jiwer.process_words(' '.join(ref),' '.join(hyp)) if ref and hyp else
                    SimpleNamespace(substitutions=0,insertions=len(hyp) if not ref else 0,
                        deletions=len(ref) if not hyp else 0,alignments=[[]]))
                assert edits==alignment.substitutions+alignment.insertions+alignment.deletions
                close=0; sub_feature=0.
                for a in alignment.alignments[0]:
                    if a.type!='substitute':continue
                    for rp,hp in zip(ref[a.ref_start_idx:a.ref_end_idx],hyp[a.hyp_start_idx:a.hyp_end_idx]):
                        pair=(rp,hp)
                        if pair not in substitution_cache:
                            substitution_cache[pair]=feature_distance(vectors([hp]),vectors([rp]))/len(vectors([rp]))
                        c=substitution_cache[pair];sub_feature+=c;close+=int(c<=.25+1e-12)
                        confusions[pair]+=1
                entry=dict(utt=r['utt'],pfer_cost=cost,n_ref_segments=len(rv),n_hyp_segments=len(hv),
                    phone_edits=edits,n_ref_phones=len(ref),n_hyp_phones=len(hyp),
                    substitutions=alignment.substitutions,insertions=alignment.insertions,deletions=alignment.deletions,
                    feature_close_substitutions=close,substitution_pair_feature_cost_sum=sub_feature)
                rows.append(entry)
                totals.update({k:v for k,v in entry.items() if k!='utt'})
                totals['empty_reference_utterances']+=int(not rv)
        assert len(rows)==modes['text_free']['utterances']==done['utterances'][split]
        per=100*totals['phone_edits']/totals['n_ref_phones']
        assert abs(per-modes['text_free']['nosil_PER_percent'])<1e-10, (split,per,modes)
        result['splits'][split]=dict(totals,utterances=len(rows),pfer_percent=100*totals['pfer_cost']/totals['n_ref_segments'],
            nosil_PER_percent=per,hyp_reference_segment_ratio=totals['n_hyp_segments']/totals['n_ref_segments'],
            feature_close_substitution_fraction=totals['feature_close_substitutions']/totals['substitutions'] if totals['substitutions'] else None,
            top_substitutions=[dict(reference=p,hypothesis=h,count=n,pair_feature_cost=substitution_cache[(p,h)]) for (p,h),n in confusions.most_common(30)])
        (output/f'{split}_per_utterance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result['predictions_sha256'][split]=sha(prediction)
        print('PFER',objective,label,split,result['splits'][split]['pfer_percent'],'PER',per,flush=True)
    write(output/'metrics.json',result)
    write(output/'PFER_COMPLETE.json',dict(objective=objective,label=label,splits=list(result['splits']),source_metrics_sha256=result['source_metrics_sha256']))
    return result


def paired(a_dir,b_dir,split):
    aa={r['utt']:r for r in map(json.loads,(Path(a_dir)/f'{split}_per_utterance.jsonl').read_text().splitlines())}
    bb={r['utt']:r for r in map(json.loads,(Path(b_dir)/f'{split}_per_utterance.jsonl').read_text().splitlines())}
    keys=sorted(aa); assert set(keys)==set(bb)
    assert all(aa[k]['n_ref_segments']==bb[k]['n_ref_segments'] and aa[k]['n_ref_phones']==bb[k]['n_ref_phones'] for k in keys)
    den=np.array([aa[k]['n_ref_segments'] for k in keys]); num=np.array([aa[k]['pfer_cost']-bb[k]['pfer_cost'] for k in keys])
    dist=[];rng=np.random.default_rng(20261016)
    for _ in range(10):
        draws=rng.integers(0,len(keys),(100,len(keys)))
        dist.extend((100*num[draws].sum(1)/den[draws].sum(1)).tolist())
    baseline=100*sum(bb[k]['pfer_cost'] for k in keys)/den.sum()
    delta=float(100*num.sum()/den.sum())
    return dict(pfer_delta_pp=delta,relative_pfer_reduction_percent=-100*delta/baseline if baseline else None,
        paired_utterance_bootstrap_95ci_pp=np.percentile(dist,[2.5,97.5]).tolist(),utterances=len(keys))


def publish(result,output):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    output=Path(output);identity=output/'wandb_run.json'
    if identity.exists() and json.loads(identity.read_text()).get('finished'):return
    defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    options=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
        name=f"libri100-{result['objective']}-{result['label']}-PFER-backfill",group='libri100_singlehead_average_20261016',
        job_type='evaluation-backfill',tags=['libri100','PFER','text-free','backfill'],dir=str(output),
        mode=os.environ.get('WANDB_MODE','online'),config=result['protocol'],settings=wandb.Settings(init_timeout=45,disable_code=True))
    if identity.exists():
        options['id']=json.loads(identity.read_text())['id']
        if options['mode']=='online':options['resume']='allow'
    try:run=init_evaluation(**options)
    except wandb.errors.CommError:
        options['mode']='offline';options.pop('resume',None);run=init_evaluation(**options)
    values={f'benchmark/{s}/text_free/{k}':v for s,m in result['splits'].items() for k,v in m.items() if isinstance(v,(int,float))}
    run.log(values);run.summary.update(values)
    run.summary.update(dict(checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],
        original_optimizer_step=int(Path(result['checkpoint']).name.split('-')[1]),historical_metric_backfill=True))
    artifact=wandb.Artifact(f"libri100-{result['objective']}-{result['label']}-PFER",type='evaluation')
    for p in output.glob('*.json*'):
        if p.name!='wandb_run.json':artifact.add_file(str(p),name=p.name)
    run.log_artifact(artifact)
    saved=dict(id=run.id,url=run.url,mode=run.settings.mode,finished=False)
    write(identity,saved);run.finish();saved['finished']=True;write(identity,saved)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--objective');p.add_argument('--label',default='last');p.add_argument('--baselines',action='store_true')
    p.add_argument('--publish',action='store_true');args=p.parse_args()
    if args.baselines:
        results={}
        for o in ['jsd_upstream','ce','jsd_standard']:
            out=HERE/'pfer'/o/'last'
            results[o]=score_evaluation(SOURCE/'evaluation'/o,out,o,'last')
            if args.publish:publish(results[o],out)
        comparisons=[]
        for o in ['jsd_upstream','jsd_standard']:
            for s in results[o]['splits']:
                comparisons.append(dict(objective=o,baseline='ce',split=s,**paired(HERE/'pfer'/o/'last',HERE/'pfer/ce/last',s)))
        write(HERE/'PFER_LAST_COMPARE.json',dict(results=results,paired_comparisons=comparisons))
    else:
        assert args.evaluation and args.output and args.objective
        result=score_evaluation(args.evaluation,args.output,args.objective,args.label)
        if args.publish:publish(result,args.output)


if __name__=='__main__':main()
