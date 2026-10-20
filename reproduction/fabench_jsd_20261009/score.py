"""Apply unmodified FA-Bench scores to saved JSD and local baseline outputs."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys

from run import BENCH, CELLS, MODEL, OUT, ROOT, digest, write
from fabench.schema import Utterance, Interval
from fabench.normalize import make_canon, manner_of
from fabench.score.core import score_pair
from fabench.score.aggregate import aggregate


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    return value


def score(cell,smoke=False):
    source=CELLS[cell]; output=OUT/cell/('smoke' if smoke else 'full')
    completion=json.loads((output/'INFERENCE_COMPLETE.json').read_text())
    predictions={r['utt_id']:r for r in map(json.loads,(output/'hyp.jsonl').read_text().splitlines())}
    gold={g['utt_id']:g for g in map(json.loads,(source/'gold.jsonl').read_text().splitlines())}
    jobs={j['utt_id']:j for j in map(json.loads,(source/'jobs.jsonl').read_text().splitlines())}
    assert set(gold)==set(jobs) and set(predictions)<=set(gold)
    if not smoke:assert set(predictions)==set(gold)
    provenance=json.loads((output/'provenance.json').read_text())
    assert provenance['signature']==completion['signature']
    assert provenance['signature']['jobs_sha256']==digest(source/'jobs.jsonl')
    systems=defaultdict(dict)
    for key,pred in predictions.items():
        for name,hyp in pred['modes'].items():systems['jsd_'+name][key]=hyp
    baseline_provenance={}
    for name in ('charsiu_10ms','charsiu_20ms'):
        directory=source/name
        if not (directory/'complete.json').exists():continue
        done=json.loads((directory/'complete.json').read_text())
        assert done['n_jobs']==len(gold) and done['failures']==0
        saved={r['utt_id']:r for r in map(json.loads,(directory/'hyp.jsonl').read_text().splitlines())}
        assert set(saved)==set(gold)
        systems[name]={k:saved[k] for k in predictions}
        baseline_provenance[name]=dict(hyp_sha256=digest(directory/'hyp.jsonl'),
                                      provenance=json.loads((directory/'provenance.json').read_text()))
    scores=defaultdict(list)
    for name,hyps in systems.items():
        for key in predictions:
            gd=gold[key];g=Utterance.from_dict(gd);hd=hyps[key]
            h=Utterance(key,'hyp',g.register,g.speaker_id,'',16000,g.duration_s,
                phones=[Interval.from_dict(p) for p in hd['phones']],words=[Interval.from_dict(w) for w in hd['words']])
            s=score_pair(g,h,condition='clean',aligner=name,mode='reference_words',
                gold_canon=make_canon('timit' if cell=='timit' else 'buckeye'),
                hyp_canon=make_canon('arpabet'),manner_of_canonical=manner_of,manner_match=True,
                input_tokens=jobs[key]['text'].split(),n_gold_utts=sum(d['split']==gd['split'] for d in gold.values()))
            scores[gd['split']].append(s)
        print('Scored',cell,name,flush=True)
    rows,types=[],[]
    for split,ss in scores.items():
        rr,tt=aggregate(ss,bootstrap_iters=1000,ta_thresholds_s=[.01,.02,.025,.05,.1],primary_tol_s=.025)
        rows.extend(dict(split=split,**r) for r in rr)
        types.extend(dict(split=split,**r) for r in tt)
    protocol=dict(track=1,condition='clean',input='reference words -> same Charsiu G2P; gold phones/times only for scoring',
        score='unmodified FA-Bench score_pair/aggregate; TIMIT39; manner_match=True; label-only Levenshtein; 1000 utterance bootstrap',
        headline='official bnd_f1_all_20ms and wbnd_f1_all_20ms; class/time matching with utterance edges',
        identity_diagnostic='bnd_f1_lbl_20ms checks aligned identities on interior boundaries; a different denominator',
        phone_mapping='ao->aa, zh->sh, matching LibriSpeech 39-phone supervision; repeated G2P occurrences retained',
        model='Libri100 four-head upstream JSD, final epoch40, no test-selected checkpoint or time shift',
        corpus='TIMIT official dev400/core-test192, original PHN/WRD' if cell=='timit' else
               'Processed Buckeye CTM dev1610/test1505; historical cuts, not the paper segmented human annotation benchmark',
        missing=['four official Kaldi degradation conditions','complete original Buckeye corpus'],smoke=smoke)
    result=finite(dict(protocol=protocol,provenance=provenance,completion=completion,
        baselines=baseline_provenance,gold_sha256=digest(source/'gold.jsonl'),rows=rows,per_boundary_type=types,
        scoring_source_sha256={str(p):digest(p) for p in [BENCH/'fabench/score/core.py',BENCH/'fabench/score/aggregate.py',
            BENCH/'fabench/score/segmentation.py',BENCH/'fabench/normalize/en/config.yaml']}))
    write(output/'metrics.json',result)
    cols=['aligner','split','n_utts','n_absent','mae_ms','mae_ci_lo_ms','mae_ci_hi_ms','bnd_f1_all_20ms',
          'bnd_f1_lbl_20ms','wbe_ms','wbnd_f1_all_20ms','per','wer','coverage']
    with (output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(result['rows'])
    report=['# LibriSpeech 100h OTTC JSD：FA-Bench','',''+protocol['corpus'],'',
        '参考词文本生成目标；两个 ARPAbet 头分别使用 K2 与原生 OT 对齐。checkpoint 固定为最终第40轮。', '',
        '| 系统 | Split | 句数 | 音素 MAE ms | 音素 F1@20ms % | 词 MAE ms | 词 F1@20ms % |','|---|---|---:|---:|---:|---:|---:|']
    def fmt(v,percent=False):return '—' if v is None else f'{v*(100 if percent else 1):.2f}'
    for r in result['rows']:
        report.append('| '+' | '.join([r['aligner'],r['split'],str(r['n_utts']),fmt(r.get('mae_ms')),
            fmt(r.get('bnd_f1_all_20ms'),True),fmt(r.get('wbe_ms')),fmt(r.get('wbnd_f1_all_20ms'),True)])+' |')
    report += ['', 'F1 使用官方评分器。MAE 只统计可匹配边界；identity 检查列、容差扫描、PER 和 WER 保存在 metrics.json。', '',
        '本轮 clean。处理版 Buckeye 与论文切句和参考来源不同；未测四种官方降质，不填 noisy 均值。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    if not smoke:publish(result,output,cell)
    write(output/'EVAL_COMPLETE.json',dict(cell=cell,smoke=smoke,n_jobs=len(predictions),systems=list(systems)))


def publish(result,output,cell):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    training=json.loads((MODEL/'wandb_run.json').read_text())
    project=os.environ.get('WANDB_PROJECT',training.get('project',defaults['project']))
    entity=os.environ.get('WANDB_ENTITY',defaults.get('entity'))
    identity=output/'wandb_run.json'
    previous=json.loads(identity.read_text()) if identity.exists() else {}
    run_id=previous.get('id') or wandb.util.generate_id()
    opts=dict(id=run_id,project=project,entity=entity,name='libri100-jsd-FAbench-'+cell,
        group='libri100_jsd_fabench_20261009',job_type='evaluation',tags=['FA-bench','jsd_upstream',cell,'clean'],
        dir=str(ROOT/'reproduction/results/wandb'),mode=os.environ.get('WANDB_MODE',defaults['mode']),
        config=dict(evaluation_only=True,training_run=training['url'],protocol=result['protocol'],
                    provenance=result['provenance']),settings=wandb.Settings(init_timeout=45,disable_code=True))
    if previous and opts['mode']=='online':opts['resume']='allow'
    try:run=init_evaluation(**opts)
    except wandb.errors.CommError:
        opts.update(mode='offline');opts.pop('resume',None);run=init_evaluation(**opts)
    write(identity,dict(id=run.id,project=project,entity=entity,mode=run.settings.mode,
          url=run.url if run.settings.mode=='online' else None))
    for row in result['rows']:
        for key,value in row.items():
            if isinstance(value,(float,int)) and value is not None:
                run.summary[f"benchmark/{cell}/{row['split']}/{row['aligner']}/{key}"]=value
    artifact=wandb.Artifact('libri100-jsd-FAbench-'+cell,type='evaluation')
    for name in ('metrics.json','metrics.csv','REPORT.md','provenance.json','hyp.jsonl','k2_source_snapshot.py'):
        artifact.add_file(str(output/name),name=name)
    run.log_artifact(artifact);run.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=CELLS,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();score(args.cell,args.smoke)
