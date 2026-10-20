"""Rescore saved MFA and K2 outputs with one frozen IF-MDD MFA scorer."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from paper_mfa_report import paper_mfa_notice

SUITE=Path(__file__).resolve().parent
REPRO=SUITE.parent
IFMDD=Path('/home/m64000/work/IF-MDD')
SCRIPTS=IFMDD/'fa_research/scripts'
OUTPUT=SUITE/'mfa_protocol_comparison'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text())


def load_scorer():
    OUTPUT.mkdir(exist_ok=True)
    snapshot=OUTPUT/'scorer_source';snapshot.mkdir(exist_ok=True)
    hashes={}
    for name in ['recompute_timit_raw_phn_tse_rvalue.py','boundary_rvalue.py']:
        source=SCRIPTS/name
        shutil.copy2(source,snapshot/name);hashes[str(source)]=sha(source)
    sys.path.insert(0,str(SCRIPTS))
    spec=importlib.util.spec_from_file_location('original_mfa_timit_scorer',SCRIPTS/'recompute_timit_raw_phn_tse_rvalue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,hashes


def variants():
    records=[]
    for objective in ['jsd_upstream','ce','ctc']:
        base=REPRO/'multihead_timit/evaluation_modes'/objective
        records.append({'objective':objective,'variant':'final_epoch40','files':{s:base/(s+'_predictions.jsonl') for s in ['dev','test']}})
        comparison=REPRO/'checkpoint_comparison'/('timit_'+objective)/'results'
        if not (comparison/'COMPARE_COMPLETE.json').exists():continue
        selection=read(comparison/'selection.json')
        for step in selection['last3']:
            records.append({'objective':objective,'variant':'step_'+str(step)+('_dev_best' if step==selection['best_single'] else ''),
                            'files':{'test':comparison/'individual'/str(step)/'test/test_predictions.jsonl'}})
        records.append({'objective':objective,'variant':'last3_best3_weight_average',
                        'files':{s:comparison/'last3_weight_average/evaluation'/(s+'_predictions.jsonl') for s in ['dev','test']}})
    return records


def export(v, manifests):
    rows=[]
    for split,path in v['files'].items():
        if not path.exists():raise FileNotFoundError(path)
        examples=[json.loads(line) for line in path.open()]
        assert {r['utt'] for r in examples}==set(manifests[split]),str(path)
        for row in examples:
            for head in ['ottc_arpabet','ifmdd_arpabet']:
                mode=row['modes']['forced_k2'][head]
                if 'failure' in mode:raise ValueError(mode)
                spans=mode['spans_seconds']
                assert [s[0] for s in spans]==mode['supplied_phones']
                assert all(math.isfinite(s) and math.isfinite(e) and e>=s for _,s,e in spans)
                rows.append({'model':v['objective']+'/'+v['variant'],'objective':v['objective'],
                             'alignment_head':head,'dataset':'timit_'+split if split=='test' else 'timit_dev',
                             'target':'reference_phone_sequence','utt':row['utt'],'pred_phones':[p for p,_,_ in spans],
                             'pred_starts':[s for _,s,_ in spans],'pred_ends':[e for _,_,e in spans],
                             'fa_backend_used':'IF-MDD K2-FSA','source':str(path),
                             'reference_times_used':mode['reference_times_used']})
    path=OUTPUT/'converted'/v['objective']/(v['variant']+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return rows,{'converted':str(path),'prediction_sha256':{str(p):sha(p) for p in v['files'].values()}}


def evaluate(module, sources, references, common):
    aggregates=defaultdict(module.new_accumulator);per_utterance=[]
    for record in sources:
        dataset=record['dataset'];utt=record['utt']
        predicted=module.prediction_rows(record,dx_map='d',hyp_round_step_ms=0.)
        ref=references[(dataset,utt)]
        metrics,_=module.evaluate_pair(predicted,ref,exclude_length_mismatch=True,
                                      filter_silence_for_metrics=True,ignore_edge_boundaries_for_tse_acc=False)
        base={'family':record.get('objective','mfa'),'dataset':dataset,'model':record['model'],
              'head':record.get('alignment_head',''),'target':record['target'],'utt':utt,
              'has_ref':1,'hyp_len_raw':len(predicted),**metrics}
        per_utterance.append(base)
        group=tuple(base[k] for k in ['family','dataset','model','head','target'])
        module.add_to_accumulator(aggregates[('all',group)],base)
        if utt in common[dataset]:module.add_to_accumulator(aggregates[('common_TSE',group)],base)
    matrix=[{'scope':scope,**module.matrix_row(group,acc)} for (scope,group),acc in sorted(aggregates.items())]
    return matrix,per_utterance


def report(matrix):
    lines=['# OTTC 与归档 CTM：统一评分与系统身份核对','',f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。','']
    lines+=paper_mfa_notice('../')
    lines+=[
           '冻结并直接复用当前 IF-MDD raw-PHN scorer 与原始 PHN reference，同时重算系统身份待核实的归档 CTM 和已完成的 K2-FSA alignment 输出。不训练模型、不重新对齐、不调整时间偏移。','',
           '归档 CTM 历史 test F1=57.22%，当前 scorer 对相同输出重算为57.36%；TSE仍精确复现47.625496ms。当前 boundary_rvalue.py 的候选匹配顺序与历史逐句结果存在差异，旧版本源码未归档，因此保留历史值，比较表统一使用本次冻结 scorer 对所有模型的重计分。','',
           '相同规则：dx→d、closure→silence、过滤 silence、保留相邻重复 phone、非静音 phone 间前一 phone 的 end 作为边界；沿用原双向匹配 PRF/R-value；hyp 不额外舍入。TSE 为每条句子的平均 |start误差|+|end误差|，再对句子取平均；仅等长序列参与。','',
           '## TIMIT core test192：统一边界指标','',
           '| 模型 / checkpoint | head | 条数 | P@20 % | R@20 % | F1@20 % | R-value % | TSE ms | TSE条数 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in matrix:
        if row['scope']!='all' or row['dataset']!='timit_test':continue
        lines.append('| '+' | '.join([row['model'],row['head'] or '归档 ARPA（身份待核实）',str(row['records']),
                     *[f'{row[k]:.2f}' for k in ['boundary_20_precision_pct','boundary_20_recall_pct','boundary_20_f1_pct','boundary_20_r_value_pct','phone_tse_ms']],str(row['eval_utts'])])+' |')
    lines+=['','上表边界全部192条；TSE按每个系统可计分条数展示，不能跨不同子集直接比较。TIMIT仅保留三个 checkpoint，best3与last3为同一集合；dev_best沿用已冻结的 dev PER选择，未按此表的 test成绩重新挑选。','',
            '## 同一152条等长子集：TSE对照','',
            '既有归档 CTM 可计TSE的156条中，原始OTTC head有4条因音素映射/相邻重复融合而和raw PHN数量不等；IF-MDD head在这156条均等长。为让全部head用同一样本，进一步取交集152条；所有系统均在此152条上计TSE，边界主表仍使用完整192条。等长按位置配对，不要求音素标签序列完全相同，因此仍可能包含音素序列与标注差异。原始OTTC目标未重新映射或重训。','',
            '| 模型 / checkpoint | head | 条数 | TSE ms | ACC20 % | ACC50 % |',
            '|---|---|---:|---:|---:|---:|']
    for row in matrix:
        if row['scope']!='common_TSE' or row['dataset']!='timit_test':continue
        lines.append('| '+' | '.join([row['model'],row['head'] or '归档 ARPA（身份待核实）',str(row['eval_utts']),
                     *[f'{row[k]:.2f}' for k in ['phone_tse_ms','phone_acc_20','phone_acc_50']]])+' |')
    lines+=['','归档导出记录的 target 为 word_transcript，原始模型与输入构图来源未核实；OTTC/CTC K2 FA 输入为参考phone序列。这是统一评分的归档诊断表，不是论文 MFA 3.0 的复现或严格同输入比较。归档 CTM 没有 text-free PER；已有 text-free 识别/分段继续单独报告。','',
            '主表为既有 scorer 的closure→silence协议，没有把OTTC的sil预测擅自移动到stop上。所有dev/test、全部容差、label-aware边界、per-utterance与hash见JSON/CSV。L2-ARCTIC的历史MFA参考已找到，但本轮四头尚无相匹配的K2输出，因此未混入这张TIMIT表。','',
            '[完整矩阵](matrix.csv) · [逐句结果](per_utterance.csv) · [评分与输入来源](provenance.json) · [论文 MFA 与归档身份核对](../MFA_TOPLINE.md) · [text-free JSD结果](../JSD_RESULTS.md)。','']
    (OUTPUT/'README.md').write_text('\n'.join(lines))


def publish(matrix):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    urls={}
    for objective in ['jsd_upstream','ce','ctc','mfa']:
        rows=[r for r in matrix if r['family']==objective]
        if not rows:continue
        directory=OUTPUT/'wandb'/objective;directory.mkdir(parents=True,exist_ok=True)
        identity_path=directory/'wandb_run.json';identity=read(identity_path) if identity_path.exists() else {}
        options=dict(project='ottc-english-multiphoneset',entity='kevingenghaopeng-',
                     name=('archived-ctm-identity-unverified' if objective=='mfa' else objective)+'-timit-raw-PHN-rescore',group='mfa_protocol_comparison',job_type='evaluation',
                     id=identity.get('id'),resume='allow' if identity else None,dir=str(directory),
                     mode=os.environ.get('WANDB_MODE','online'),
                     config={'objective':objective,'historical_predictions_rescored':True,'training_updates':0,
                             'archive_model_identity_verified':False,'paper_MFA_3_0_locally_reproduced':False,
                             'scorer':'original IF-MDD MFA rawref_hyp0','test_used_for_checkpoint_selection':False})
        try:run=init_evaluation(**options)
        except wandb.errors.CommError:
            options.update(mode='offline');options.pop('resume',None);run=init_evaluation(**options)
        keys=['scope','dataset','model','head','records','boundary_20_f1_pct','boundary_20_r_value_pct','phone_tse_ms','eval_utts']
        run.log({'mfa_protocol_comparison':wandb.Table(columns=keys,data=[[r[k] for k in keys] for r in rows])})
        artifact=wandb.Artifact(objective+'-mfa-protocol-rescore',type='evaluation')
        for name in ['matrix.csv','per_utterance.csv','provenance.json','README.md']:artifact.add_file(str(OUTPUT/name))
        run.log_artifact(artifact)
        identity_path.write_text(json.dumps({'id':run.id,'url':run.url,'mode':run.settings.mode})+'\n')
        urls[objective]=run.url;run.finish()
    with (OUTPUT/'README.md').open('a') as f:
        f.write('\nW&B：'+' · '.join(f'[{o}]({u})' for o,u in urls.items() if u)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-wandb',action='store_true');args=parser.parse_args()
    module,hashes=load_scorer()
    manifests={s:{r['utt']:r for r in map(json.loads,(REPRO/'multihead_timit/data'/name).open())}
               for s,name in [('dev','train-dev.jsonl'),('test','test.jsonl')]}
    assert len(manifests['dev'])==400 and len(manifests['test'])==192
    mfa_path=IFMDD/'fa_research/results/timit_mfa_ctm_baseline/alignments.jsonl'
    mfa=[json.loads(line) for line in mfa_path.open()]
    assert {(r['dataset'],r['utt']) for r in mfa}=={('timit_'+s,u) for s,records in manifests.items() for u in records}
    index=module.build_phn_index(module.DEFAULT_TIMIT_PHN_ROOT)
    references={};phn_hashes={}
    for split,records in manifests.items():
        for utt in records:
            path=module.phn_path_for_utt(index,utt)
            if path is None:raise ValueError('Missing PHN '+utt)
            phn_hashes[str(path)]=sha(path)
            references[('timit_'+split,utt)]=module.load_raw_phn_ref(index,utt,sample_rate=16000.,dx_map='d',closure_mode='silence')
    common=defaultdict(set)
    for r in mfa:
        predicted=module.prediction_rows(r,dx_map='d',hyp_round_step_ms=0.)
        ref=references[(r['dataset'],r['utt'])]
        if len(module.filter_silence(predicted))==len(module.filter_silence(ref)):common[r['dataset']].add(r['utt'])
    assert len(common['timit_dev'])==364 and len(common['timit_test'])==156
    sources=list(mfa);provenance=[]
    for v in variants():
        converted,origin=export(v,manifests);sources+=converted;provenance.append({'objective':v['objective'],'variant':v['variant'],**origin})
    mfa_eligible={k:set(v) for k,v in common.items()}
    for r in sources:
        predicted=module.prediction_rows(r,dx_map='d',hyp_round_step_ms=0.)
        ref=references[(r['dataset'],r['utt'])]
        if len(module.filter_silence(predicted))!=len(module.filter_silence(ref)):
            common[r['dataset']].discard(r['utt'])
    assert len(common['timit_test'])==152
    matrix,per_utterance=evaluate(module,sources,references,common)
    # Replay TSE/reference counts exactly. Preserve boundary version drift:
    # current scorer changed greedy candidate assignment since the old report.
    old=read(IFMDD/'fa_research/results/timit_mfa_ctm_raw_phn_tse_rvalue/timit_mfa_rawref_hyp0_summary.json')['matrix']
    boundary_drift=[]
    for expected in old:
        actual=next(r for r in matrix if r['scope']=='all' and r['family']=='mfa' and r['dataset']==expected['dataset'])
        for key in ['records','eval_utts','length_mismatch_utts','phone_tse_ms']:
            assert math.isclose(actual[key],expected[key],rel_tol=0,abs_tol=1e-10),(key,actual[key],expected[key])
        boundary_drift.append({'dataset':expected['dataset'],
            'historical_F1_20ms':expected['boundary_20_f1_pct'],'rescored_F1_20ms':actual['boundary_20_f1_pct'],
            'historical_R_value_20ms':expected['boundary_20_r_value_pct'],'rescored_R_value_20ms':actual['boundary_20_r_value_pct']})
    assert not any(r['missing_ref'] or r['empty_pred'] for r in matrix)
    assert all(r['eval_utts']==r['records'] for r in matrix if r['scope']=='common_TSE')
    module.write_csv(OUTPUT/'matrix.csv',matrix);module.write_csv(OUTPUT/'per_utterance.csv',per_utterance)
    (OUTPUT/'provenance.json').write_text(json.dumps({'captured_at':datetime.now().astimezone().isoformat(),
        'scorer_source_sha256':hashes,'mfa_source':str(mfa_path),'mfa_source_sha256':sha(mfa_path),
        'raw_PHN_sha256':phn_hashes,'predictions':provenance,'common_TSE_utterances':{k:sorted(v) for k,v in common.items()},
        'MFA_TSE_eligible_utterances':{k:sorted(v) for k,v in mfa_eligible.items()},
        'common_subset_excluded_for_OTTC_phone_count':{k:sorted(v-common[k]) for k,v in mfa_eligible.items()},
        'historical_mfa_TSE_exact_replay':True,'historical_mfa_boundary_exact_replay':False,
        'mfa_boundary_version_drift':boundary_drift,
        'archive_model_identity_status':'unverified; closure-merge scores match paper Kaldi TDNN-F, not paper MFA 3.0',
        'legacy_mfa_identifiers':'family=mfa, model=TIMIT_MFA and paths are archival labels only; do not establish model identity',
        'paper_MFA_3_0_reproduction':'not completed; cited test TSE 31.95ms / R-value 87.44% is separate from archived CTM scores',
        'paper_identity_audit':'../paper_mfa_audit/audit.json',
        'comparison_rule':'Archived CTM and all CTC/OTTC rows recomputed with identical frozen current IF-MDD scorer; archive identity unverified',
        'gpu_inference_rerun':False,'training_updates':0,
        'protocol':{'dx_map':'d','closure_mode':'silence','filter_silence':True,'hyp_round_step_ms':0,
                    'ignore_edges_for_TSE':False,'exclude_length_mismatch_for_TSE':True,
                    'MFA_input':'Legacy exporter target=word_transcript; original model and graph inputs unverified','OTTC_CTC_input':'reference phone sequence'}},ensure_ascii=False,indent=2)+'\n')
    report(matrix)
    if not args.no_wandb:publish(matrix)
    (OUTPUT/'RESCORE_COMPLETE.json').write_text(json.dumps({'historical_MFA_TSE_replay':'PASS','boundary_version_drift':boundary_drift,'full_dev':400,'full_test':192,'common_test_TSE':len(common['timit_test']),'training_updates':0,'archive_model_identity_status':'unverified; not verified as paper MFA 3.0','verification_scope':'Numeric rescoring only'})+'\n')
    print(json.dumps([r for r in matrix if r['dataset']=='timit_test' and r['head'] in ['','ottc_arpabet'] and ('dev_best' in r['model'] or r['family']=='mfa')],ensure_ascii=False))


if __name__=='__main__':main()
