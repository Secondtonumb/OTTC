"""Audit archived CTMs separately from the paper MFA 3.0 baseline.

Reuse existing scorer functions verbatim via AST, avoiding GPU/model imports.
This evaluates existing CTMs; it does not run or train an MFA acoustic model.
"""
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from paper_mfa_report import paper_mfa_notice

SUITE=Path(__file__).resolve().parent
REPRO=SUITE.parent
IFMDD=Path('/home/m64000/work/IF-MDD')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=IFMDD/'fa_research/results/timit_mfa_ctm_baseline/alignments.jsonl'
    manifest=REPRO/'multihead_timit/data/test.jsonl'
    sources=[REPRO/'train_multihead.py',REPRO/'evaluate_multihead_modes.py']
    wanted={'boundary_matches','accumulator','score_boundaries','summarize'}
    functions=[]
    for p in sources:
        functions += [n for n in ast.parse(p.read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name in wanted]
    namespace={'Counter':Counter}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<existing OTTC boundary scorer>','exec'),namespace)
    records={r['utt']:r for r in map(json.loads,manifest.open())}
    mfa={r['utt']:r for r in map(json.loads,source.open()) if r['dataset']=='timit_test'}
    missing=sorted(set(records)-set(mfa))
    if missing:raise ValueError('Missing MFA utterances: '+str(missing))
    counts=namespace['accumulator']()
    for utt,row in records.items():
        alignment=mfa[utt]
        edges=alignment['pred_starts']+alignment['pred_ends']
        namespace['score_boundaries'](counts,row['references']['ottc_arpabet'],edges,row['duration'])
        counts['supervised']+=1
    current=namespace['summarize'](counts,'archived_ctm_identity_unverified','ottc_arpabet')
    historical=[]
    for folder in ['timit_mfa_ctm_raw_phn_tse_rvalue','timit_mfa_ctm_raw_phn_tse_rvalue_closuremerge']:
        for p in (IFMDD/'fa_research/results'/folder).glob('*matrix.csv'):
            for row in csv.DictReader(p.open()):
                historical.append({'protocol':folder,'source':str(p),'source_sha256':sha(p),
                                   **{k:row[k] for k in ['dataset','records','length_mismatch_utts','eval_utts',
                                                        'phone_tse_ms','boundary_20_f1_pct','boundary_20_r_value_pct']}})
    result={'captured_at':datetime.now().astimezone().isoformat(timespec='seconds'),
            'native_mfa_source':str(source),'source_sha256':sha(source),'manifest':str(manifest),
            'manifest_sha256':sha(manifest),'scorer_sha256':{str(p):sha(p) for p in sources},
            'missing_utterances':missing,'native_mfa_current_boundary_protocol':current,'historical':historical,
            'archive_identity_status':'unverified; closure-merge metrics match paper Kaldi TDNN-F, not paper MFA 3.0',
            'legacy_native_mfa_keys':'historical key names only; do not infer model identity',
            'paper_audit':str(SUITE/'paper_mfa_audit/audit.json'),
            'input_protocol':'Archived exporter says word_transcript; original recipe unverified. OTTC K2 uses reference phones.',
            'text_free_PER':'not measured from supplied-transcript CTMs',
            'training':'none; reused existing archived CTM timestamps'}
    (SUITE/'mfa_topline_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 论文 MFA 3.0 与归档 CTM 身份核对','',f'更新：{result["captured_at"]}。','']
    lines+=paper_mfa_notice()
    lines+=['## 系统身份待核实的既有 TIMIT CTM 数值','',
           '以下路径与 JSON 键名保留历史名称，仅用于追溯，不能凭其中的 mfa/native_mfa 字样认定为论文 MFA 3.0。','',
           '| 协议 | Split | 边界计分条数 | TSE计分条数 | F1@20ms % | R-value % | TSE ms |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in historical:
        lines.append('| '+' | '.join([row['protocol'],row['dataset'],row['records'],row['eval_utts'],
                     f'{float(row["boundary_20_f1_pct"]):.2f}',f'{float(row["boundary_20_r_value_pct"]):.2f}',
                     f'{float(row["phone_tse_ms"]):.2f}'])+' |')
    lines+=['','TIMIT core test192 有36条归档 CTM/GT phone 数量不等，历史 TSE 只在156条等长样本计算；历史 boundary F1/R-value仍使用192条。TSE不等于当前报告的单侧 start/end MAE。','',
            '## 沿用当前 OTTC 边界评分协议重新计分','',
            '直接复用当前 scorer，读取同一 test192 manifest/reference：所有 phone interval 起止边界去重、排除音频两端20ms、按容差做一对一匹配。重用已保存的 CTM，未重跑 aligner，未修正时间偏移。','',
            '| 系统 | 条数 | F1@10ms % | F1@20ms % | F1@30ms % |',
            '|---|---:|---:|---:|---:|',
            '| 归档 CTM（身份待核实） | 192 | '+' | '.join(f'{current["boundaries"][str(ms)]["f1_percent"]:.2f}' for ms in [10,20,30])+' |','',
            '此行与当前 OTTC 使用相同 utterance/reference/边界算法，但归档 CTM 的模型和输入构图来源未核实。不能凭这一行宣称超过论文 MFA，也不能从 transcript-conditioned CTM 推算 text-free PER。','',
            '当前参考含 silence/closure 区间的起止边界（共7123个），历史 no-sil scorer 的参考边界为5562个，因此当前 F1=53.89% 与历史57.22%属于不同评分口径，不能解释为模型性能变化。','',
            'L2-ARCTIC 既有 MFA10ms baseline：test900 的 F1@20ms=79.57%、R-value=82.57%；严格 TSE=29.04ms仅计778条等长样本（另122条不等长）。来源：[原始 L2-ARCTIC MFA报告]('+str(IFMDD/'fa_research/results/l2arctic_mfa_gt_fa_10ms/summary.md')+')。','',
            '[CE/JSD/CTC 与归档 CTM 沿用同一个冻结 scorer 的对照](mfa_protocol_comparison/README.md)：完整 test192 边界与各系统共同等长 TSE 子集。','',
            '[TIMIT原始汇总]('+str(IFMDD/'fa_research/results/paper_prep_experiment_audit/final_paper_tables_zh.md')+') · [核对JSON](mfa_topline_audit.json) · [当前JSD结果](JSD_RESULTS.md)。','']
    (SUITE/'MFA_TOPLINE.md').write_text('\n'.join(lines))
    print(json.dumps({'current_protocol_mfa':current,'historical':historical},ensure_ascii=False))


if __name__=='__main__':main()
