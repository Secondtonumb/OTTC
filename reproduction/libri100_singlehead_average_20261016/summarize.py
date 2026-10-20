#!/usr/bin/env python3
"""Report only finished evaluations; retain incomplete averaging status."""
import fcntl
import json
from pathlib import Path
from pfer import HERE, SOURCE, sha, write, paired

NAMES={'ce':'CE','jsd_upstream':'官方 JSD','jsd_standard':'standard JSD'}


def common_fa(a,b,split):
    cache=HERE/'common_fa'/f'{sha(Path(a)/"metrics.json")[:16]}_{sha(Path(b)/"metrics.json")[:16]}_{split}.json'
    if cache.exists():return json.loads(cache.read_text())
    def read(d):
        out={}
        with (Path(d)/f'{split}_predictions.jsonl').open() as f:
            for line in f:
                r=json.loads(line);out[r['utt']]=(r['reference'],r['forced_k2'].get('predicted_spans'))
        return out
    aa,bb=read(a),read(b);assert set(aa)==set(bb)
    n=utts=0;err_a=err_b=0.
    for k in aa:
        ref,pa=aa[k];ref_b,pb=bb[k];assert ref==ref_b
        if not ref or pa is None or pb is None:continue
        assert len(ref)==len(pa)==len(pb)
        utts+=1;n+=len(ref)
        for (rp,rs,re),(ap,ast,ae),(bp,bst,be) in zip(ref,pa,pb):
            assert rp==ap==bp
            err_a+=abs(rs-ast)+abs(re-ae);err_b+=abs(rs-bst)+abs(re-be)
    result=dict(utterances=utts,aligned_tokens=n,a_boundary_MAE_ms=500*err_a/n if n else None,
        b_boundary_MAE_ms=500*err_b/n if n else None,protocol='Common successful exact-phone K2 FA subset only; full-set F1/R retained separately')
    write(cache,result);return result


def main():
    lock=(HERE/'.summary.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    entries={};rows=[];completed=[];selections={}
    for o in NAMES:
        pf=HERE/'pfer'/o/'last';ev=SOURCE/'evaluation'/o
        if (pf/'PFER_COMPLETE.json').exists():entries[(o,'last')]=(ev,pf)
        base=HERE/'results'/o
        if (base/'selection.json').exists():selections[o]=json.loads((base/'selection.json').read_text())
        status=base/'COMPLETE.json' if (base/'COMPLETE.json').exists() else base/'STATUS.json'
        if status.exists():
            state=json.loads(status.read_text())
            for policy,artifact in state['artifacts'].items():
                ev,pf=Path(artifact['evaluation']),Path(artifact['pfer'])
                if (ev/'EVAL_COMPLETE.json').exists() and (pf/'PFER_COMPLETE.json').exists():entries[(o,policy)]=(ev,pf)
        if (base/'COMPLETE.json').exists():completed.append(o)
    for (o,label),(ev,pf) in entries.items():
        mm=json.loads((ev/'metrics.json').read_text());pp=json.loads((pf/'metrics.json').read_text())
        for split,modes in mm['splits'].items():
            m=modes['forced_k2'];p=pp['splits'][split];b=m['boundaries']['20']
            rows.append(dict(objective=o,label=label,split=split,utterances=m['utterances'],
                PER_percent=p['nosil_PER_percent'],PFER_percent=p['pfer_percent'],
                FA_F1_20ms=b['f1_percent'],FA_R_20ms=b['r_value_percent'],FA_boundary_MAE_ms=m.get('boundary_MAE_ms'),
                FA_failures=m['alignment_failures'],evaluation=str(ev),pfer=str(pf)))
    comparisons=[]
    for (o,label),(ev,pf) in entries.items():
        baselines=[]
        if label!='last' and (o,'last') in entries:baselines.append((o,'last'))
        if label in ['dev_fa_best3','dev_per_best3'] and (o,'last3') in entries:baselines.append((o,'last3'))
        if o!='ce' and ('ce',label) in entries:baselines.append(('ce',label))
        for bo,bl in baselines:
            bev,bpf=entries[(bo,bl)]
            for s in json.loads((pf/'metrics.json').read_text())['splits']:
                comparisons.append(dict(objective=o,label=label,baseline_objective=bo,baseline_label=bl,split=s,
                    **paired(pf,bpf,s),common_successful_FA=common_fa(ev,bev,s)))
    result=dict(completed_objectives=completed,all_complete=len(completed)==3,rows=rows,paired_comparisons=comparisons,selections=selections,
        selection_scope='Only five retained checkpoints in epochs39-40; dev clean+other pooled, no test/Buckeye selection',
        selection_metrics='K2 FA F1@20ms or no-sil PER; PFER not used for selection',
        protocol='Singlehead42,20ms,wav2vec2-large-lv60,Libri100; native supplied-phone K2 FA separately from audio-only PER/PFER',
        original_weights_and_results_preserved=True)
    write(HERE/'COMPARE_STATUS.json',result)
    if result['all_complete']:write(HERE/'COMPARE_COMPLETE.json',result)
    lines=['# Libri100：checkpoint 平均、PER / PanPhon PFER / K2 FA','',
        'PFER 是音频自由解码结果的 PanPhon 0.22.2 非加权特征编辑距离 / 参考 IPA 段数；展开双元音，移除 sil/q，dx→d。不使用 supplied-phone FA 的标签计算 PFER。PER 与原报告逐项一致。','',
        'K2 FA 提供参考音素序列，不提供边界时间；F1/R 是全量集合，MAE 仅计成功对齐，直接差异另附共同成功集合。Libri 边界来自已有自动对齐且身份未核验；Buckeye 为原人工 partial19 子集、zero-shot。','',
        '比较 Last、最后3个平均、dev FA最好3个平均、dev PER最好3个平均。dev只用 Libri dev-clean+dev-other、池化计数。可选权重仅为保留的5个（epoch39–40），不代表全程最优；完全相同的选中集合复用权重及推理。','',
        '完成全部平均策略的目标：'+(', '.join(NAMES[o] for o in completed) or '暂无；dev-best 仍在评测。'),
        '各策略完成数：'+', '.join(p+' '+str(sum((o,p) in entries for o in NAMES))+'/3' for p in ['last3','dev_fa_best3','dev_per_best3'])+'。','']
    if selections:
        lines+=['| 目标 | last3 steps | dev FA best3 steps | dev PER best3 steps |',
            '|---|---|---|---|']
        for o,selected in selections.items():
            policies=selected['policies']
            lines.append('| '+NAMES[o]+' | '+' | '.join(', '.join(map(str,policies[p])) for p in ['last3','dev_fa_best3','dev_per_best3'])+' |')
        lines+=['','同一组steps只推理一次；多个选择策略可能复用同一结果。','',
            '| 目标 | 保留的 step | pooled dev PER ↓ | pooled dev K2 FA F1@20 ↑ | dev FA MAE ms ↓ |',
            '|---|---:|---:|---:|---:|']
        for o,selected in selections.items():
            for candidate in selected['candidates']:
                d=candidate['development']
                lines.append(f"| {NAMES[o]} | {candidate['step']} | {d['nosil_PER_percent']:.4f} | {d['fa_f1_20ms']:.4f} | {d['fa_MAE_ms']:.4f} |")
        lines+=['']
    lines+=['| 目标 | 权重 | split | PER ↓ | PFER ↓ | FA F1@20 ↑ | FA R ↑ | FA MAE ms ↓ |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {NAMES[r['objective']]} | {r['label']} | {r['split']} | {r['PER_percent']:.4f} | {r['PFER_percent']:.4f} | {r['FA_F1_20ms']:.4f} | {r['FA_R_20ms']:.4f} | {r['FA_boundary_MAE_ms']:.4f} |")
    lines+=['','PFER差异：负值较好。相对变化和配对置信区间分别列出；1000次utterance bootstrap，未处理同说话人相关性。','',
        '| 目标 / 权重 | 基线 | split | ΔPFER pp ↓ | 相对降低 % ↑ | 配对95% CI pp |',
        '|---|---|---|---:|---:|---|']
    for r in comparisons:
        lo,hi=r['paired_utterance_bootstrap_95ci_pp']
        lines.append(f"| {NAMES[r['objective']]} / {r['label']} | {NAMES[r['baseline_objective']]} / {r['baseline_label']} | {r['split']} | {r['pfer_delta_pp']:+.4f} | {r['relative_pfer_reduction_percent']:+.2f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['','Last结果：官方JSD在Libri test-clean / other 的PFER略高于CE；Buckeye降低0.228pp（相对1.72%），配对CI跨0，目前不能支持“大幅提升”。standard JSD在Buckeye降低0.893pp（相对6.74%），CI不跨0；Libri两个test仍未优于CE。','',
        '所有三个目标的替换错误多数属于特征接近的音素。特征距离、S/I/D和预测/参考IPA段数比保存于 pfer/{objective}/{artifact}/metrics.json，逐句成本保存于 *_per_utterance.jsonl。PFER改善不能仅由它数值小于PER来判断。','']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines))
    print('SUMMARY',dict(completed=completed,rows=len(rows),comparisons=len(comparisons)),flush=True)


if __name__=='__main__':main()
