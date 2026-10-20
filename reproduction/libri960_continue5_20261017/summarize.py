"""Before/after continuation comparison; select variants using development only."""
import bootstrap
from bootstrap import HERE,ROOT
import json
from pathlib import Path
from archive import atomic_json,digest

OBJECTIVES=['jsd_upstream','ce','jsd_standard']
VARIANTS=['final','best_dev','average_last5']
SPLITS=['dev_clean','dev_other','test_clean','test_other','buckeye_dev','buckeye_test']


def main():
    pending=[];results={};selection={};common={}
    for objective in OBJECTIVES:
        prior=ROOT/'reproduction/libri100_singlehead_20261014/evaluation'/objective
        results[objective]={'libri100_epoch40':json.loads((prior/'metrics.json').read_text())}
        for variant in VARIANTS:
            output=HERE/'evaluation'/objective/variant
            if not (output/'EVAL_COMPLETE.json').exists():
                pending.append(objective+'/'+variant);continue
            current=json.loads((output/'metrics.json').read_text())
            assert current['label_independence_verified']
            assert current['data_manifest_sha256']==results[objective]['libri100_epoch40']['data_manifest_sha256']
            for key in ['boundary_protocol','timestamp_grid','r_value_source_sha256']:
                assert current['protocol'][key]==results[objective]['libri100_epoch40']['protocol'][key]
            results[objective][variant]=current
        if all(v in results[objective] for v in VARIANTS):
            output=HERE/'models'/objective
            assert json.loads((output/'TRAIN_COMPLETE.json').read_text())['epoch']==5
            average=json.loads((output/'AVERAGE_LAST5.json').read_text())
            assert average['member_count']==5 and [m['stage_epoch'] for m in average['members']]==[1,2,3,4,5]
            for member in average['members']:
                assert digest(Path(member['path'])/'model.safetensors')==member['sha256']
            weights={s:sum(len(json.loads(l)['reference']) for l in (HERE/'data'/(s+'.jsonl')).open()) for s in ['dev_clean','dev_other']}
            values={v:sum(results[objective][v]['splits'][s]['text_free']['nosil_PER_percent']*n for s,n in weights.items())/sum(weights.values()) for v in VARIANTS}
            selection[objective]=dict(variant=min(values,key=values.get),criterion='reference-phone-weighted FP32 development-clean/other PER only',
                development_PER_percent=values,test_used_for_selection=False)
            atomic_json(output/'DEV_VARIANT_SELECTION.json',selection[objective])
        common[objective]={}
        for split in SPLITS:
            paths={'libri100_epoch40':prior/(split+'_predictions.jsonl')}
            paths.update({v:HERE/'evaluation'/objective/v/(split+'_predictions.jsonl') for v in VARIANTS if v in results[objective]})
            rows={v:{r['utt']:r for r in (json.loads(l) for l in p.open())} for v,p in paths.items()}
            first=rows['libri100_epoch40']
            for variant,mapping in rows.items():
                assert set(first)==set(mapping)
                assert all(first[k]['reference']==mapping[k]['reference'] and first[k]['duration']==mapping[k]['duration'] for k in first)
            successful=[k for k in first if all('failure' not in mapping[k]['forced_k2'] for mapping in rows.values())]
            common[objective][split]=dict(utterances=len(successful),models={})
            for variant,mapping in rows.items():
                error=0.;tokens=0
                for key in successful:
                    r=mapping[key]
                    for (_,rs,re),(_,ps,pe) in zip(r['reference'],r['forced_k2']['predicted_spans']):
                        error+=abs(rs-ps)+abs(re-pe);tokens+=1
                common[objective][split]['models'][variant]=dict(tokens=tokens,boundary_MAE_ms=500*error/tokens if tokens else None)
    lines=['# Libri100 epoch40 → Libri960 追加5轮：三种OTTC代价','',
        '状态：'+('全部完成。' if not pending else '待完成：'+', '.join(pending)), '',
        '三组分别从自己的Libri100 epoch40/17840step完整权重继承AdamW moments；新的Libri960阶段为281,191句、960.87小时、5epoch/21,970step，累计39,810 optimizer step。BF16、全局batch64、lr2e-5、warmup500、线性调度；保持既有regfrozen对齐分支及CNN冻结，Transformer继续微调。未更改任何OTTC代价公式。','',
        '每epoch在约1/3、2/3和结束保存完整可恢复checkpoint；所有5个完整epoch权重独立保留。比较固定最后一轮、按dev PER选择的best-dev（含原始epoch40基线）以及五个新epoch的平均。仅新增5轮，不能称作last10平均。所有父checkpoint和旧source snapshot保持原样。','',
        '测试从未参与模型/平均成员选择。最终变体按FP32 dev-clean/other参考音素数加权PER选择。Buckeye始终为零样本人工标注评测；Libri边界仅为既有自动对齐参考，来源模型身份未核实。','',
        'Text-free与给定音素K2 FA分开。唯一端点、去句首句尾20ms、一对一10/20/30ms时间匹配；冻结IF-MDD R-value公式、FA duration/frame_count时间网格、blank归前一音素。F1/R覆盖全集，FA失败零命中；MAE为同一目标各变体和加训前的共同成功子集。','',
        '| Objective | Variant | Split | PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |',
        '|---|---|---|---:|---:|---:|']
    for objective in OBJECTIVES:
        for variant,current in results[objective].items():
            for split in SPLITS:
                m=current['splits'][split]['text_free'];b=m['boundaries']['20']
                lines.append(f"| {objective} | {variant} | {split} | {m['nosil_PER_percent']:.2f}% | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% |")
    lines+=['','| Objective | Variant | Split | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | 共同子集边界MAE ↓ |',
            '|---|---|---|---:|---:|---:|']
    for objective in OBJECTIVES:
        for variant,current in results[objective].items():
            for split in SPLITS:
                b=current['splits'][split]['forced_k2']['boundaries']['20']
                mae=common[objective][split]['models'][variant]['boundary_MAE_ms']
                value=f'{mae:.2f} ms' if mae is not None else '无共同子集'
                lines.append(f"| {objective} | {variant} | {split} | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% | {value} |")
    lines+=['','开发集选择：'+json.dumps(selection,ensure_ascii=False),'']
    target=HERE/'REPORT_ZH.md';temporary=target.with_suffix('.md.pending');temporary.write_text('\n'.join(lines));temporary.replace(target)
    status=dict(pending=pending,selection=selection,common_successful_subsets=common)
    atomic_json(HERE/'COMPARE_STATUS.json',status)
    if not pending:atomic_json(HERE/'COMPARE_COMPLETE.json',status)
    print('CONTINUATION_COMPARISON',json.dumps(dict(pending=pending,selection=selection)),flush=True)


if __name__=='__main__':main()
