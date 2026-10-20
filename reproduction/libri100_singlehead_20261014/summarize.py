"""Audit matched three-cost experiments and separate reference sources."""
import bootstrap
from bootstrap import HERE
import json
import math
from train import model_directory
EVALUATION_SPLITS = ['dev_clean','dev_other','test_clean','test_other','buckeye_dev','buckeye_test']

OBJECTIVES = [('ce','CE'), ('jsd_upstream','原上游 JSD'), ('jsd_standard','standard JSD')]


def write_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def main():
    audit = json.loads((HERE/'data/PREPARE_COMPLETE.json').read_text())
    steps = 40*math.ceil(audit['splits']['train']['utterances']/64)
    expected = {s:sum(1 for _ in (HERE/'data'/(s+'.jsonl')).open()) for s in EVALUATION_SPLITS}
    results={};experiments={};records={};pending=[]
    for objective,_ in OBJECTIVES:
        output = HERE/'evaluation'/objective
        if not (output/'EVAL_COMPLETE.json').exists():
            pending.append(objective);continue
        complete = json.loads((output/'EVAL_COMPLETE.json').read_text())
        result = json.loads((output/'metrics.json').read_text())
        directory = model_directory(objective)
        training = json.loads((directory/'TRAIN_COMPLETE.json').read_text())
        experiment = json.loads((directory/'experiment.json').read_text())
        assert complete['objective']==result['objective']==objective and not complete['smoke']
        assert complete['utterances']==expected and result['label_independence_verified']
        assert training['epoch']==40 and training['global_step']==steps
        assert training['checkpoint']==result['checkpoint']==complete['checkpoint']
        results[objective]=result;experiments[objective]=experiment;records[objective]={}
        for split in EVALUATION_SPLITS:
            with (output/(split+'_predictions.jsonl')).open() as f:
                rows=[json.loads(line) for line in f]
            assert len(rows)==len({r['utt'] for r in rows})==expected[split]
            for row in rows:
                assert not row['reference_times_used_by_aligner']
                fa=row['forced_k2']
                if 'failure' in fa:continue
                collapsed=[];previous=None
                for token in fa['frame_token_ids']:
                    if token!=previous and token!=0:collapsed.append(token)
                    previous=token
                assert collapsed==fa['supplied_target_ids']
                assert [p for p,_,_ in fa['predicted_spans']]==[p for p,_,_ in row['reference']]
            records[objective][split]={r['utt']:r for r in rows}
    common={};available=list(results)
    if available:
        base=available[0]
        recipe=['dataset','initialization','frame_ms','single_head','epochs','batch_size','learning_rate',
                'seed','precision','freeze_cnn','transformer_finetuned','framewise_CE','selection','data','source_sha256']
        for objective in available[1:]:
            assert all(experiments[base][k]==experiments[objective][k] for k in recipe)
            assert results[base]['evaluator_sha256']==results[objective]['evaluator_sha256']
            assert results[base]['data_manifest_sha256']==results[objective]['data_manifest_sha256']
        for split in EVALUATION_SPLITS:
            first=records[base][split]
            for objective in available[1:]:
                other=records[objective][split]
                assert set(first)==set(other)
                assert all(first[k]['reference']==other[k]['reference'] and first[k]['duration']==other[k]['duration'] and
                    first[k]['forced_k2']['supplied_target_ids']==other[k]['forced_k2']['supplied_target_ids'] for k in first)
            keys=[k for k in first if all('failure' not in records[o][split][k]['forced_k2'] for o in available)]
            common[split]=dict(utterances=len(keys),models={})
            for objective in available:
                error=0.;tokens=0
                for k in keys:
                    row=records[objective][split][k]
                    for (_,rs,re),(_,ps,pe) in zip(row['reference'],row['forced_k2']['predicted_spans']):
                        error+=abs(rs-ps)+abs(re-pe);tokens+=1
                common[split]['models'][objective]=dict(aligned_tokens=tokens,boundary_MAE_ms=500*error/tokens if tokens else None)
    lines=['# Libri100 单头 OTTC：CE / 原上游 JSD / standard JSD','',
        '状态：'+('三组最终评测完成。' if not pending else '等待最终评测：'+', '.join(pending)+'。'),'',
        f"train-clean-100 全部 {audit['splits']['train']['utterances']:,}句，{audit['splits']['train']['hours']:.2f}小时；官方说话人隔离。三组单头 wav2vec2-large-lv60 自监督预训练、随机phone/mass头，CNN冻结、Transformer微调；seed42、batch64、lr2e-4、AdamW weight decay0.005、warmup500、BF16/FP32、40epoch/{steps}step、regfrozen30epoch。固定最后epoch40评测。",'',
        '复用 Buckeye 的原 OTTC 实现，只改变现有运输代价分支。原上游JSD为0.5 KL(m||p)+0.5 KL(m||q)，沿用target epsilon；standard为0.5 KL(p||m)+0.5 KL(q||m)。raw loss定义及尺度不同，不直接排名。','',
        'Libri clean/other 参考来自已有自动对齐（模型身份未核实），不能称作人工分割精度。Buckeye dev/test 为同一原始人工标注19位子集上的零样本评测，可直接比较 Buckeye-trained 三组；它不参与Libri训练或checkpoint选择。','',
        '参考speech-phone区间只映射ao→aa、zh→sh，不合并；去sil/q。训练目标复用canonical ottc_arpabet序列（含sil、合并相邻重复），缓存只含音频/标签/长度，无时间戳。','',
        'Text-free与给定音素K2 FA分别报告；FA不使用参考时间戳或sil目标。FP32推理、20ms网格；FA为duration/frame_count、blank归前一音素、无偏移。边界为唯一区间端点、去句首句尾20ms、一对一时间匹配。R-value沿用冻结IF-MDD公式并保存哈希。F1/R使用全集，失败零命中；MAE为完成组共同成功子集，全部完成后为三组交集。','',
        '| Split | OTTC代价 | Text-free PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |',
        '|---|---|---:|---:|---:|']
    for split in EVALUATION_SPLITS:
        for objective,label in OBJECTIVES:
            if objective not in results:continue
            m=results[objective]['splits'][split]['text_free'];b=m['boundaries']['20']
            lines.append(f"| {split} | {label} | {m['nosil_PER_percent']:.2f}% | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% |")
    lines+=['','| Split | OTTC代价 | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | 共同子集边界MAE ↓ | 对齐失败 |',
            '|---|---|---:|---:|---:|---:|']
    for split in EVALUATION_SPLITS:
        for objective,label in OBJECTIVES:
            if objective not in results:continue
            m=results[objective]['splits'][split]['forced_k2'];b=m['boundaries']['20']
            mae=common[split]['models'][objective]['boundary_MAE_ms']
            value=f'{mae:.2f} ms' if mae is not None else '无共同成功子集'
            lines.append(f"| {split} | {label} | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% | {value} | {m['alignment_failures']}/{m['utterances']} |")
    cross_training={}
    historical=HERE.parent/'buckeye_singlehead_20261014/evaluation'
    if available:
        lines+=['','同一人工标注 Buckeye test：训练语料对照（完整集合，未做checkpoint平均）','',
            '| OTTC代价 | 训练语料 | Text-free PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ | K2 FA F1@20ms ↑ | K2 FA R-value ↑ |',
            '|---|---|---:|---:|---:|---:|---:|']
        for objective,label in OBJECTIVES:
            if objective not in results:continue
            prior_output=historical/objective
            if not (prior_output/'EVAL_COMPLETE.json').exists():continue
            prior=json.loads((prior_output/'metrics.json').read_text())
            current=results[objective]
            for key in ['boundary_protocol','timestamp_grid','r_value_source_sha256']:
                assert prior['protocol'][key]==current['protocol'][key]
            with (prior_output/'test_predictions.jsonl').open() as f:
                for line in f:
                    row=json.loads(line);new=records[objective]['buckeye_test'][row['utt']]
                    assert row['reference']==new['reference'] and row['duration']==new['duration']
            cross_training[objective]={'buckeye_trained':prior['splits']['test'],
                'libri100_trained':current['splits']['buckeye_test'],
                'prior_evaluator_sha256':prior['evaluator_sha256']}
            for dataset,m in [('Buckeye',prior['splits']['test']),('Libri100',current['splits']['buckeye_test'])]:
                tf=m['text_free'];tb=tf['boundaries']['20'];fb=m['forced_k2']['boundaries']['20']
                lines.append(f"| {label} | {dataset} | {tf['nosil_PER_percent']:.2f}% | {tb['f1_percent']:.2f}% | {tb['r_value_percent']:.2f}% | {fb['f1_percent']:.2f}% | {fb['r_value_percent']:.2f}% |")
    lines+=['','完整P/R/F1/R、10/20/30ms、逐句路径与来源位于evaluation目录，训练与最终评测均记录W&B；checkpoint平均尚未包含。','']
    tmp=HERE/'REPORT_ZH.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(HERE/'REPORT_ZH.md')
    status=dict(objectives=[o for o,_ in OBJECTIVES],completed=available,pending=pending,
                common_successful_subset=common,data=audit,buckeye_cross_training=cross_training)
    write_json(HERE/'COMPARE_STATUS.json',status)
    if not pending:write_json(HERE/'COMPARE_COMPLETE.json',status)
    print('THREE_WAY_STATUS',json.dumps(dict(completed=available,pending=pending)),flush=True)


if __name__=='__main__':main()
