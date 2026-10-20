"""Audit all three Buckeye costs; safely report incomplete evaluations."""
import bootstrap
from bootstrap import HERE
import json
from train import model_directory

OBJECTIVES = [('ce','CE'), ('jsd_upstream','原上游 JSD'), ('jsd_standard','standard JSD')]


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def main():
    results={};rows={};experiments={};pending=[]
    for objective,_ in OBJECTIVES:
        output=HERE/'evaluation'/objective
        if not (output/'EVAL_COMPLETE.json').exists():
            pending.append(objective);continue
        complete=json.loads((output/'EVAL_COMPLETE.json').read_text())
        assert complete['utterances']=={'dev':1524,'test':1840}
        assert complete['objective']==objective and not complete['smoke']
        result=json.loads((output/'metrics.json').read_text())
        assert result['objective']==objective and result['label_independence_verified']
        directory=model_directory(objective)
        training=json.loads((directory/'TRAIN_COMPLETE.json').read_text())
        assert training['epoch']==40 and training['global_step']==6840
        assert training['checkpoint']==result['checkpoint']==complete['checkpoint']
        experiments[objective]=json.loads((directory/'experiment.json').read_text())
        results[objective]=result;rows[objective]={}
        for split,n in [('dev',1524),('test',1840)]:
            records=[json.loads(l) for l in (output/(split+'_predictions.jsonl')).read_text().splitlines()]
            assert len(records)==len({r['utt'] for r in records})==n
            rows[objective][split]={r['utt']:r for r in records}
            for r in records:
                assert not r['reference_times_used_by_aligner']
                fa=r['forced_k2']
                if 'failure' in fa:continue
                collapsed=[];previous=None
                for token in fa['frame_token_ids']:
                    if token!=previous and token!=0:collapsed.append(token)
                    previous=token
                assert collapsed==fa['supplied_target_ids']
                assert [p for p,_,_ in fa['predicted_spans']]==[p for p,_,_ in r['reference']]
    available=list(results);common={}
    if available:
        base=available[0]
        recipe=['dataset','initialization','frame_ms','single_head','epochs','batch_size',
                'learning_rate','seed','precision','freeze_cnn','transformer_finetuned',
                'framewise_CE','selection','data']
        # Only the standard adapter is additional; all shared training sources must agree.
        for objective in available[1:]:
            assert all(experiments[base][k]==experiments[objective][k] for k in recipe)
            shared=set(experiments[base]['source_sha256']) & set(experiments[objective]['source_sha256'])
            assert all(experiments[base]['source_sha256'][p]==experiments[objective]['source_sha256'][p] for p in shared)
            assert results[base]['data_manifest_sha256']==results[objective]['data_manifest_sha256']
            assert results[base]['evaluator_sha256']==results[objective]['evaluator_sha256']
        for split in ['dev','test']:
            first=rows[base][split]
            for objective in available[1:]:
                other=rows[objective][split]
                assert set(first)==set(other)
                assert all(first[k]['reference']==other[k]['reference'] and
                           first[k]['duration']==other[k]['duration'] and
                           first[k]['forced_k2']['supplied_target_ids']==other[k]['forced_k2']['supplied_target_ids']
                           for k in first)
            keys=[k for k in first if all('failure' not in rows[o][split][k]['forced_k2'] for o in available)]
            common[split]={'utterances':len(keys),'models':{}}
            for objective in available:
                error=0.;tokens=0
                for k in keys:
                    record=rows[objective][split][k]
                    for (_,rs,re),(_,ps,pe) in zip(record['reference'],record['forced_k2']['predicted_spans']):
                        error+=abs(rs-ps)+abs(re-pe);tokens+=1
                common[split]['models'][objective]={'aligned_tokens':tokens,'boundary_MAE_ms':500*error/tokens if tokens else None}

    lines=['# Buckeye 单头 OTTC：CE / 原上游 JSD / standard JSD','',
        '状态：'+('三组最终评测完成。' if not pending else '比较尚未完成，等待：'+', '.join(pending)+'。下表只列固定epoch40已完成结果。'),'',
        '已有原始人工标注19位说话人子集：train 10,927句（15位，7.33小时）、dev 1,524句（2位，1.11小时）、test 1,840句（2位，0.87小时）。说话人、句子、源录音互斥；不是完整40位Buckeye。','',
        '三组保留同一自监督预训练wav2vec2-large-lv60，随机phone/mass单头。CNN冻结、Transformer微调。20ms、seed42、batch64、AdamW lr2e-4、weight decay0.005、warmup500、40epoch/6840step、regfrozen30epoch，BF16 autocast与FP32主权重。固定最终epoch40评测，不用test选择checkpoint。','',
        '只改变OTTC运输代价：CE；原上游JSD（0.5 KL(m||p)+0.5 KL(m||q)，q按上游加epsilon）；standard JSD（0.5 KL(p||m)+0.5 KL(q||m)）。m=(p+q)/2。standard复用已有jsd_standard实现。CE在OTTC内部，无帧级CE。三种raw loss尺度不同，不直接比较数值高低。','',
        '训练仅输入音频与音素序列，沿用perceived_train_target及边缘sil、合并相邻重复标签，ao→aa。最终评分保留所有人工speech-phone区间，不合并参考时间区间。最终PER来自同一FP32独立评测器；训练日志的BF16诊断PER采用折叠训练目标，不能混用。','',
        'K2 FA输入真实speech-phone顺序，无参考时间戳、无人工edge-sil。native K2 CTC graph、DenseFsaVec、pruned intersection、shortest_path，blank归前一音素，时间网格duration/frames，无偏移。唯一区间端点、一对一时间匹配、去句首/句尾20ms，R-value使用冻结IF-MDD公式。','',
        '| Split | OTTC代价 | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | FA边界MAE ↓ | 对齐失败 |',
        '|---|---|---:|---:|---:|---:|']
    for split in ['dev','test']:
        for objective,label in OBJECTIVES:
            if objective not in results:
                lines.append(f'| {split} | {label} | 待完成 | 待完成 | 待完成 | — |');continue
            m=results[objective]['splits'][split]['forced_k2'];b=m['boundaries']['20']
            mae=common[split]['models'][objective]['boundary_MAE_ms']
            mae_text=f'{mae:.2f} ms' if mae is not None else '无共同成功对齐'
            lines.append(f"| {split} | {label} | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% | {mae_text} | {m['alignment_failures']} / {m['utterances']} |")
    if common:
        lines+=['','MAE使用当前已完成'+str(len(available))+'组共同成功对齐句子：dev '+str(common['dev']['utterances'])+'句、test '+str(common['test']['utterances'])+'句；三组完成后统一用三组交集。F1/R保留完整集合，失败句记零命中。']
    lines+=['','| Split | OTTC代价 | Text-free 去sil PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |',
            '|---|---|---:|---:|---:|']
    for split in ['dev','test']:
        for objective,label in OBJECTIVES:
            if objective not in results:
                lines.append(f'| {split} | {label} | 待完成 | 待完成 | 待完成 |');continue
            m=results[objective]['splits'][split]['text_free'];b=m['boundaries']['20']
            lines.append(f"| {split} | {label} | {m['nosil_PER_percent']:.2f}% | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% |")
    deltas={}
    if 'ce' in results:
        for objective,label in OBJECTIVES[1:]:
            if objective not in results:continue
            ce=results['ce']['splits']['test'];js=results[objective]['splits']['test']
            delta={'text_free_PER_percentage_points':js['text_free']['nosil_PER_percent']-ce['text_free']['nosil_PER_percent']}
            for mode in ['forced_k2','text_free']:
                for metric in ['f1_percent','r_value_percent']:
                    delta[mode+'_'+metric+'_percentage_points']=js[mode]['boundaries']['20'][metric]-ce[mode]['boundaries']['20'][metric]
            deltas[objective+'_minus_ce']=delta
            lines+=['',f"Test {label}相对CE：PER {delta['text_free_PER_percentage_points']:+.2f}个百分点；K2 FA F1 {delta['forced_k2_f1_percent_percentage_points']:+.2f}个百分点，R-value {delta['forced_k2_r_value_percent_percentage_points']:+.2f}个百分点。"]
    if all(o in results for o in ['jsd_upstream','jsd_standard']):
        up=results['jsd_upstream']['splits']['test'];std=results['jsd_standard']['splits']['test']
        delta={'text_free_PER_percentage_points':std['text_free']['nosil_PER_percent']-up['text_free']['nosil_PER_percent']}
        for mode in ['forced_k2','text_free']:
            for metric in ['f1_percent','r_value_percent']:
                delta[mode+'_'+metric+'_percentage_points']=std[mode]['boundaries']['20'][metric]-up[mode]['boundaries']['20'][metric]
        deltas['jsd_standard_minus_jsd_upstream']=delta
        lines+=['',f"Test standard JSD相对原上游JSD：PER {delta['text_free_PER_percentage_points']:+.2f}个百分点；K2 FA F1 {delta['forced_k2_f1_percent_percentage_points']:+.2f}个百分点，R-value {delta['forced_k2_r_value_percent_percentage_points']:+.2f}个百分点。"]
    lines+=['','FA音素标签已知，不报告FA PER。这里是给定音素FA，与参考词/G2P输入的FA-Benchmark分开。此处仅一个seed的同配方比较。','',
            '每组完整指标与逐句路径位于evaluation/{ce,jsd_upstream,jsd_standard}。训练及评测独立记录W&B，身份保存在对应目录的wandb_run.json。','']
    report=HERE/'REPORT_ZH.md';temporary=report.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines));temporary.replace(report)
    status=dict(objectives=[o for o,_ in OBJECTIVES],completed=available,pending=pending,
                common_successful_subset=common,test_deltas=deltas)
    write_json(HERE/'COMPARE_STATUS.json',status)
    if not pending:
        status['data']=json.loads((HERE/'data/audit.json').read_text())
        write_json(HERE/'COMPARE_COMPLETE.json',status)
    print('THREE_WAY_STATUS',json.dumps({'completed':available,'pending':pending}),flush=True)


if __name__=='__main__':main()
