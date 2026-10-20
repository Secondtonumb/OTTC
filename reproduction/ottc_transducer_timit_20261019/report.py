"""Verify full-set prediction coverage and write a local TIMIT result report."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .common import write_json
from .data import load_rows, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    out=args.output
    complete=json.loads((out/'TRAIN_COMPLETE.json').read_text())
    config=json.loads((out/'run_config.json').read_text())
    assert complete['full_training_complete'] and complete['progress']['epoch']==config['epochs']==40
    assert complete['progress']['step']==2920
    rows,_,data=load_rows()
    assert data['manifests_sha256']==config['manifests_sha256']
    assert data['vocab_sha256']==config['vocab_sha256']
    for name,digest in config['sources_sha256'].items():
        assert sha(Path(name))==digest,name
    manifests=json.loads((out/'checkpoint_manifest.json').read_text())
    assert set(manifests)=={str(i) for i in range(1,41)}
    selection=json.loads((out/'DEV_SELECTION.json').read_text())
    assert not selection['test_used']
    results={}
    for name in ('best_dev','final_5','final_10'):
        results[name]={}
        for split in ('dev','test'):
            path=out/f'benchmark_{name}_{split}.json'
            metrics=json.loads(path.read_text())
            records=[json.loads(line) for line in path.with_suffix('.predictions.jsonl').read_text().splitlines()]
            assert len(records)==metrics['utterances']==len(rows[split])
            assert {r['utt'] for r in records}=={r['utt'] for r in rows[split]}
            assert len({r['utt'] for r in records})==len(records)
            assert metrics['audio_only'] and metrics['blank_free'] and not metrics['forced_alignment']
            assert not metrics['partial'] and not metrics['tiny']
            for record in records:
                assert len(record['hypothesis'])==record['predicted_count']
                assert len(record['text_free_boundaries'])==record['predicted_count']-1
            if split=='test': assert not metrics['test_used_for_selection']
            results[name][split]=metrics
    assert selection['selected']==min(results,key=lambda k:results[k]['dev']['raw_PER_percent'])
    for n in (5,10):
        members=json.loads((out/f'final_{n}_members.json').read_text())['members']
        assert [m['epoch'] for m in members]==list(range(41-n,41))
        for member in members:
            assert member['sha256']==manifests[str(member['epoch'])]['sha256']
            assert member['step']==manifests[str(member['epoch'])]['step']
    epochs=defaultdict(lambda:dict(utterances=0,ottc_sum=0.,length_sum=0.))
    for line in (out/'history.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'train_loss' not in row: continue
        epoch=math.ceil(row['epoch']-1e-9)
        agg=epochs[epoch]
        agg['utterances']+=row['utterances']
        agg['ottc_sum']+=row['train_ottc_sum']
        agg['length_sum']+=row['train_length_sum']
    assert set(epochs)==set(range(1,41))
    for epoch,agg in epochs.items():
        assert agg['utterances']==4620,epoch
        agg['objective_per_utterance']=(agg['ottc_sum']+config['length_weight']*agg['length_sum'])/agg['utterances']
    report=dict(selection=selection,config=config,results=results,epochs=dict(epochs),
        prediction_coverage_verified=True,training_complete=True,evaluation_uploaded=False,
        boundary_protocol='text-free internal predicted-count mass quantiles; duration/T fractional frame clock; frozen IF-MDD directional pooled P/R/F1/R at10/20/30ms')
    write_json(out/'RESULTS.json',report)
    lines=['# TIMIT 无 blank OTTC transducer','',
        '训练完成40轮、2920次更新。train4620 / dev400 / core-test192；同一预训练 Wav2Vec2-LV60、seed42。',
        '原版 upstream-JSD 作用于 Predictor/Joint 的单调运输对；独立 log-count Huber 权重1。该目标不是标准 RNN-T likelihood。',
        '按 dev raw PER 选择模型与平均方案；测试集不参与选择。识别只输入音频，不提供真实音素数，不合并重复标签。',
        '','|模型|划分|PER raw %|PER nosil %|数量 MAE|Precision %|Recall %|F1 %|R-value %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,by_split in results.items():
        for split,metric in by_split.items():
            b=metric['text_free']['20']
            numbers=[metric['raw_PER_percent'],metric['normalized_nosil_PER_percent'],metric['predicted_count_MAE'],
                b['precision_pct'],b['recall_pct'],b['f1_pct'],b['r_value_pct']]
            lines.append('|'+name+'|'+split+'|'+'|'.join(f'{x:.3f}' for x in numbers)+'|')
    lines += ['',f'最终 dev 选择：`{selection["selected"]}`。',
        '分段为音频独立的质量分位点，20ms主表；10/30ms与完整预测见同目录JSON/JSONL，非强制对齐。',
        '现有 TIMIT40 manifest 无相邻重复标签；重复 target 结构能力已由合成测试覆盖，此表不证明真实重复音素识别效果。',
        '训练曲线上传W&B；全部评测指标、预测和本报告仅保存本地。所有40个epoch权重、best-dev、完整resume以及平均成员信息均保留。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'REPORT_COMPLETE.json',dict(full_sets=True,dev=400,test=192,epochs=40,steps=2920,
        selected=selection['selected'],results_sha256=sha(out/'RESULTS.json'),evaluation_uploaded=False))
    print('REPORT_COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
