"""Validate full outputs and write an unambiguous text-free versus FA report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    results = {}
    records = {}
    for objective in ('ce', 'jsd_upstream'):
        output = ROOT / 'full' / objective
        complete = json.loads((output / 'EVAL_COMPLETE.json').read_text())
        assert complete['utterances'] == 192 and complete['alignment_failures'] == 0
        result = json.loads((output / 'metrics.json').read_text())
        rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['utt'] for r in rows}) == 192
        for r in rows:
            assert r['text_free_replay_identical'] and not r['reference_times_used_by_aligner']
            assert len(r['frame_token_ids']) == r['frames']
            collapsed = []
            previous = None
            for token in r['frame_token_ids']:
                if token != previous and token != 0:
                    collapsed.append(token)
                previous = token
            assert collapsed == r['supplied_target_ids']
            assert [p for p, _, _ in r['reference']] == [p for p, _, _ in r['predicted_spans']]
            assert r['predicted_spans'][0][1] == 0
            assert abs(r['predicted_spans'][-1][2] - r['duration']) < 1e-9
            assert all(abs(a[2] - b[1]) < 1e-9 for a, b in zip(r['predicted_spans'], r['predicted_spans'][1:]))
        results[objective] = result
        records[objective] = rows
    assert all((a['utt'], a['reference']) == (b['utt'], b['reference'])
               for a, b in zip(records['ce'], records['jsd_upstream']))
    lines = [
        '# 最早 TIMIT 单头 OTTC：K2-FSA forced alignment', '',
        '本次补测原始 CE / upstream JSD 的 epoch 40、step 2920 checkpoint。模型为 wav2vec2-large-lv60，原生输出间隔 20 ms，单头 42 类；没有 S3T2。', '',
        '同一套 TIMIT core-test 192 句。FA 给定历史 manifest 的参考音素顺序（包含 sil），不向模型或对齐器输入参考时间戳；native K2 CTC graph → DenseFsaVec → pruned intersection → shortest_path，没有替代后端。', '',
        '| OTTC 代价 | Text-free F1@20ms | Text-free R-value | K2 FA F1@20ms | K2 FA R-value | FA 边界 MAE |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for objective, label in [('ce', 'CE'), ('jsd_upstream', '原上游 JSD')]:
        result = results[objective]
        old = result['historical_text_free']['boundaries']['20']
        fa = result['forced_k2']['boundaries']['20']
        lines.append(f"| {label} | {100*old['f1']:.2f}% | {100*old['r_value']:.2f}% | "
            f"{fa['f1_percent']:.2f}% | {fa['r_value_percent']:.2f}% | {result['forced_k2']['boundary_MAE_ms']:.2f} ms |")
    ce, js = (results[x]['forced_k2'] for x in ['ce', 'jsd_upstream'])
    lines += ['', f"JSD 相对 CE：FA F1 {js['boundaries']['20']['f1_percent']-ce['boundaries']['20']['f1_percent']:+.2f} 个百分点；"
              f"R-value {js['boundaries']['20']['r_value_percent']-ce['boundaries']['20']['r_value_percent']:+.2f} 个百分点；"
              f"边界 MAE {js['boundary_MAE_ms']-ce['boundary_MAE_ms']:+.2f} ms。", '',
        '| OTTC 代价 | FA P@20ms | FA Recall@20ms | start MAE | end MAE | start+end TSE | 对齐失败 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for objective, label in [('ce', 'CE'), ('jsd_upstream', '原上游 JSD')]:
        m = results[objective]['forced_k2']
        b = m['boundaries']['20']
        lines.append(f"| {label} | {b['precision_percent']:.2f}% | {b['recall_percent']:.2f}% | "
            f"{m['start_MAE_ms']:.2f} ms | {m['end_MAE_ms']:.2f} ms | {m['start_plus_end_TSE_ms']:.2f} ms | 0 / 192 |")
    lines += ['', '评分与解释：', '',
        '- 边界 P/R/F1：所有参考 interval 的唯一端点，去掉距句首/句尾不超过 20 ms 的端点；一对一、仅时间匹配，容差 10/20/30 ms。R-value 使用冻结 IF-MDD 公式，源码哈希保存在 metrics.json。',
        '- FA 时间戳使用 IF-MDD 的 duration / frame_count 网格，blank 归前一个音素；旧 text-free 使用 frame_index × 0.02。未调时间偏移。',
        '- FA 边界 MAE 是全部给定音素的 start/end 绝对误差平均；start+end TSE 是每音素两个误差的和，因此为 MAE 的两倍。没有筛选正确识别的音素。',
        '- 历史 core192 标注已 dx→d、去 q；去 q 留下的参考时间空隙保留不填补，评分仍保留空隙两端。未更改音素标签或移除 sil。',
        '- 全部 192 句的旧预测音素、分割区间和边界已逐句精确回放；CE/JSD 对齐路径均逐句验证折叠后完全等于给定音素顺序。',
        '- FA 的音素序列已知，因此不能用其 PER 表示识别能力。这是 supplied-phone FA，不是词文本 G2P 输入的 FA-Benchmark，也不能直接等同于 raw-PHN MFA topline。', '',
        '结果文件：`full/ce/metrics.json`、`full/jsd_upstream/metrics.json`；逐句路径和区间：对应目录 `predictions.jsonl`。W&B 的独立 evaluation run 在对应 `wandb_run.json`。历史训练及旧评分文件保持原样。', '',
    ]
    (ROOT / 'REPORT_ZH.md').write_text('\n'.join(lines))
    (ROOT / 'COMPARE_COMPLETE.json').write_text(json.dumps(dict(utterances_per_model=192, alignment_failures=0,
        identical_text_free_replay=True, delta_jsd_minus_ce=dict(
        f1_20ms_percentage_points=js['boundaries']['20']['f1_percent']-ce['boundaries']['20']['f1_percent'],
        r_value_20ms_percentage_points=js['boundaries']['20']['r_value_percent']-ce['boundaries']['20']['r_value_percent'],
        boundary_MAE_ms=js['boundary_MAE_ms']-ce['boundary_MAE_ms'])), indent=2) + '\n')
    print('\n'.join(lines[:12]), flush=True)


if __name__ == '__main__':
    main()
