"""Verify full prediction coverage and summarize completed interim evaluations."""
from datetime import datetime
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ('ottc_jsd_0p5', 'ce', 'ctc_0p5')
LABELS = {'ottc_jsd_0p5': 'CE + Quantity + OTTC',
          'ce': 'CE + Quantity', 'ctc_0p5': 'CE + Quantity + CTC'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    report = dict(checked_at=datetime.now().astimezone().isoformat(),
                  evaluation_only=True, interim_training_evaluation=True,
                  selection='best dev text-free R-value@20ms available at evaluation start; no test selection',
                  quantity_weight=1, tolerance_ms=20,
                  boundary_protocol='text-free fractional CIF fires, duration/T clock, endpoints excluded, repeats retained, frozen IF-MDD directional greedy',
                  runs={}, source_sha256={})
    reference_ids = {}
    reference_sequences = {}
    for variant in VARIANTS:
        directory = HERE / 'runs' / variant
        completion = json.loads((directory / 'EVAL_COMPLETE.json').read_text())
        config = json.loads((directory / 'run_config.json').read_text())
        assert completion['full_test_coverage'] and config['quantity_weight'] == 1
        assert completion['checkpoint_sha256'] == config['checkpoint_sha256']
        item = dict(config=config, completion=completion,
                    wandb=json.loads((directory / 'wandb_run.json').read_text()),
                    selected_dev=json.loads((directory / 'selected_dev.json').read_text()), metrics={})
        for name, expected in [('libri_test', 5558), ('buckeye_test', 1840)]:
            path = directory / (name + '.predictions.jsonl')
            rows = [json.loads(line) for line in path.open()]
            ids = {x['utt'] for x in rows}
            references = {x['utt']: (x['reference'], x['reference_boundaries']) for x in rows}
            assert len(rows) == len(ids) == expected
            if name in reference_ids:
                assert ids == reference_ids[name] and references == reference_sequences[name]
            else:
                reference_ids[name], reference_sequences[name] = ids, references
            report['source_sha256'][str(path.relative_to(ROOT))] = digest(path)
        for name, expected in [('libri_test', 5558), ('libri_test_clean', 2620),
                               ('libri_test_other', 2938), ('buckeye_test', 1840)]:
            path = directory / (name + '.json')
            metrics = json.loads(path.read_text())
            assert metrics['utterances'] == expected
            assert metrics['audio_only'] and not metrics['forced_alignment']
            assert metrics['checkpoint_sha256'] == completion['checkpoint_sha256']
            assert metrics['rvalue_source_sha256'] == config['rvalue_source_sha256']
            item['metrics'][name] = metrics
            report['source_sha256'][str(path.relative_to(ROOT))] = digest(path)
        whole = item['metrics']['libri_test']
        clean = item['metrics']['libri_test_clean']
        other = item['metrics']['libri_test_other']
        for key in ('errors', 'phones', 'normalized_errors', 'normalized_phones'):
            assert whole['counts'][key] == clean['counts'][key] + other['counts'][key]
        for mode in ('text_free', 'text_free_frame'):
            for tolerance in ('10', '20', '30'):
                for key in ('precision_matches', 'recall_matches', 'predicted_boundaries', 'reference_boundaries'):
                    assert whole[mode][tolerance][key] == (
                        clean[mode][tolerance][key] + other[mode][tolerance][key])
        report['runs'][variant] = item
    (HERE / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Libri100 CIF 中期 checkpoint 测试结果', '',
             '核验时间：' + report['checked_at'], '',
             '这是训练进行中的独立评测：三组均保留 Quantity（权重 1），CTC / OTTC 辅助权重 0.5；OTTC 使用原始 upstream jsd。各组选取测试开始前 dev 分段 R-value@20ms 最优 checkpoint，测试集未参与选点。现有训练继续至 40 epochs。', '',
             '分段无需参考文本或 phone 数量；fractional duration/T 时钟，20ms 容差，排除句首句尾，保留重复 phone，冻结 IF-MDD directional greedy scorer。Precision、Recall、F1、R-value、PER 的单位均为百分比。', '']
    for title, name in [('Libri 合并 test（5558 条）', 'libri_test'),
                        ('Libri test-clean（2620 条）', 'libri_test_clean'),
                        ('Libri test-other（2938 条）', 'libri_test_other'),
                        ('Buckeye 人工边界 test（1840 条；Libri 训练模型跨语料评测）', 'buckeye_test')]:
        lines += ['## ' + title, '', '| 损失 | 选中 epoch | PER raw | PER nosil | Precision | Recall | F1 | R-value |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant, row in report['runs'].items():
            m = row['metrics'][name]
            s = m['text_free']['20']
            values = [m['raw_PER_percent'], m['normalized_nosil_PER_percent'],
                      *[s[key + '_pct'] for key in ('precision', 'recall', 'f1', 'r_value')]]
            lines.append('| ' + LABELS[variant] + ' | ' + str(row['config']['selected_epoch']) +
                         ' | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
        lines.append('')
    lines += ['三组完整 Libri/Buckeye 预测数量、ID 唯一性、跨组参考 phone 序列及边界一致性均通过核验。10/30ms 和 right-cell grid 指标见各组 JSON。Libri 边界参考为自动对齐、aligner 身份未验证；Buckeye 为人工 phone 边界，两域分别报告。', '',
              '当前实验没有保存逐 epoch 权重，不能据此提供 last-5 / last-10 checkpoint 平均结果。这些是单个 best-dev checkpoint 的结果。', '', '## W&B', '']
    for variant, row in report['runs'].items():
        w = row['wandb']
        lines.append(f'- {LABELS[variant]}: https://wandb.ai/{w["entity"]}/{w["project"]}/runs/{w["id"]}')
    lines += ['', '来源模型、数据、评测器及预测 SHA256，完整指标和选点信息见 `RESULTS.json`。', '']
    (HERE / 'RESULTS_ZH.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
