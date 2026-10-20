"""Compile verified model comparisons from saved evaluation artifacts only."""
import ast
from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path

from reproduction.segmentation_metrics import SOURCE, _scorer
from reproduction.cif_comparison_20261016.compare_english import read

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction/all_models_comparison_20261017'


def main():
    OUT.mkdir(exist_ok=True)
    hashes, rows = {}, []

    def track(p):
        p = Path(p)
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def load(p):
        return json.loads(track(p).read_text())

    def row(group, model, per, boundary, source, **extra):
        def value(*keys):
            for k in keys:
                if k in boundary:
                    return boundary[k]
            raise KeyError(keys)
        p = value('precision_pct', 'precision_percent')
        r = value('recall_pct', 'recall_percent')
        rows.append(dict(group=group, model=model, PER=per, precision=p, recall=r,
                         F1=value('f1_pct', 'f1_percent'),
                         R_value=100 * _scorer.r_value(p / 100, r / 100)[0],
                         source=str(source.relative_to(ROOT)), **extra))

    # The earlier CIF re-score has a fully pinned source manifest. Verify it
    # before reusing any metrics, including the individual prediction files.
    timit = load(HERE / 'metrics.json')
    english = load(HERE / 'english_metrics.json')
    for cache in [timit, english]:
        for name, expected in cache['sources_sha256'].items():
            assert hashlib.sha256(track(ROOT / name).read_bytes()).hexdigest() == expected, name
    labels = {'CTC': 'CTC', 'OTTC_CE': 'OTTC CE',
              'OTTC_upstream_JSD': 'OTTC upstream-JSD',
              'CIF_no_ctc': 'CIF CE+Quantity',
              'CIF_ctc_0p5': 'CIF CE+Quantity+CTC',
              'CIF_ottc_jsd_0p5': 'CIF CE+Quantity+OTTC'}
    for name, label in labels.items():
        m = timit['runs'][name]
        row('TIMIT → core-test192', label, m['normalized_nosil_PER_percent'],
            m['boundaries']['fixed20'], HERE / 'metrics.json', epoch=m['selected_epoch'])
    for label, suffix in [('OTTC standard-JSD', 'runexp_ottc_fuse_phoneme_jsd_standard'),
                          ('WavLM OTTC CE (batch128)', 'runexp_ottc_fuse_phoneme_ce_wavlm_b128_20261013'),
                          ('WavLM OTTC upstream-JSD (batch64)', 'runexp_ottc_fuse_phoneme_jsd_wavlm_20261013')]:
        directory = ROOT / 'reproduction/results' / suffix
        assert (directory / 'TRAIN_COMPLETE.json').exists() and (directory / 'EVAL_COMPLETE.json').exists()
        source = directory / 'metrics.json'
        result = load(source)
        m = result['splits']['timit_core_test_192']
        b = m['boundaries']['20']
        boundary = {f'{k}_pct': b[k] * 100 for k in ['precision', 'recall', 'f1']}
        group = 'TIMIT → core-test192' if suffix.endswith('standard') else 'WavLM supplementary → core-test192'
        row(group, label, 100 * m['normalized_nosil_PER'], boundary, source,
            epoch=result['epoch'], step=result['global_step'])

    groups = {('buckeye', 'test'): 'Buckeye → test1840',
              ('libri100', 'test_clean'): 'Libri100 → test-clean2620',
              ('libri100', 'test_other'): 'Libri100 → test-other2938',
              ('libri100', 'buckeye_test'): 'Libri100 → Buckeye zero-shot1840'}
    cif_labels = {'ce': 'CIF CE+Quantity', 'ctc_0p5': 'CIF CE+Quantity+CTC',
                  'ottc_jsd_0p5': 'CIF CE+Quantity+OTTC'}
    for m in english['comparisons']:
        if m['variant'] != 'native_ottc':
            row(groups[m['training'], m['split']], cif_labels[m['variant']],
                m['PER_nosil'], m['boundary'], HERE / 'english_metrics.json')

    # Native CE/JSD branches share the reference manifests. Replay their
    # boundary counts against the exact native evaluator function.
    source = track(ROOT / 'reproduction/buckeye_singlehead_20261014/train.py')
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boundary_matches')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    for dataset in ['buckeye', 'libri100']:
        base = ROOT / f'reproduction/{dataset}_singlehead_20261014/evaluation'
        for objective in ['ce', 'jsd_upstream', 'jsd_standard']:
            source = base / objective / 'metrics.json'
            result = load(source)
            assert (base / objective / 'EVAL_COMPLETE.json').exists()
            for (training, split), group in groups.items():
                if training != dataset:
                    continue
                predictions = read(track(base / objective / f'{split}_predictions.jsonl'))
                reference = read(base / 'jsd_upstream' / f'{split}_predictions.jsonl')
                assert predictions.keys() == reference.keys()
                counts = Counter()
                for key, prediction in predictions.items():
                    assert prediction['reference'] == reference[key]['reference']
                    duration = prediction['duration']
                    rb = sorted({float(t) for _, s, e in prediction['reference']
                                 for t in (s, e) if .02 < t < duration - .02})
                    frames = prediction['text_free']['frame_token_ids']
                    pb = [i * .02 for i in range(1, len(frames))
                          if frames[i] != frames[i - 1] and .02 < i * .02 < duration - .02]
                    counts.update(matches=namespace['boundary_matches'](rb, pb, .02),
                                  reference=len(rb), predicted=len(pb))
                m = result['splits'][split]['text_free']
                assert all(counts[k] == m['boundaries']['20'][k] for k in counts)
                label = {'ce': 'OTTC CE', 'jsd_upstream': 'OTTC upstream-JSD',
                         'jsd_standard': 'OTTC standard-JSD'}[objective]
                row(group, label, m['nosil_PER_percent'], m['boundaries']['20'], source,
                    epoch=40, prediction_count=len(predictions), exact_boundary_count_replay=True)

    # Preserve the scratch Zapformer protocol separately: it is a different
    # encoder, clock and boundary reference, not a matched CIF loss ablation.
    zap = ROOT / 'reproduction/zapformer_timit_20261013/runs/jsd_seed42'
    complete = load(zap / 'TRAIN_COMPLETE.json')
    assert complete['epoch'] == 40
    m = load(zap / 'final_test.json')
    row('Scratch Zapformer native protocol → core-test192', 'Zapformer upstream-JSD (10ms)',
        m['normalized_nosil_PER_percent'], m['text_free']['20'], zap / 'final_test.json', epoch=40)

    order = ['CTC', 'OTTC CE', 'OTTC upstream-JSD', 'OTTC standard-JSD',
             'CIF CE+Quantity', 'CIF CE+Quantity+CTC', 'CIF CE+Quantity+OTTC']
    group_order = list(dict.fromkeys(r['group'] for r in rows))
    rows.sort(key=lambda r: (group_order.index(r['group']), order.index(r['model']) if r['model'] in order else 99))
    average_path = ROOT / 'reproduction/libri100_singlehead_average_20261016/COMPARE_COMPLETE.json'
    averages = load(average_path)
    assert averages['all_complete']
    track(Path(__file__))
    track(SOURCE)
    audit = dict(verified_at=datetime.datetime.now().astimezone().isoformat(),
                 evaluation_only=True, new_training_or_inference=False, rows=rows,
                 sources_sha256=hashes, checkpoint_average_rows=averages['rows'])
    (OUT / 'metrics.json').write_text(json.dumps(audit, indent=2) + '\n')
    fields = ['group', 'model', 'PER', 'precision', 'recall', 'F1', 'R_value', 'source', 'epoch', 'step',
              'prediction_count', 'exact_boundary_count_replay']
    with (OUT / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# CTC、纯 OTTC 与 CIF：模型全集比较', '',
             '本轮主表纳入单头 Wav2Vec2-large-LV60：CTC、纯OTTC CE、原 upstream-JSD、standard-JSD、CIF CE+Quantity、CIF+CTC、CIF+OTTC；另列 WavLM 与 scratch Zapformer。仅重用训练/预测/评分产物，没有新训练或推理。', '',
             '主表均为无文本识别与分段，PER去sil；20ms容差、固定20ms时钟、同一参考、一对一边界匹配及冻结IF-MDD R-value。CIF保留分数帧crossing。各语料内比较，不把不同语料指标汇总成排名。', '',
             '历史单头CTC/OTTC固定第40轮；CIF按dev R-value选单点。冻结日程、decoder、重复音素/静音处理不同，因此是现有完整配方比较，非单因素消融。全部单seed。Libri边界为自动对齐（身份未验证），Buckeye为人工边界。Libri CIF已统一到保留重复音素的参考，故PER有别于旧CIF报告。', '']
    for group in group_order:
        lines += ['## ' + group, '', '|模型|PER↓|P|Recall|F1↑|R-value↑|',
                  '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['group'] == group:
                lines.append('|' + r['model'] + '|' + '|'.join(f'{r[k]:.3f}' for k in ['PER', 'precision', 'recall', 'F1', 'R_value']) + '|')
        lines.append('')
    lines += ['WavLM CE使用batch128/1480步，JSD使用batch64/2920步，不能将差异全部归因于loss。scratch Zapformer为10ms、从零训练，表中保留其原生协议，不能与预训练Wav2Vec2主表直接做架构优劣归因。', '',
              '## Checkpoint平均（单独的给定音素FA结果）', '',
              'Libri100三种纯OTTC均已完成last3、dev FA best3和dev PER best3；成员仅来自保留的epoch39–40共5个step checkpoint。这不是最后3个epoch平均，也不是last-5/last-10。下表FA使用参考音素输入，与前面text-free分开。', '',
              '|目标|平均策略|测试集|PER↓|FA F1↑|FA R-value↑|', '|---|---|---|---:|---:|---:|']
    for r in averages['rows']:
        if r['split'] in ['test_clean', 'test_other', 'buckeye_test']:
            lines.append(f"|{r['objective']}|{r['label']}|{r['split']}|{r['PER_percent']:.3f}|{r['FA_F1_20ms']:.3f}|{r['FA_R_20ms']:.3f}|")
    lines += ['', 'CIF九组只保存best/latest，不能做last-5/last-10平均。', '',
              '## 其他已完成系列的入口', '',
              '- 四头联合训练及各phoneset、给定音素FA：../english_suite/RESULTS.md。其标签体系和训练任务独立，不能将四个head冒充四个独立模型或与单头主表混排。',
              '- 四头checkpoint比较：../checkpoint_comparison/README.md。',
              '- Soft-DTW诊断：../RESULTS_SOFTDTW_ZH.md；该配置出现严重删除，独立保留。',
              '- Libri100平均权重的成员、hash、PFER和完整FA：../libri100_singlehead_average_20261016/REPORT_ZH.md。', '',
              'Buckeye/Libri100没有本次同配方单头CTC对照；不拿四头CTC替代。错配的Quantity=0历史实验已排除。归档MFA CTM身份未核实，不列为MFA 3.0基线。', '',
              '来源与校验：metrics.json保存输入hash和逐组完整指标；metrics.csv可直接导入表格。原训练曲线和W&B run均保留，不插值、不改变原时间轴。', '',
              '复现：python -m reproduction.cif_comparison_20261016.all_models。']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print('Compiled', len(rows), 'model/split rows and', len(averages['rows']), 'checkpoint comparison rows.')
    print(OUT / 'REPORT_ZH.md')
    for r in rows:
        print(r['group'], r['model'], *(round(r[k], 3) for k in ['PER', 'F1', 'R_value']))


if __name__ == '__main__':
    main()
