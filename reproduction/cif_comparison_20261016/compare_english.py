"""Replay saved CIF and native OTTC predictions under the native OTTC protocol.

No training, inference, fitted offsets, or test-based model selection.
"""
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from reproduction.segmentation_metrics import SOURCE, _scorer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {Path(r['utt']).stem: r for r in rows}
    assert len(rows) == len(result)
    return result


def normalize(phones):
    return ['d' if p == 'dx' else p for p in phones if p not in ('sil', 'q')]


def edits(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def main():
    source = ROOT / 'reproduction/buckeye_singlehead_20261014/train.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'boundary_matches')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    matches = namespace['boundary_matches']
    active = json.loads((ROOT / 'reproduction/cif_quantity_20261016/active_runs.json').read_text())
    audit = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
                 evaluation_only=True, sources_sha256={str(p.relative_to(ROOT)): digest(p)
                 for p in [Path(__file__), source, SOURCE]}, comparisons=[])
    scenarios = [('buckeye', 'test', 'final_test', 1840),
                 ('libri100', 'test_clean', 'benchmark_libri_test_clean', 2620),
                 ('libri100', 'test_other', 'benchmark_libri_test_other', 2938),
                 ('libri100', 'buckeye_test', 'benchmark_buckeye_test', 1840)]
    for dataset, split, cif_file, expected in scenarios:
        base = ROOT / f'reproduction/{dataset}_singlehead_20261014/evaluation/jsd_upstream'
        native_path = base / f'{split}_predictions.jsonl'
        native = read(native_path)
        saved = json.loads((base / 'metrics.json').read_text())['splits'][split]['text_free']
        assert len(native) == expected
        for variant in ['native_ottc', 'ottc_jsd_0p5', 'ce', 'ctc_0p5']:
            path = native_path if variant == 'native_ottc' else Path(active[f'{dataset}_{variant}']) / f'{cif_file}.predictions.jsonl'
            rows = native if variant == 'native_ottc' else read(path)
            assert rows.keys() == native.keys()
            counts = Counter()
            for key, row in rows.items():
                reference = native[key]
                duration = reference['duration']
                if variant == 'native_ottc':
                    phones = row['text_free']['hypothesis']
                    frames = row['text_free']['frame_token_ids']
                    predicted = [i * .02 for i in range(1, len(frames)) if frames[i] != frames[i - 1]]
                else:
                    assert abs(row['duration'] - duration) < .001
                    phones = row['hypothesis']
                    # Convert the CIF fractional frame position to the native
                    # OTTC text-free fixed 20ms clock, without a fitted offset.
                    predicted = [t * row['frame_count'] * .02 / row['duration']
                                 for t in row['boundaries']['text_free']]
                    counts['reference_sequence_mismatches'] += normalize(row['reference']) != normalize([p for p, _, _ in reference['reference']])
                rb = sorted({float(t) for _, s, e in reference['reference']
                             for t in (s, e) if .02 < t < duration - .02})
                pb = sorted({float(t) for t in predicted if .02 < t < duration - .02})
                ref = normalize([p for p, _, _ in reference['reference']])
                counts.update(matches=matches(rb, pb, .02), reference=len(rb),
                              predicted=len(pb), errors=edits(ref, normalize(phones)),
                              phones=len(ref), utterances=1)
            metrics = _scorer.finalize_counts(counts['matches'], counts['matches'],
                                               counts['predicted'], counts['reference'])
            per = 100 * counts['errors'] / counts['phones']
            if variant == 'native_ottc':
                assert all(counts[k] == saved['boundaries']['20'][k]
                           for k in ['matches', 'reference', 'predicted'])
                assert abs(per - saved['nosil_PER_percent']) < 1e-8
            item = dict(training=dataset, split=split, variant=variant,
                        PER_nosil=per, boundary=metrics, counts=dict(counts))
            audit['comparisons'].append(item)
            audit['sources_sha256'][str(path.relative_to(ROOT))] = digest(path)
            print(dataset, split, variant, 'PER/F1/R',
                  *[round(x, 3) for x in [per, metrics['f1_pct'], metrics['r_value_pct']]],
                  'reference mismatches', counts['reference_sequence_mismatches'])
    (HERE / 'english_metrics.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# CIF 与纯 upstream-JSD OTTC：英语语料统一口径比较', '',
             '仅重评分保存的预测。参考为纯 OTTC 的 speech-phone 区间，去 sil/q、dx→d；边界为区间端点去重、句首句尾排除20ms、一对一匹配、20ms容差，固定20ms编码器帧时钟。CIF保留分数帧crossing，不拟合偏移。R-value复用冻结IF-MDD公式。', '',
             '全部是无文本推理。纯OTTC固定第40轮；CIF按dev无文本分段R-value选单个checkpoint。相同预训练Wav2Vec2-large-LV60、相同训练语料，但冻结日程、训练标签重复/静音处理及decoder不同，不是单因素消融。Libri参考为身份未核实的自动对齐，Buckeye为人工边界。', '',
             '|训练语料|测试集|方法|PER↓|P|Recall|F1|R-value↑|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in audit['comparisons']:
        m = r['boundary']
        lines.append(f"|{r['training']}|{r['split']}|{r['variant']}|{r['PER_nosil']:.3f}|" +
                     '|'.join(f'{m[k]:.3f}' for k in ['precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct']) + '|')
    lines += ['', '纯OTTC的原始PER和边界计数全部精确复算通过。输入文件hash、逐组参考标签差异句数及完整计数见 english_metrics.json。', '',
              '复现：python -m reproduction.cif_comparison_20261016.compare_english。']
    (HERE / 'ENGLISH_REPORT_ZH.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
