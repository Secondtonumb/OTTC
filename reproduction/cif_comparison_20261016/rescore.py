"""Compare saved CIF and historical single-head predictions without inference.

Use historical core192 references, the historical one-to-one matcher, and
the frozen IF-MDD R-value. CIF fixed20 recovers fractional frame positions
from saved duration/T times, then maps them to the historical 20ms clock.
No fitted offset or test-conditioned checkpoint selection is performed.
"""
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from reproduction.segmentation_metrics import _scorer, SOURCE

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    rows = [json.loads(line) for line in Path(p).read_text().splitlines()]
    keyed = {Path(r.get('wav', r['utt'])).stem: r for r in rows}
    assert len(rows) == len(keyed) == 192, p
    return keyed


def edits(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    evaluator = ROOT / 'reproduction/evaluate.py'
    tree = ast.parse(evaluator.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'matches')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(evaluator), 'exec'), namespace)
    matches = namespace['matches']
    mapping_path = ROOT / 'ottc/tools/dataset_loader/timit.py'
    mapping_tree = ast.parse(mapping_path.read_text())
    assignment = next(n for n in mapping_tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'timit_phoneme_fusion_map' for t in n.targets))
    mapping = dict(ast.literal_eval(assignment.value), dx='d')
    silence = {'sil', '<sil>', 'sp', 'spn', 'pau', 'h#', 'epi', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q'}

    def normalize(seq):
        return [mapping.get(p.lower().rstrip('012'), p.lower().rstrip('012'))
                for p in seq if p.lower().rstrip('012') not in silence]

    old = {
        'CTC': 'runexp_ctc_fuse_phoneme',
        'OTTC_CE': 'runexp_ottc_fuse_phoneme',
        'OTTC_upstream_JSD': 'runexp_ottc_fuse_phoneme_jsd',
        'WavLM_OTTC_upstream_JSD': 'runexp_ottc_fuse_phoneme_jsd_wavlm_20261013',
    }
    paths = {n: ROOT/'reproduction/results'/d/'timit_core_test_192.jsonl' for n, d in old.items()}
    active = json.loads((ROOT/'reproduction/cif_timit_20261016/active_runs.json').read_text())
    paths.update({'CIF_' + n: Path(p)/'final_test.predictions.jsonl' for n, p in active.items()})
    base = read(paths['OTTC_upstream_JSD'])
    cif_base = read(paths['CIF_no_ctc'])
    report = {
        'created_at': datetime.datetime.now().astimezone().isoformat(),
        'type': 'evaluation-only re-score of saved predictions; not new training or inference',
        'reference': 'Historical core-test192 39-phone reference; silence retained for raw PER; shared normalize for no-sil PER',
        'boundary_reference': 'Historical saved interval endpoints; 20ms utterance-edge exclusion',
        'matching': 'Historical sorted one-to-one greedy, tolerance20ms plus original1e-9 epsilon',
        'clock': 'native: each saved clock; fixed20: CIF positions converted to0.02sec/frame; fixed20_grid: CIF right-cell frame edges',
        'selection': 'Historical models final40-epoch weights; CIF best dev fractional R-value20ms checkpoints. No matched selection-policy claim.',
        'sources_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), evaluator, SOURCE, mapping_path, *paths.values()]},
        'runs': {},
    }
    for name, path in paths.items():
        rows = read(path)
        assert rows.keys() == base.keys() == cif_base.keys()
        totals = Counter()
        modes = {m: Counter() for m in ['native', 'fixed20', 'fixed20_grid']}
        records = []
        for key, row in rows.items():
            reference = [p for p, _, _ in base[key]['reference']]
            hypothesis = row['hypothesis']
            rec = dict(utt=key, raw_errors=edits(reference, hypothesis), raw_phones=len(reference),
                       nosil_errors=edits(normalize(reference), normalize(hypothesis)),
                       nosil_phones=len(normalize(reference)))
            totals.update({k: v for k, v in rec.items() if k != 'utt'})
            duration = cif_base[key]['duration']
            rb = base[key]['reference_boundaries']
            assert rb == sorted(set(rb))
            for mode in modes:
                if name.startswith('CIF_'):
                    if mode == 'native':
                        pb = row['boundaries']['text_free']
                    elif mode == 'fixed20':
                        pb = [v * .02 * row['frame_count'] / row['duration'] for v in row['boundaries']['text_free']]
                    else:
                        # Saved float32 timestamps have roundoff; recover integer ceil-frame indices.
                        pb = [round(v * row['frame_count'] / row['duration']) * .02
                              for v in row['boundaries']['text_free_frame']]
                    pb = [v for v in pb if .02 < v < duration - .02]
                else:
                    assert row['reference'] == base[key]['reference']
                    pb = row['predicted_boundaries']
                assert pb == sorted(pb)
                counts = dict(matches=matches(rb, pb, .02), reference=len(rb), predicted=len(pb))
                modes[mode].update(counts)
                rec[mode] = counts
            records.append(rec)
        result = dict(utterances=len(rows), raw_PER_percent=100*totals['raw_errors']/totals['raw_phones'],
                      normalized_nosil_PER_percent=100*totals['nosil_errors']/totals['nosil_phones'], counts=dict(totals))
        result['boundaries'] = {}
        for mode, c in modes.items():
            result['boundaries'][mode] = _scorer.finalize_counts(c['matches'], c['matches'], c['predicted'], c['reference'])
        if name in old:
            original = json.loads((path.parent/'metrics.json').read_text())['splits']['timit_core_test_192']
            assert abs(result['raw_PER_percent'] - original['raw_PER']*100) < 1e-8
            assert abs(result['normalized_nosil_PER_percent'] - original['normalized_nosil_PER']*100) < 1e-8
            c = modes['native']; expected = original['boundaries']['20']
            assert all(c[k] == expected[k] for k in ['matches', 'reference', 'predicted'])
            result['historical_metrics_exact_count_replay'] = True
            result['selected_epoch'] = 40
        else:
            selected = json.loads(path.with_name('final_test.json').read_text())
            assert not selected['partial']
            result['selected_epoch'] = selected['step']//73
        report['runs'][name] = result
        (HERE/(name+'.per_utterance.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    (HERE/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# CIF 与历史单头 TIMIT 实验：统一口径评分', '',
        '复用192条保存的预测，没有训练或推理。参考统一为历史core192，边界统一为20ms固定帧时钟、20ms边缘排除和旧一对一匹配；R-value复用冻结IF-MDD公式。CIF仍保留分数帧crossing，没有拟合偏移。', '',
        '|模型|选择epoch|no-sil PER %|P %|Recall %|F1 %|R-value %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, x in report['runs'].items():
        b=x['boundaries']['fixed20']
        lines.append(f"|{name}|{x['selected_epoch']}|{x['normalized_nosil_PER_percent']:.3f}|{b['precision_pct']:.3f}|{b['recall_pct']:.3f}|{b['f1_pct']:.3f}|{b['r_value_pct']:.3f}|")
    lines += ['', '历史四组的PER、匹配数、参考数和预测数均精确复算通过。CIF源报告使用7307个参考音素，旧报告为7215；此处统一用旧参考重新计算，所以数值有别于CIF原报告。', '',
        '保留原生duration/T时钟和改成固定20ms右帧边缘的敏感性结果都在metrics.json；不将原生时钟数字和固定时钟数字混报。', '',
        '训练均为40轮。前三组历史模型和CIF均为Wav2Vec2-large-LV60，4620条TIMIT训练语音；历史模型用regfrozen（前2165步冻结声学transformer），CIF全程训练transformer。旧模型取最终checkpoint，CIF按最佳dev分段R选点，非严格仅改变loss的消融。历史CTC学习率5e-5，其他上述W2v实验2e-4。WavLM行是不同编码器，只作历史参照。', '',
        '本次单seed结果尚未显示CIF分段超过历史原upstream-JSD OTTC。不能将CIF+OTTC辅助loss与原生OTTC目标直接等同；其归一化、decoder、编码器冻结策略均不同。', '',
        '复现：`python -m reproduction.cif_comparison_20261016.rescore`。数据源、源码SHA256及逐语音匹配计数均随报告保存。']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    main()
