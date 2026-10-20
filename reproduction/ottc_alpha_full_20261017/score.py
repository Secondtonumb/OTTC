"""Full-set scoring only: references enter here after audio-only decoding."""
import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from reproduction.ottc_alpha_full_20261017.prepare import ROOT, HERE, digest, write
from reproduction.segmentation_metrics import _scorer, SOURCE

METHODS = dict(argmax='argmax_boundaries', alpha_grid='native_alpha_grid_boundaries',
               alpha_fractional='native_alpha_boundaries')
TOLERANCES = [10, 20, 25, 30]


def normalize(phones):
    return ['d' if p == 'dx' else p for p in phones if p not in ('sil', 'q')]


def distance(a, b):
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        previous = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            current = [i]
            for j, y in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
            previous = current
        return previous[-1]


def match_function():
    source = ROOT / 'reproduction/buckeye_singlehead_20261014/train.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'boundary_matches')
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return ns['boundary_matches'], source


def historical(split):
    if split == 'timit_core_test_192':
        path = ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_jsd/metrics.json'
        old = json.loads(path.read_text())['splits'][split]
    else:
        dataset = 'buckeye' if split == 'buckeye_test' else 'libri100'
        path = ROOT / f'reproduction/{dataset}_singlehead_20261014/evaluation/jsd_upstream/metrics.json'
        part = 'test' if dataset == 'buckeye' else split.replace('libri_', '')
        old = json.loads(path.read_text())['splits'][part]['text_free']
    return old, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true', help='Write clearly labeled interim results for completed model files')
    args = parser.parse_args()
    inputs = json.loads((HERE / 'INPUTS.json').read_text())
    if not args.partial:
        complete = json.loads((HERE / 'DECODE_COMPLETE.json').read_text())
        assert complete['utterances'] == 7590
    refs = json.loads((HERE / 'SCORING_REFERENCES.json').read_text())
    prepare = json.loads((HERE / 'PREPARE_COMPLETE.json').read_text())
    assert digest(HERE / 'SCORING_REFERENCES.json') == prepare['references_sha256']
    match, matcher_source = match_function()
    totals, diag, phones, seen, per_case, sources = {}, {}, {}, set(), [], {}
    summaries = []
    for job in inputs['jobs']:
        marker = HERE / (job['dataset'] + '_COMPLETE.json')
        if not marker.exists():
            if args.partial:
                continue
            raise AssertionError('Incomplete model: ' + job['dataset'])
        model = json.loads(marker.read_text())
        path = HERE / (job['dataset'] + '_predictions.jsonl')
        assert digest(path) == model['predictions_sha256']
        summaries.append(model)
        sources[str(path)] = digest(path)
        expected_ids = {c['recording_id'] for c in job['cases']}
        model_ids = set()
        for line in path.open():
            row = json.loads(line)
            rid, split, duration = row['recording_id'], row['split'], row['duration']
            assert rid not in seen and rid in expected_ids
            seen.add(rid)
            model_ids.add(rid)
            assert row['recognition_replay_identical']
            assert not any(row[k] for k in ['reference_phone_count_used', 'reference_phone_labels_used', 'reference_times_used'])
            reference = refs[rid]['reference']
            rb = sorted({float(t) for _, s, e in reference for t in (s, e) if .02 < t < duration - .02})
            saved_rb = refs[rid]['saved_reference_boundaries']
            if saved_rb is not None:
                assert rb == saved_rb, ('reference boundary protocol mismatch', rid)
            rp = normalize([p for p, _, _ in reference])
            hp = normalize(row['hypothesis'])
            rec = phones.setdefault(split, Counter())
            err = distance(rp, hp)
            rec.update(errors=err, reference_phones=len(rp), hypothesis_phones=len(hp), utterances=1)
            d = diag.setdefault(split, Counter())
            d.update(empty_hypotheses=row['empty_hypothesis'], rounded_boundary_collisions=row['rounded_boundary_collisions'],
                     subframe_boundary_count=row['subframe_boundary_count'], sum_abs_subframe_shift_ms=row['sum_abs_subframe_shift_ms'],
                     no_speech_reference=not bool(rp),
                     greedy_blank_frames=sum(v == 0 for v in row['frame_token_ids']), total_frames=row['frame_count'],
                     dense_marginal_checks=row['dense_transport_marginals_verified'])
            record = dict(recording_id=rid, split=split, normalized_nosil_errors=err,
                normalized_nosil_reference_phones=len(rp), predicted_phone_count=row['predicted_phone_count'],
                reference_phone_count=len(reference), methods={})
            for method, key in METHODS.items():
                pb = sorted({float(t) for t in row[key] if .02 < t < duration - .02})
                record['methods'][method] = {}
                for ms in TOLERANCES:
                    hits = match(rb, pb, ms / 1000)
                    counts = totals.setdefault(split, {}).setdefault(method, {}).setdefault(str(ms), Counter())
                    counts.update(matches=hits, predicted=len(pb), reference=len(rb))
                    record['methods'][method][str(ms)] = dict(matches=hits, predicted=len(pb), reference=len(rb))
            per_case.append(record)
        assert model_ids == expected_ids
    splits, flat = {}, []
    for split, by_method in totals.items():
        old, path = historical(split)
        sources[str(path)] = digest(path)
        for ms in [10, 20, 30]:
            new = by_method['argmax'][str(ms)]
            prior = old['boundaries'][str(ms)]
            for key in ['matches', 'reference', 'predicted']:
                assert new[key] == prior[key], ('Historical argmax mismatch', split, ms, key, new[key], prior[key])
        rec = phones[split]
        per = 100 * rec['errors'] / rec['reference_phones']
        old_per = 100 * old['normalized_nosil_PER'] if split == 'timit_core_test_192' else old['nosil_PER_percent']
        assert abs(per - old_per) < 1e-8, ('PER mismatch', split, per, old_per)
        assert rec['utterances'] == inputs['expected_by_split'][split] == old['utterances']
        metrics = {}
        for method, by_ms in by_method.items():
            metrics[method] = {}
            for ms, c in by_ms.items():
                metric = _scorer.finalize_counts(c['matches'], c['matches'], c['predicted'], c['reference'])
                metrics[method][ms] = metric
                flat.append(dict(split=split, utterances=rec['utterances'], method=method, tolerance_ms=int(ms),
                    normalized_nosil_PER_percent=per,
                    **{k: metric[k] for k in ['matches', 'predicted_boundaries', 'reference_boundaries',
                                             'precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct']}))
        diagnostics = dict(diag[split])
        diagnostics['mean_abs_subframe_shift_ms'] = diagnostics['sum_abs_subframe_shift_ms'] / max(1, diagnostics['subframe_boundary_count'])
        splits[split] = dict(utterances=rec['utterances'], recognition_counts=dict(rec), normalized_nosil_PER_percent=per,
            recognition_sequence_unchanged=True, historical_argmax_counts_replayed=True,
            reference_kind=next(r['reference_kind'] for r in refs.values() if r['split'] == split),
            diagnostics=diagnostics, methods=metrics)
    if not args.partial:
        assert seen == set(refs) and len(seen) == 7590
        assert set(splits) == set(inputs['expected_by_split'])
    sources.update({str(p): digest(p) for p in [Path(__file__), matcher_source, SOURCE,
        HERE / 'SCORING_REFERENCES.json', HERE / 'INPUTS.json']})
    result = dict(status='partial' if args.partial else 'complete', utterances=len(seen), scope=inputs['scope'],
        objective='original upstream JSD; native single-head wav2vec2-large-lv60', models=summaries, splits=splits,
        protocol=dict(text_free=True, reference_used_only_by_scorer=True,
            beta='uniform 1/own greedy predicted phone count; no reference count or labels',
            frame_clock='same fixed 20ms cells for all methods; no fitted time offset',
            alpha_grid='inverse CDF boundaries rounded to nearest frame via floor(position+.5)',
            alpha_fractional='inverse CDF crossing with uniform within-frame mass assumption',
            boundary_matching='unique endpoints; exclude first/last20ms; original monotone one-to-one time-only matcher',
            tolerances_ms=TOLERANCES, pooling='micro counts over every test utterance, including empty hypotheses/references',
            silence='native recognition only; blank is not relabeled silence; no added silence paths',
            no_test_tuning=True, optimizer_updates=0, raw_training_or_evaluation_loss='not evaluated; decoding-only inference',
            r_value_source_sha256=digest(SOURCE)), source_sha256=sources)
    stem = 'INTERIM' if args.partial else 'RESULTS'
    write(HERE / (stem + '.json'), result)
    if not args.partial:
        with (HERE / 'BOUNDARY_METRICS.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        with (HERE / 'per_utterance_metrics.jsonl').open('w') as f:
            for row in per_case:
                f.write(json.dumps(row) + '\n')
    def cell(s, method):
        m = s['methods'][method]['20']
        return f'{m["f1_pct"]:.2f} / {m["r_value_pct"]:.2f}'
    lines = ['# upstream-JSD 全测试集 α 解码对比', '',
        f'状态：{result["status"]}；已覆盖 {len(seen):,} 条语音。使用三个既有单头模型，不重训，不更新参数。', '',
        '三种解码均为无文本分段，识别序列与历史结果逐条一致。α 的目标质量为 1/自身预测音素数，不使用参考音素数、序列或时间。', '',
        '下表为边界 F1 / R-value（%，20ms容差）；按全测试集累计匹配数评分。', '',
        '|测试集|条数|argmax|α取整到帧|α帧内插值|无sil PER %|', '|---|---:|---:|---:|---:|---:|']
    for split, s in splits.items():
        lines.append(f'|{split}|{s["utterances"]}|{cell(s,"argmax")}|{cell(s,"alpha_grid")}|{cell(s,"alpha_fractional")}|{s["normalized_nosil_PER_percent"]:.3f}|')
    lines += ['', 'P/R/F1/R-value 在 10/20/25/30ms 的完整结果见 BOUNDARY_METRICS.csv。原 argmax 的匹配数、边界数、PER 已与历史全量结果核对；推理前后 checkpoint hash 不变。', '',
        'argmax→α取整衡量按累计质量重新分段的作用；α取整→α插值衡量分数帧时间戳的作用。20ms 为事先沿用的主指标，未根据测试结果调时间偏移或挑容差。', '',
        'TIMIT/Buckeye 使用人工音素边界；Libri 使用既有自动对齐，模型身份未经确认，不应解释为人工标注精度。α 插值是假定一帧内质量均匀，并非新增声学时间分辨率。', '',
        '本方法保留原识别序列。若缺少 sil，不会额外推断静音路径；归一化 α 总和为1，本身不能确定音素数。', '',
        '复现：GPU 执行 `bash reproduction/ottc_alpha_full_20261017/run.sh`；CPU 执行 `python -m reproduction.ottc_alpha_full_20261017.score`。', '',
        '逐条原始 α 和边界位于各模型的 *_predictions.jsonl；参考仅在 SCORING_REFERENCES.json。源文件、输入与权重 SHA256 见 DECODE_CONTRACT.json、RESULTS.json 和各 *_COMPLETE.json。']
    (HERE / (stem + '_ZH.md')).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[7:]), flush=True)
    if not args.partial:
        write(HERE / 'SCORE_COMPLETE.json', dict(status='complete', utterances=len(seen),
            expected_by_split=inputs['expected_by_split'], historical_argmax_replayed=True,
            results_sha256=digest(HERE / 'RESULTS.json')))


if __name__ == '__main__':
    main()
