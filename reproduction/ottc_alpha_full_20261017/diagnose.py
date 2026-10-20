"""Post-hoc scoring diagnostics; never changes decoding or selects parameters."""
import json
from collections import Counter

import numpy as np

from reproduction.ottc_alpha_full_20261017.prepare import HERE, write
from reproduction.segmentation_metrics import _scorer


def main():
    result = json.loads((HERE / 'RESULTS.json').read_text())
    assert result['status'] == 'complete'
    groups = {}
    for line in (HERE / 'per_utterance_metrics.jsonl').open():
        row = json.loads(line)
        group = 'equal_native_phone_count' if row['predicted_phone_count'] == row['reference_phone_count'] else 'different_native_phone_count'
        g = groups.setdefault(row['split'], {}).setdefault(group, dict(utterances=0, methods={}))
        g['utterances'] += 1
        for method, by_ms in row['methods'].items():
            g['methods'].setdefault(method, Counter()).update(by_ms['20'])
    for by_group in groups.values():
        for g in by_group.values():
            for method, c in list(g['methods'].items()):
                g['methods'][method] = _scorer.finalize_counts(c['matches'], c['matches'], c['predicted'], c['reference'])
    deviations = {}
    for model in result['models']:
        for line in (HERE / (model['dataset'] + '_predictions.jsonl')).open():
            row = json.loads(line)
            g = deviations.setdefault(row['split'], dict(equal_boundary_count_utterances=0, different_boundary_count_utterances=0, values=[]))
            argmax, alpha = row['argmax_boundaries'], row['native_alpha_boundaries']
            if len(argmax) == len(alpha):
                g['equal_boundary_count_utterances'] += 1
                g['values'].extend((1000 * np.abs(np.asarray(argmax) - np.asarray(alpha))).tolist())
            else:
                g['different_boundary_count_utterances'] += 1
    for g in deviations.values():
        values = np.asarray(g.pop('values'))
        g.update(paired_internal_boundaries=len(values), mean_abs_alpha_vs_argmax_ms=float(values.mean()),
                 median_abs_alpha_vs_argmax_ms=float(np.median(values)), p95_abs_alpha_vs_argmax_ms=float(np.quantile(values, .95)))
    reference_grid = {}
    refs = json.loads((HERE / 'SCORING_REFERENCES.json').read_text())
    for r in refs.values():
        bounds = sorted({float(t) for _, s, e in r['reference'] for t in (s, e) if .02 < t < r['duration'] - .02})
        g = reference_grid.setdefault(r['split'], Counter())
        g.update(boundaries=len(bounds), on_10ms_grid=sum(abs(t / .01 - round(t / .01)) < 1e-6 for t in bounds),
                 on_20ms_grid=sum(abs(t / .02 - round(t / .02)) < 1e-6 for t in bounds))
    effects = {}
    tolerance_effects = {}
    for split, s in result['splits'].items():
        m = s['methods']
        effects[split] = dict(alpha_fractional_minus_argmax_F1_pp=m['alpha_fractional']['20']['f1_pct'] - m['argmax']['20']['f1_pct'],
            alpha_grid_minus_argmax_F1_pp=m['alpha_grid']['20']['f1_pct'] - m['argmax']['20']['f1_pct'],
            alpha_fractional_minus_grid_F1_pp=m['alpha_fractional']['20']['f1_pct'] - m['alpha_grid']['20']['f1_pct'],
            alpha_fractional_minus_argmax_R_value_pp=m['alpha_fractional']['20']['r_value_pct'] - m['argmax']['20']['r_value_pct'])
        tolerance_effects[split] = {str(ms): dict(
            alpha_fractional_minus_argmax_F1_pp=m['alpha_fractional'][str(ms)]['f1_pct'] - m['argmax'][str(ms)]['f1_pct'],
            alpha_fractional_minus_grid_F1_pp=m['alpha_fractional'][str(ms)]['f1_pct'] - m['alpha_grid'][str(ms)]['f1_pct']) for ms in [10, 20, 25, 30]}
    output = dict(status='complete', utterances=7590, effects_at20ms=effects,
        tolerance_effects=tolerance_effects, reference_boundary_grid_counts=reference_grid,
        native_phone_count_subgroups_at20ms=groups, alpha_vs_argmax_boundary_displacement=deviations,
        interpretation='Post-hoc descriptive analysis only, no tuned parameters or causal attribution. Native reference counts include corpus-specific silence/repetition conventions; mismatches are not all recognition errors.')
    write(HERE / 'DIAGNOSTICS.json', output)
    lines = ['# 全量结果的事后诊断', '', '以下只做描述性统计，未用于调整解码、挑容差或选择测试结果。', '',
        '|测试集|α取整−argmax F1 pp|α插值−α取整 F1 pp|α插值−argmax F1 pp|α插值−argmax R pp|', '|---|---:|---:|---:|---:|']
    for split, e in effects.items():
        lines.append(f'|{split}|{e["alpha_grid_minus_argmax_F1_pp"]:+.2f}|{e["alpha_fractional_minus_grid_F1_pp"]:+.2f}|{e["alpha_fractional_minus_argmax_F1_pp"]:+.2f}|{e["alpha_fractional_minus_argmax_R_value_pp"]:+.2f}|')
    lines += ['', '“插值−取整”单独衡量帧内时间戳的影响；“α−argmax”还包含全句累计质量重分段。', '',
        '|测试集|边界数相同语音数|α相对argmax的平均位移 ms|中位位移 ms|95%分位 ms|', '|---|---:|---:|---:|---:|']
    for split, g in deviations.items():
        lines.append(f'|{split}|{g["equal_boundary_count_utterances"]}|{g["mean_abs_alpha_vs_argmax_ms"]:.2f}|{g["median_abs_alpha_vs_argmax_ms"]:.2f}|{g["p95_abs_alpha_vs_argmax_ms"]:.2f}|')
    lines += ['', '上表只在两种方法内部边界数完全相同时按顺序配对，未用参考时间匹配。', '',
        '|测试集|预测/参考原生音素数相同？|条数|argmax F1|α取整 F1|α插值 F1|', '|---|---|---:|---:|---:|---:|']
    for split, by_group in groups.items():
        for name, g in by_group.items():
            f1 = [g['methods'][m]['f1_pct'] for m in ['argmax', 'alpha_grid', 'alpha_fractional']]
            lines.append(f'|{split}|{"是" if name.startswith("equal") else "否"}|{g["utterances"]}|{f1[0]:.2f}|{f1[1]:.2f}|{f1[2]:.2f}|')
    lines += ['', '数量分组仅在评分后使用参考，不回传给解码器。参考与预测的静音/重复音素约定不同，尤其Libri参考去除了首尾静音，因此数量差异不能全部归因于识别错误；同数量也不表示序列正确。这些分组不构成因果实验。']
    lines += ['', '|测试集|参考边界数|参考在10ms网格 %|参考在20ms网格 %|α插值−α取整 F1 pp @20ms|同差值 @25ms|', '|---|---:|---:|---:|---:|---:|']
    for split, g in reference_grid.items():
        lines.append(f'|{split}|{g["boundaries"]}|{100*g["on_10ms_grid"]/g["boundaries"]:.2f}|{100*g["on_20ms_grid"]/g["boundaries"]:.2f}|{tolerance_effects[split]["20"]["alpha_fractional_minus_grid_F1_pp"]:+.2f}|{tolerance_effects[split]["25"]["alpha_fractional_minus_grid_F1_pp"]:+.2f}|')
    lines += ['', '参考量化网格与评分门限会影响插值前后的匹配计数：在容差边缘的网格点，连续移动可改变是否计为命中。上表是敏感性描述，不能将固定20ms评分差值直接解释为同等幅度的真实时间误差变化；主结果仍保留事先指定的20ms，完整10/20/25/30ms均报告。']
    (HERE / 'DIAGNOSTICS_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(effects), flush=True)


if __name__ == '__main__':
    main()
