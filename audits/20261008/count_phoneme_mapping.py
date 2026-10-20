"""Count mapping-sensitive phones in original TIMIT PHN annotations.

Read only from IF-MDD and TIMIT. Output proportions use all original PHN
intervals (including silence/closures), not a mapped or no-sil denominator.
"""
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
OTTC = OUT.parents[1]
IFMDD = Path('/home/m64000/work/IF-MDD')
TIMIT = Path('/work/gm64/m64000/dataset/TIMIT/timit')
FOCUS = ['ao', 'zh', 'q', 'dx', 'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl']
CLOSURES = {'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl'}
COMPATIBLE = {'pcl': {'p'}, 'tcl': {'t', 'ch'}, 'kcl': {'k'},
              'bcl': {'b'}, 'dcl': {'d', 'jh'}, 'gcl': {'g'}}
tree = ast.parse((OTTC / 'ottc/tools/dataset_loader/timit.py').read_text())
fusion = next(ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'timit_phoneme_fusion_map'
                      for t in n.targets))
legacy_map = {}
for line in (IFMDD / 'data/phones.60-48-39.map').read_text().splitlines():
    fields = line.split()
    if fields:
        legacy_map[fields[0]] = fields[-1] if len(fields) > 1 else None
legacy_map.update({'sil': 'sil', 'dx': 'd'})
index = {(p.parent.name.upper(), p.stem.upper()): p for p in TIMIT.rglob('*.phn')}


def raw_rows(path):
    return [(phone, int(start), int(end))
            for start, end, phone in (line.split() for line in path.read_text().splitlines()
                                     if line.strip())]


def mapped(rows, mapping, merge=False):
    result = []
    for phone, start, end in rows:
        label = mapping.get(phone, phone)
        if label is None:
            continue
        if merge and result and result[-1][0] == label:
            result[-1] = (label, result[-1][1], end)
        else:
            result.append((label, start, end))
    return result


def summarize(paths):
    counts, samples, utterances = Counter(), Counter(), Counter()
    groups = {'ao_zh': {'ao', 'zh'}, 'q_dx': {'q', 'dx'}, 'closures': CLOSURES,
              'all_focus': set(FOCUS)}
    group_utterances = Counter()
    totals = Counter()
    closure_lengths, compatible_lengths = [], []
    for path in paths:
        rows = raw_rows(path)
        present = {p for p, _, _ in rows}
        utterances.update(present)
        for name, phones in groups.items():
            group_utterances[name] += bool(phones & present)
        for i, (phone, start, end) in enumerate(rows):
            counts[phone] += 1
            samples[phone] += end - start
            if phone in CLOSURES:
                closure_lengths.append((end-start)/16)
                if i+1 < len(rows) and rows[i+1][0] in COMPATIBLE[phone]:
                    compatible_lengths.append((end-start)/16)
        official_unmerged = mapped(rows, fusion)
        official = mapped(rows, fusion, merge=True)
        old = mapped(rows, legacy_map)
        totals['official_mapped_intervals'] += len(official)
        totals['official_adjacent_merges'] += len(official_unmerged) - len(official)
        totals['legacy_mapped_intervals'] += len(old)
        totals['legacy_adjacent_same_label_boundaries'] += sum(a[0] == b[0] for a,b in zip(old, old[1:]))
    n, duration = sum(counts.values()), sum(samples.values())
    def stat(phones, affected):
        count = sum(counts[p] for p in phones)
        length = sum(samples[p] for p in phones)
        return {'count': count, 'token_pct': 100*count/n,
                'duration_seconds': length/16000, 'duration_pct': 100*length/duration,
                'affected_utterances': affected, 'utterance_pct': 100*affected/len(paths)}
    return {'utterances': len(paths), 'raw_intervals': n,
            'raw_duration_seconds': duration/16000, 'observed_raw_phone_types': len(counts),
            'phones': {p: stat({p}, utterances[p]) for p in sorted(counts)},
            'groups': {g: stat(ps, group_utterances[g]) for g,ps in groups.items()},
            'mapping': dict(totals),
            'closures': {'compatible_with_following_stop_count': len(compatible_lengths),
                         'unmatched_count': len(closure_lengths)-len(compatible_lengths),
                         'mean_duration_ms': sum(closure_lengths)/len(closure_lengths),
                         'compatible_mean_duration_ms': sum(compatible_lengths)/len(compatible_lengths)},
            'source_phn_files': [str(p) for p in paths]}


core = json.loads((IFMDD / 'data/timit_alignments_39phone_unrounded/json/test.json').read_text())
core_paths = [index[tuple(Path(entry['wav']).stem.upper().split('_', 1))] for entry in core.values()]
readme = json.loads((OTTC / 'reproduction/provenance/readme_test_manifest.json').read_text())
readme_paths = [Path(entry['wav']).with_suffix('.phn') for entry in readme.values()]
train_paths = sorted(p for p in index.values() if p.relative_to(TIMIT).parts[0] == 'train')
sets = {'timit_core192': core_paths, 'ottc_readme_test240': readme_paths,
        'timit_original_train4620': train_paths}
results = {name: summarize(paths) for name, paths in sets.items()}
# Verify the legacy mapping against the actual old unrounded reference.
expected = [mapped(raw_rows(p), legacy_map) for p in core_paths]
actual = [[(p['phoneme'], round(p['start']*16000), round(p['end']*16000))
           for p in entry['phonemes']] for entry in core.values()]
assert expected == actual, 'Legacy mapping does not reproduce saved reference'
results['timit_core192']['legacy_reference_exactly_verified'] = True
expected_official = [mapped(raw_rows(p), fusion, merge=True) for p in readme_paths]
actual_official = [[(p['phoneme'], round(p['start']*16000), round(p['end']*16000))
                    for p in entry['phonemes']] for entry in readme.values()]
assert expected_official == actual_official, 'Official mapping does not reproduce README reference'
results['ottc_readme_test240']['official_reference_exactly_verified'] = True
provenance = {'denominator': 'All raw TIMIT PHN intervals, including silence and closure phones',
              'sample_rate': 16000, 'original_train_includes_sa1_sa2': True,
              'legacy_reference_manifest': str(IFMDD / 'data/timit_alignments_39phone_unrounded/json/test.json'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'phoneme_mapping_counts.json').write_text(json.dumps({'provenance': provenance, 'sets': results}, indent=2)+'\n')
with (OUT / 'phoneme_mapping_counts.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['dataset', 'phone', 'count', 'token_pct', 'duration_seconds', 'duration_pct', 'affected_utterances', 'utterance_pct'])
    for name,d in results.items():
        for p,s in d['phones'].items():
            writer.writerow([name,p]+[s[k] for k in ['count','token_pct','duration_seconds','duration_pct','affected_utterances','utterance_pct']])
lines = ['# TIMIT 音素映射数量审计', '',
         '比例分母为原始 `.phn` 的全部音素区间，包含静音和闭塞段。时长按原始 16 kHz 采样点计算；统计不等于 PER 或 F1 的变化。', '',
         '旧 TIMIT unrounded 192 条参考已逐区间核验，与旧映射重新生成的结果完全一致。训练集包含 SA1/SA2，README 测试集为 240 条。', '']
for name,d in results.items():
    lines += [f'## {name}', '', f"{d['utterances']} 条语音，{d['raw_intervals']} 个原始区间，时长 {d['raw_duration_seconds']:.2f} 秒。", '',
              '| 原始音素 | 个数 | 区间占比 | 时长（秒） | 时长占比 | 涉及语音数 |',
              '|---|---:|---:|---:|---:|---:|']
    for p in FOCUS:
        s=d['phones'].get(p,dict(count=0,token_pct=0,duration_seconds=0,duration_pct=0,affected_utterances=0))
        lines += [f"| {p} | {s['count']} | {s['token_pct']:.2f}% | {s['duration_seconds']:.2f} | {s['duration_pct']:.2f}% | {s['affected_utterances']} |"]
    for g,label in [('ao_zh','ao + zh'),('q_dx','q + dx'),('closures','六类闭塞音合计'),('all_focus','以上合计')]:
        s=d['groups'][g]
        lines += [f"| {label} | {s['count']} | {s['token_pct']:.2f}% | {s['duration_seconds']:.2f} | {s['duration_pct']:.2f}% | {s['affected_utterances']} |"]
    m,c=d['mapping'],d['closures']
    lines += ['', f"原始 OTTC 映射并合并相邻同名区间后：{m['official_mapped_intervals']} 个区间，合并掉 {m['official_adjacent_merges']} 个内部边界。旧 IF-MDD 常规 TIMIT 映射：{m['legacy_mapped_intervals']} 个区间，保留 {m['legacy_adjacent_same_label_boundaries']} 个相邻同名边界。", '',
              f"闭塞段中 {c['compatible_with_following_stop_count']} 个可按旧 merge_stop 规则并入后续兼容音素；{c['unmatched_count']} 个不匹配。全部闭塞段平均时长 {c['mean_duration_ms']:.2f} ms。", '']
lines += ['解释：ao/zh 是旧英语模型标签集合与 OTTC 的区别，但旧 TIMIT 39phone 参考本身也把它们合并。q/dx 是 OTTC 与旧 TIMIT 常规参考的映射区别。六类闭塞音在两套常规映射中均为 sil；若采用旧 merge_stop 评估，区别在区间边界和时长定义。']
(OUT / 'PHONEME_MAPPING_COUNTS_ZH.md').write_text('\n'.join(lines)+'\n')
for name,d in results.items():
    print(name, {k:v for k,v in d.items() if k in {'utterances','raw_intervals','mapping','closures','groups'}})
