"""Show fixed sentences, reference/hypothesis edit alignment and state progress."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'reproduction/ipa_arpabet_comparison_20261012'))
from rescore import mapped, TOKENS
import jiwer


def main():
    reference_file = ROOT / 'reproduction/spottc_jsd_s3t2_framece_libri100_20261012/eval_data/development.jsonl'
    refs = {r['id']: r for r in map(json.loads, reference_file.open())}
    ids = list(refs)[:2]
    lines = ['固定 dev400 前两句：reference 与解码输出', '',
             '句子按数据文件原顺序固定，不按好坏筛选。REF/HYP 都使用本次评分的同一 IPA→ARPAbet 映射。',
             '对齐中 * 表示插入/删除处的空位；每个完整 topology 音素 occurrence 保留，未事后合并重复音素。', '']
    results = []
    for arm in ('jsd_fs', 'jsd_only', 'ce_fs', 'ce_only'):
        for variant in ('final', 'cumulative_average'):
            file = HERE / arm / f'{variant}_development_predictions.jsonl'
            if not file.exists():
                continue
            predictions = {}
            for line in file.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r['id'] in ids:
                    predictions[r['id']] = r
            for key in ids:
                if key not in predictions:
                    continue
                r, gold = predictions[key], refs[key]
                ref = mapped(gold['tokens'], 'english_collapsed')
                hyp = mapped(r['predicted_ids'], 'english_collapsed')
                edits = jiwer.process_words(' '.join(ref), ' '.join(hyp))
                ref_aligned, hyp_aligned = [], []
                for chunk in edits.alignments[0]:
                    a = ref[chunk.ref_start_idx:chunk.ref_end_idx]
                    b = hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]
                    n = max(len(a), len(b))
                    ref_aligned.extend(a + ['*'] * (n - len(a)))
                    hyp_aligned.extend(b + ['*'] * (n - len(b)))
                runs = []
                for i, state in enumerate(r['legal_state_path']):
                    if not runs or runs[-1]['state_id'] != state:
                        runs.append(dict(state_id=state, start_frame=i, end_frame=i + 1,
                                         label='blank' if state == 0 else TOKENS[(state - 1) // 3 + 1] + ':S' + str((state - 1) % 3)))
                    else:
                        runs[-1]['end_frame'] = i + 1
                example = dict(arm=arm, checkpoint_variant=variant, id=key, wav=gold['wav'],
                    reference_ipa=[TOKENS[x] for x in gold['tokens']],
                    predicted_ipa=[TOKENS[x] for x in r['predicted_ids']],
                    reference_arpabet=ref, hypothesis_arpabet=hyp,
                    greedy_arpabet=mapped(r['greedy_ids'], 'english_collapsed'),
                    alignment_reference=ref_aligned, alignment_hypothesis=hyp_aligned,
                    matches=edits.hits, substitutions=edits.substitutions, deletions=edits.deletions,
                    insertions=edits.insertions, PER_percent=100 * edits.wer, legal_state_runs=runs)
                results.append(example)
                lines.extend([f'{arm} / {variant} / {key}', '',
                    f"参考 {len(ref)} 音素，预测 {len(hyp)} 音素；S={edits.substitutions} D={edits.deletions} I={edits.insertions}，PER={100 * edits.wer:.2f}%", '',
                    '```text', 'REF: ' + ' '.join(ref), 'HYP: ' + ' '.join(hyp),
                    'GREEDY: ' + ' '.join(example['greedy_arpabet']), '```', '', '编辑对齐：', '', '```text'])
                for start in range(0, len(ref_aligned), 25):
                    pairs = list(zip(ref_aligned[start:start + 25], hyp_aligned[start:start + 25]))
                    widths = [max(len(a), len(b), 2) for a, b in pairs]
                    lines.append('REF: ' + ' '.join(a.ljust(w) for (a, b), w in zip(pairs, widths)))
                    lines.append('HYP: ' + ' '.join(b.ljust(w) for (a, b), w in zip(pairs, widths)))
                lines.extend(['```', '', '前40段合法状态路径（每帧10ms）：', '', '```text'])
                for run in runs[:40]:
                    lines.append(f"{run['start_frame'] * 10:4d}–{run['end_frame'] * 10:4d} ms  {run['label']}")
                lines.extend(['```', ''])
    (HERE / 'examples.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'EXAMPLES.md').write_text('\n'.join(lines))
    for r in results:
        if r['checkpoint_variant'] == 'final' and r['id'] == ids[0]:
            print(json.dumps({k: v for k, v in r.items() if k not in ['legal_state_runs', 'alignment_reference', 'alignment_hypothesis']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
