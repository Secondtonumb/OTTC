"""Reference-anchored ABA excursion audit, using saved BPE frame paths only."""
import csv
import itertools
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm

from backfill_error_breakdown import HERE, counts, sha256

DATA = HERE.parent / 'timit_bpe500_20261020/data'
OUT = HERE / 'aba_diagnostic'
TOKENS = json.loads((DATA / 'tokenizer/tokens.json').read_text())
SP = spm.SentencePieceProcessor(model_file=str(DATA / 'tokenizer/bpe.model'))


def decode(ids):
    text = SP.decode(ids)
    text = unicodedata.normalize('NFKC', text).lower().replace('\u2019', "'")
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)*", text))


def alignment(reference, hypothesis):
    """Same diagonal > deletion > insertion tie rule as the I/D/S backfill."""
    matrix = [list(range(len(hypothesis) + 1))]
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            row.append(min(matrix[-1][j - 1] + (ref != hyp), matrix[-1][j] + 1, row[-1] + 1))
        matrix.append(row)
    operations = [None] * len(hypothesis)
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            operations[j - 1] = ('match' if reference[i - 1] == hypothesis[j - 1] else 'sub', i - 1)
            i, j = i - 1, j - 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            i -= 1
        else:
            operations[j - 1] = ('ins', None)
            j -= 1
    return operations, matrix[-1][-1]


def segments(frames, subphone):
    runs = []
    cursor = 0
    for token, group in itertools.groupby(frames):
        length = sum(1 for _ in group)
        if token > 0:
            if not subphone:
                runs.append(dict(token=token, start=cursor, end=cursor + length))
            else:
                phone, state = (token - 1) // 3 + 1, (token - 1) % 3
                if runs and runs[-1]['token'] == phone and state >= runs[-1]['last_state']:
                    runs[-1].update(end=cursor + length, last_state=state)
                else:
                    runs.append(dict(token=phone, start=cursor, end=cursor + length, last_state=state))
        cursor += length
    return runs


def write_report():
    summary = json.loads((OUT / 'summary.json').read_text())
    lines = ['# BPE A-B-A excursion diagnostic', '',
        'Local backfill using saved frame paths; original models and evaluation results unchanged.',
        'Structural ABA is not sufficient evidence of an error. Strict events require an optimal BPE alignment',
        'with exactly one matching A and two inserted tokens. Disjoint triples only, with no recursive collapse.',
        'Reported rescoring is reference-assisted oracle analysis, not a deployable decoding result.',
        'Counts depend on tokenization and optimal-alignment tie rules; other repetition patterns are excluded.', '',
        '## Test192, original dev-WER selection, no middle-duration limit', '',
        '| Model | All ABA | Strict ABA | Utterances | Word INS before/after | Word SUB before/after | WER before/after | CER before/after |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['records']:
        if r['selected_by_dev_WER'] and r['split'] == 'test' and r['max_middle_ms'] == 'unlimited':
            lines.append(f"| {r['model']} | {r.get('structural_ABA', 0)} | {r.get('events', 0)} | "
                f"{r.get('affected_utterances', 0)} | {r.get('word_original_INS', 0)}/{r.get('word_corrected_INS', 0)} | "
                f"{r.get('word_original_SUB', 0)}/{r.get('word_corrected_SUB', 0)} | "
                f"{r['word_original_rate']:.2f}/{r['word_corrected_rate']:.2f} | "
                f"{r['character_original_rate']:.2f}/{r['character_corrected_rate']:.2f} |")
    lines += ['', '## Duration sensitivity (middle B run)', '',
              '| Model | Split | Candidate | Middle limit ms | Events | Word errors reduced | Character errors reduced |',
              '|---|---|---|---|---:|---:|---:|']
    for r in summary['records']:
        lines.append(f"| {r['model']} | {r['split']} | {r['candidate']} | {r['max_middle_ms']} | "
            f"{r.get('events', 0)} | {r['word_errors_reduced']} | {r['character_errors_reduced']} |")
    lines += ['', '## OTTC upstream-JSD test examples', '']
    examples = [json.loads(line) for line in (OUT / 'utterances.jsonl').open()]
    examples = [r for r in examples if r['model'] == 'ottc_jsd' and r['candidate'] == 'best_dev' and r['split'] == 'test']
    examples.sort(key=lambda r: r['original_word']['SUB'] - r['corrected_word']['SUB'], reverse=True)
    for r in examples:
        lines += [f"### {r['utt']}", '', f"Reference: {r['reference']}", '',
                  f"Original: {r['hypothesis']}", '', f"Oracle collapsed: {r['oracle_collapsed_hypothesis']}", '']
    lines += ['Full provenance and protocol: `summary.json`; full numeric table: `summary.csv`.',
              'Token events and frame times: `events.jsonl`; changed utterances: `utterances.jsonl`.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True)
    manifests = {split: {r['utt']: r for r in map(json.loads, (DATA / (split + '.jsonl')).open())}
                 for split in ['dev', 'test']}
    cached = {}
    for line in (HERE / 'error_breakdown/per_utterance.jsonl').open():
        r = json.loads(line)
        cached[r['model'], r['candidate'], r['split'], r['utt']] = r
    records, provenance = [], []
    with (OUT / 'events.jsonl').open('w') as event_file, (OUT / 'utterances.jsonl').open('w') as utterance_file:
        for model in ['ctc', 'ottc_ce', 'ottc_jsd', 'spottc_ce', 'spottc_jsd']:
            results = json.loads((HERE / 'models' / model / 'RESULTS.json').read_text())
            for candidate in ['best_dev', 'last5', 'last10']:
                for split in ['dev', 'test']:
                    path = HERE / 'models' / model / 'evaluation' / candidate / (split + '.jsonl')
                    aggregate = Counter()
                    by_threshold = {limit: Counter() for limit in [1, 2, 3, 5, 999999]}
                    for line in path.open():
                        row = json.loads(line)
                        identity = dict(model=model, candidate=candidate, split=split, utt=row['utt'])
                        ref_ids = manifests[split][row['utt']]['labels']
                        assert decode(ref_ids) == row['reference']
                        runs = segments(row['frame_ids'], model.startswith('spottc'))
                        pred_ids = [r['token'] for r in runs]
                        assert decode(pred_ids) == row['hypothesis'], identity
                        operations, token_errors = alignment(ref_ids, pred_ids)
                        aggregate.update(utterances=1, reference_tokens=len(ref_ids), token_errors=token_errors,
                                         token_insertions=sum(op[0] == 'ins' for op in operations))
                        events, occupied = [], set()
                        motifs = 0
                        for pos in range(len(runs) - 2):
                            a, b, c = runs[pos:pos + 3]
                            if a['token'] != c['token'] or a['token'] == b['token']:
                                continue
                            motifs += 1
                            ops = [operations[k][0] for k in range(pos, pos + 3)]
                            # Exactly one A matches GT; B and the other A are insertions.
                            if ops not in [['match', 'ins', 'ins'], ['ins', 'ins', 'match']]:
                                continue
                            if occupied.intersection(range(pos, pos + 3)):
                                aggregate['overlapping_strict_candidates_skipped'] += 1
                                continue
                            occupied.update(range(pos, pos + 3))
                            anchor = pos if ops[0] == 'match' else pos + 2
                            remove = [k for k in range(pos, pos + 3) if k != anchor]
                            event = dict(**identity, hypothesis_token_index=pos,
                                pieces=[TOKENS[r['token']] for r in [a, b, c]],
                                reference_token_index=operations[anchor][1],
                                middle_frames=b['end'] - b['start'], middle_ms=20 * (b['end'] - b['start']),
                                start_seconds=.02 * a['start'], end_seconds=.02 * c['end'],
                                remove_indices=remove, reference=row['reference'], hypothesis=row['hypothesis'],
                                token_context=[TOKENS[token] for token in pred_ids[max(0, pos-3):pos+6]])
                            events.append(event)
                            event_file.write(json.dumps(event, ensure_ascii=False) + '\n')
                        aggregate.update(structural_ABA=motifs, strict_ABA=len(events),
                                         utterances_with_strict_ABA=bool(events))
                        baseline = cached[model, candidate, split, row['utt']]
                        for limit, total in by_threshold.items():
                            selected = [e for e in events if e['middle_frames'] <= limit]
                            removed = {k for e in selected for k in e['remove_indices']}
                            corrected_ids = [token for k, token in enumerate(pred_ids) if k not in removed]
                            corrected = decode(corrected_ids) if removed else row['hypothesis']
                            total.update(events=len(selected), affected_utterances=bool(selected),
                                         removed_inserted_tokens=len(removed))
                            corrected_errors = {}
                            for level in ['word', 'character']:
                                before = baseline[level]
                                if removed:
                                    ref, hyp = row['reference'], corrected
                                    if level == 'word':
                                        ref, hyp = ref.split(), hyp.split()
                                    after = counts(ref, hyp)
                                else:
                                    after = before
                                for field in ['INS', 'DEL', 'SUB', 'errors', 'reference_length']:
                                    total[f'{level}_original_{field}'] += before[field]
                                    total[f'{level}_corrected_{field}'] += after[field]
                                corrected_errors[level] = after
                            if selected and limit == 999999:
                                utterance_file.write(json.dumps(dict(**identity, events=len(selected),
                                    reference=row['reference'], hypothesis=row['hypothesis'],
                                    oracle_collapsed_hypothesis=corrected,
                                    original_word=baseline['word'], corrected_word=corrected_errors['word'],
                                    original_character=baseline['character'],
                                    corrected_character=corrected_errors['character']), ensure_ascii=False) + '\n')
                    for limit, total in by_threshold.items():
                        for level in ['word', 'character']:
                            original = results['candidates'][candidate][split]
                            assert total[f'{level}_original_errors'] == original['word_errors' if level == 'word' else 'char_errors']
                            total[f'{level}_errors_reduced'] = total[f'{level}_original_errors'] - total[f'{level}_corrected_errors']
                            denominator = total[f'{level}_original_reference_length']
                            total[f'{level}_original_rate'] = 100 * total[f'{level}_original_errors'] / denominator
                            total[f'{level}_corrected_rate'] = 100 * total[f'{level}_corrected_errors'] / denominator
                        records.append(dict(model=model, candidate=candidate, split=split,
                            selected_by_dev_WER=candidate == results['selected_by_dev_WER'],
                            max_middle_ms=20 * limit if limit != 999999 else 'unlimited', **aggregate, **total))
                    provenance.append(dict(path=str(path), sha256=sha256(path),
                        checkpoint=results['candidates'][candidate]['provenance']))
    protocol = dict(backfill=True, created_at=datetime.now(timezone.utc).isoformat(), evaluation_uploads=False,
        scope='BPE500, not AA/AE phone outputs', frame_ms=20,
        structural='Decoded consecutive token occurrences A B A with A != B; blank omitted per original decoder',
        strict='Optimal BPE alignment has MATCH INSERT INSERT or INSERT INSERT MATCH; all three match one GT A',
        overlap='Left-to-right disjoint triples; no iterative or recursive collapse',
        alignment_ties='Diagonal > deletion > insertion; same deterministic rule as prior backfill',
        counterfactual='Oracle: remove the two aligned insertions per strict event and rescore normalized text',
        caveats='Reference-assisted diagnostic, not deployable decoding or causal proof; BPE segmentation and optimal alignment ambiguities apply. Other repeated patterns and A blank A excluded.',
        tokenizer_sha256=sha256(DATA / 'tokenizer/bpe.model'), script_sha256=sha256(Path(__file__)))
    (OUT / 'summary.json').write_text(json.dumps(dict(protocol=protocol, records=records, sources=provenance), indent=2) + '\n')
    with (OUT / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in records))))
        writer.writeheader()
        writer.writerows(records)
    write_report()
    print('Completed', len(records), 'rows', OUT, flush=True)
    for r in records:
        if r['selected_by_dev_WER'] and r['split'] == 'test' and r['max_middle_ms'] in [20, 40, 60, 'unlimited']:
            print(json.dumps({k: r.get(k, 0) for k in ['model', 'max_middle_ms', 'structural_ABA', 'strict_ABA',
                'events', 'affected_utterances', 'token_insertions', 'removed_inserted_tokens',
                'word_original_rate', 'word_corrected_rate', 'word_errors_reduced',
                'character_original_rate', 'character_corrected_rate', 'character_errors_reduced']}), flush=True)


if __name__ == '__main__':
    main()
