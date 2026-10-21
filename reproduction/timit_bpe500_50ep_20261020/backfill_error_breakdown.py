"""Local-only I/D/S backfill from saved text; no model or tracking imports."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'error_breakdown'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(reference, hypothesis):
    # Unit-cost Levenshtein. Ties: diagonal (match/sub), deletion, insertion.
    previous = [(j, j, 0, 0) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            cost, ins, delete, sub = previous[j - 1]
            diagonal = (cost + (ref != hyp), ins, delete, sub + (ref != hyp))
            cost, ins, delete, sub = previous[j]
            deletion = (cost + 1, ins, delete + 1, sub)
            cost, ins, delete, sub = current[j - 1]
            insertion = (cost + 1, ins + 1, delete, sub)
            current.append(min((diagonal, deletion, insertion), key=lambda x: x[0]))
        previous = current
    errors, ins, delete, sub = previous[-1]
    assert errors == ins + delete + sub
    return dict(INS=ins, DEL=delete, SUB=sub, errors=errors, reference_length=len(reference))


def main():
    OUT.mkdir(exist_ok=True)
    records, sources = [], []
    with (OUT / 'per_utterance.jsonl').open('w') as per_utterance:
        for model in ['ctc', 'ottc_ce', 'ottc_jsd', 'spottc_ce', 'spottc_jsd']:
            root = HERE / 'models' / model
            result_path = root / 'RESULTS.json'
            result = json.loads(result_path.read_text())
            selected = result['selected_by_dev_WER']
            for candidate in ['best_dev', 'last5', 'last10']:
                for split in ['dev', 'test']:
                    path = root / 'evaluation' / candidate / (split + '.jsonl')
                    totals = {level: dict(INS=0, DEL=0, SUB=0, errors=0, reference_length=0)
                              for level in ['word', 'character']}
                    ids = set()
                    for line in path.open():
                        row = json.loads(line)
                        assert row['utt'] not in ids
                        ids.add(row['utt'])
                        item = dict(model=model, candidate=candidate, split=split, utt=row['utt'])
                        for level in totals:
                            ref, hyp = row['reference'], row['hypothesis']
                            if level == 'word':
                                ref, hyp = ref.split(), hyp.split()
                            item[level] = counts(ref, hyp)
                            for key, value in item[level].items():
                                totals[level][key] += value
                        per_utterance.write(json.dumps(item) + '\n')
                    original = result['candidates'][candidate][split]
                    assert len(ids) == original['utterances']
                    for level, total in totals.items():
                        error_key, length_key, rate_key = (
                            ('word_errors', 'reference_words', 'WER_percent') if level == 'word'
                            else ('char_errors', 'reference_characters', 'CER_percent'))
                        assert total['errors'] == original[error_key], (model, candidate, split, level)
                        assert total['reference_length'] == original[length_key]
                        denominator = total['reference_length']
                        rate = 100 * total['errors'] / denominator
                        assert abs(rate - original[rate_key]) < 1e-9
                        records.append(dict(model=model, candidate=candidate, split=split, level=level,
                            selected_by_dev_WER=candidate == selected, utterances=len(ids), **total,
                            INS_percent=100 * total['INS'] / denominator,
                            DEL_percent=100 * total['DEL'] / denominator,
                            SUB_percent=100 * total['SUB'] / denominator, error_rate_percent=rate))
                    sources.append(dict(model=model, candidate=candidate, split=split,
                        predictions=str(path), predictions_sha256=sha256(path),
                        original_results=str(result_path), original_results_sha256=sha256(result_path),
                        checkpoint_provenance=result['candidates'][candidate]['provenance']))
    protocol = dict(backfill=True, created_at=datetime.now(timezone.utc).isoformat(), evaluation_uploads=False,
        input='Saved normalized reference/hypothesis, no further normalization',
        word_unit='Whitespace split', character_unit='Unicode characters, including spaces (original CER protocol)',
        algorithm='Unit-cost Levenshtein; equal-cost ties prefer diagonal, deletion, insertion',
        ambiguity='Multiple optimal alignments can have different I/D/S counts; tie rule is fixed here.',
        percentages='100 * count / reference length; INS+DEL+SUB equals WER or CER',
        selection='Reuse original selected_by_dev_WER; no test selection',
        validation='All 60 level totals, denominators and rates match original RESULTS.json',
        script_sha256=sha256(Path(__file__)))
    (OUT / 'summary.json').write_text(json.dumps(dict(protocol=protocol, records=records, sources=sources), indent=2) + '\n')
    with (OUT / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# TIMIT BPE500 error breakdown (local backfill)', '',
        'Counts and rates use saved normalized text. Character counts include spaces, matching original CER.',
        'All totals match the original metrics. Equal-cost alignments use diagonal > deletion > insertion.',
        'Percentages use reference length. Original dev-WER checkpoint selection is preserved.', '',
        '| Model | Candidate | Split | Level | INS | DEL | SUB | INS % | DEL % | SUB % | Error % |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append('| {model} | {candidate} | {split} | {level} | {INS} | {DEL} | {SUB} | {INS_percent:.2f} | {DEL_percent:.2f} | {SUB_percent:.2f} | {error_rate_percent:.2f} |'.format(**r))
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(OUT), metric_rows=len(records), validated=True)))
    for r in records:
        if r['selected_by_dev_WER'] and r['split'] == 'test':
            print(json.dumps(r))


if __name__ == '__main__':
    main()
