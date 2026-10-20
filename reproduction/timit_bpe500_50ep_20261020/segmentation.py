"""Text-free word-boundary scoring of BPE paths against human TIMIT WRD."""
from collections import Counter
import itertools
import json
from pathlib import Path

from reproduction.timit_bpe500_stages_20261020 import train as parent
from reproduction.native_stride_ottc_20261018.evaluate import greedy_spans
from reproduction.english_suite.mfa_protocol_comparison.scorer_source import boundary_rvalue as scorer

base = parent.base
TIMIT = Path('/work/gm64/m64000/dataset/TIMIT/timit')
PROTOCOL = dict(unit='word', mode='text_free', reference='human TIMIT .wrd sample offsets / 16000',
    clock='native 20ms frame index; no duration rescaling',
    spans='token onset to next token onset, last token to audio end; blank assigned to previous token; BPE pieces joined into words',
    boundaries='unique word start/end times; exclude boundaries within 20ms of audio edges',
    matching='frozen IF-MDD separate directional greedy precision/recall; no lexical constraint',
    tolerances_ms=[10, 20, 30], reference_text_used_for_decoding=False,
    forced_alignment='not performed; these are not supplied-transcript or phone-boundary scores',
    evaluation_uploads=False)


def load_references(rows):
    assert TIMIT.is_dir(), TIMIT
    index = {}
    for path in TIMIT.rglob('*.wrd'):
        key = path.parent.name.upper() + '_' + path.stem.upper()
        assert key not in index, key
        index[key] = path
    references = {}
    for split in ['dev', 'test']:
        references[split] = {}
        for row in rows[split]:
            path = index[row['utt']]
            words = []
            for line in path.read_text().splitlines():
                start, end, text = line.split(maxsplit=2)
                begin, finish = int(start) / 16000, int(end) / 16000
                assert 0 <= begin <= finish <= row['duration'] + 1 / 16000, (row['utt'], line)
                words.append([text, begin, finish])
            assert words, row['utt']
            references[split][row['utt']] = dict(words=words, duration=row['duration'], source=str(path),
                source_sha256=base.sha256(path),
                overlapping_word_pairs=sum(left[2] > right[1] for left, right in zip(words, words[1:])),
                zero_duration_words=sum(word[1] == word[2] for word in words),
                text_matches_manifest=parent.bpe.normalize_text(' '.join(word[0] for word in words)) == row['text'])
    return references


def word_spans(ids, tokenizer, topology, duration):
    if topology is not None:
        pieces = [(int(label.removeprefix('bpe')), begin, end)
                  for label, begin, end in greedy_spans(ids, None, topology, .02, duration)]
    else:
        pieces = []
        for token, group in itertools.groupby(enumerate(ids), key=lambda item: int(item[1])):
            first = next(group)[0]
            if token > 0:
                pieces.append([token, min(first * .02, duration), duration])
        for left, right in zip(pieces, pieces[1:]):
            left[2] = right[1]
    words, pending = [], []
    start, end = 0., 0.

    def flush():
        text = parent.bpe.normalize_text(tokenizer.decode(pending))
        return [[word, start, end] for word in text.split()]

    for token, begin, finish in pieces:
        piece = tokenizer.tokens[token]
        if piece.startswith('\u2581') and pending:
            words.extend(flush())
            pending = []
        if not pending:
            start = begin
        pending.append(token)
        end = finish
    if pending:
        words.extend(flush())
    assert ' '.join(word[0] for word in words) == base.decode_path(ids, tokenizer, topology)
    return words


def boundaries(spans, duration):
    return sorted({float(value) for _, begin, end in spans for value in (begin, end)
                   if .02 < value < duration - .02})


def score_predictions(predictions, references, tokenizer, topology):
    totals = {str(tolerance): Counter() for tolerance in PROTOCOL['tolerances_ms']}
    seen, records = set(), []
    for line in predictions.open():
        row = json.loads(line)
        assert row['utt'] not in seen
        seen.add(row['utt'])
        reference = references[row['utt']]
        spans = word_spans(row['frame_ids'], tokenizer, topology, reference['duration'])
        assert ' '.join(span[0] for span in spans) == row['hypothesis']
        expected = boundaries(reference['words'], reference['duration'])
        predicted = boundaries(spans, reference['duration'])
        for tolerance, total in totals.items():
            total.update(scorer.precision_recall_counts(predicted, expected, int(tolerance) / 1000))
        records.append(dict(utt=row['utt'], reference_words=reference['words'], predicted_words=spans,
            reference_boundaries=expected, predicted_boundaries=predicted,
            annotation_text_matches_manifest=reference['text_matches_manifest']))
    assert seen == set(references), 'Incomplete segmentation coverage'
    output = predictions.with_suffix('.word_segments.jsonl')
    with output.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True) + '\n')
    metrics = {tolerance: scorer.finalize_counts(total['precision_matches'], total['recall_matches'],
        total['predicted_boundaries'], total['reference_boundaries']) for tolerance, total in totals.items()}
    result = dict(utterances=len(seen), boundaries=metrics, protocol=PROTOCOL,
        annotation_text_mismatches=sum(not value['text_matches_manifest'] for value in references.values()),
        overlapping_reference_word_pairs=sum(value['overlapping_word_pairs'] for value in references.values()),
        zero_duration_reference_words=sum(value['zero_duration_words'] for value in references.values()),
        scorer_sha256=base.sha256(Path(scorer.__file__)), predictions_sha256=base.sha256(predictions),
        word_segments=str(output), evaluation_uploads=False)
    parent.write_json(predictions.with_suffix('.segmentation.json'), result)
    return result


def check(rows, tokenizer):
    references = load_references(rows)
    assert len(references['dev']) == 400 and len(references['test']) == 192
    tokens = tokenizer.encode('she she')
    for topology in [None, base.build_topology(tokenizer, 'spottc_s3t2')]:
        states = tokens if topology is None else [state for token in tokens for state in topology['mapping'][token]]
        ids = [0, 0] + [item for token in states for item in [token, token, 0]]
        duration = len(ids) * .02 + .1
        spans = word_spans(ids, tokenizer, topology, duration)
        assert [span[0] for span in spans] == ['she', 'she']
        edges = boundaries(spans, duration)
        metrics = scorer.score_events(edges, edges, .02)
        assert metrics['f1_pct'] > 99.99 and metrics['r_value_pct'] > 99.99
        assert word_spans([0] * 20, tokenizer, topology, .4) == []
    print('PASS: human WRD coverage dev400/test192, BPE/state word roundtrip and frozen word-boundary scorer')
    return references