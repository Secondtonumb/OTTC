"""Saved text-free outputs only; A-B-A counts are not automatically errors."""
import json
from pathlib import Path


def edit_counts(reference, hypothesis):
    """Deterministic minimum-edit backtrace; return S/I/D and deleted ref ids."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (reference[i-1] != hypothesis[j-1]),
                           dp[i-1][j] + 1, dp[i][j-1] + 1)
    i, j = n, m
    counts = dict(S=0, I=0, D=0)
    deleted = []
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (reference[i-1] != hypothesis[j-1]):
            counts['S'] += int(reference[i-1] != hypothesis[j-1]); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            counts['D'] += 1; deleted.append(i-1); i -= 1
        else:
            counts['I'] += 1; j -= 1
    return counts, deleted


def summarize(path, vocab, frame_ms=10):
    inverse = {v: k for k, v in vocab.items()}
    total = dict(utterances=0, reference_phones=0, S=0, I=0, D=0, aba=0,
                 aba_middle_le20ms=0, predicted_runs=0, predicted_runs_le20ms=0,
                 short_reference_le30ms=0, short_reference_deleted=0)
    examples, seen = [], set()
    def phone(p): return 'd' if p == 'dx' else p
    for line in Path(path).open():
        row = json.loads(line)
        if row['utt'] in seen: raise ValueError('duplicate utterance')
        seen.add(row['utt']); total['utterances'] += 1
        reference = [(phone(p), end-start) for p, start, end in row['reference'] if p not in ('sil', 'q')]
        hyp = [phone(p) for p in row['text_free']['hypothesis'] if p not in ('sil', 'q')]
        counts, deleted = edit_counts([p for p, _ in reference], hyp)
        total['reference_phones'] += len(reference)
        for k, v in counts.items(): total[k] += v
        short = {i for i, (_, duration) in enumerate(reference) if duration <= .03 + 1e-9}
        total['short_reference_le30ms'] += len(short)
        total['short_reference_deleted'] += len(short.intersection(deleted))
        # Keep raw vocabulary identities and silence runs in temporal patterns.
        runs = []
        for t, token in enumerate(row['text_free']['frame_token_ids']):
            p = inverse[token]  # Unknown symbols must fail loudly.
            if runs and runs[-1][0] == p: runs[-1][2] = t + 1
            else: runs.append([p, t, t + 1])
        speech = [r for r in runs if r[0] not in ('sil', 'q', '<pad>')]
        total['predicted_runs'] += len(speech)
        total['predicted_runs_le20ms'] += sum((e-s)*frame_ms <= 20 for _, s, e in speech)
        for left, mid, right in zip(runs, runs[1:], runs[2:]):
            if left[0] == right[0] and left[0] != mid[0] and all(r[0] not in ('sil', 'q', '<pad>') for r in (left, mid, right)):
                total['aba'] += 1
                short_middle = (mid[2] - mid[1]) * frame_ms <= 20
                total['aba_middle_le20ms'] += short_middle
                if len(examples) < 50:
                    examples.append(dict(utt=row['utt'], phones=[left[0], mid[0], right[0]],
                        middle_start_ms=mid[1]*frame_ms, middle_duration_ms=(mid[2]-mid[1])*frame_ms))
    total['PER_percent'] = 100 * sum(total[k] for k in ('S', 'I', 'D')) / max(total['reference_phones'], 1)
    total['short_reference_deletion_rate'] = total['short_reference_deleted'] / max(total['short_reference_le30ms'], 1)
    total['aba_short_per_1000_reference_phones'] = 1000 * total['aba_middle_le20ms'] / max(total['reference_phones'], 1)
    return dict(counts=total, examples=examples,
        protocol='audio-only raw frame runs; ABA is a pattern, not an error label; S/I/D remove sil/q and map dx->d; short deletions use deterministic minimum-edit alignment, tie-dependent; never edit predictions')
