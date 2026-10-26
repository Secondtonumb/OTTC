"""Frozen validated PanPhon feature distance on saved audio-only predictions."""
import importlib.util
import json
from pathlib import Path

DISTANCE = Path('/work/gm64/m64000/fa-bench/analysis/panphon_pfer/distance.py')
TABLE = Path('/work/gm64/m64000/seamless_communication/outputs/panphon_libri100/panphon_table.json')
_backend = None
_table = None


def backend():
    global _backend, _table
    if _backend is None:
        spec = importlib.util.spec_from_file_location('timit_seed_feature_distance', DISTANCE)
        _backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_backend)
        _table = json.loads(TABLE.read_text())['phones']
    return _backend, _table


def normalized(phones):
    return ['d' if phone == 'dx' else phone for phone in phones if phone not in ('sil', 'q')]


def score_predictions(path, metrics):
    distance, table = backend()
    def vectors(phones):
        missing = set(phones) - set(table)
        if missing:
            raise ValueError(f'Unmapped PFER phones: {sorted(missing)}')
        return [vector for phone in phones for vector in table[phone]['segment_features']]
    cost = 0.
    denominator = 0
    phone_errors = 0
    reference_phones = 0
    seen = set()
    for line in Path(path).open():
        row = json.loads(line)
        assert row['utt'] not in seen
        seen.add(row['utt'])
        reference = normalized([phone for phone, _, _ in row['reference']])
        hypothesis = normalized(row['text_free']['hypothesis'])
        ref_vectors, hyp_vectors = vectors(reference), vectors(hypothesis)
        cost += distance.feature_distance(hyp_vectors, ref_vectors)
        denominator += len(ref_vectors)
        phone_errors += distance.token_distance(hypothesis, reference)
        reference_phones += len(reference)
    assert len(seen) == metrics['utterances']
    per = 100 * phone_errors / reference_phones
    assert abs(per - metrics['modes']['text_free']['nosil_PER_percent']) < 1e-8
    return dict(PFER_percent=100 * cost / denominator, feature_edit_cost=cost,
                reference_IPA_segments=denominator, utterances=len(seen), PER_crosscheck_percent=per,
                definition='validated PanPhon0.22.2 unweighted feature edit distance / pooled reference IPA segments',
                normalization='remove sil/q; dx->d, identical to PER; diphthongs expanded from frozen feature table',
                input='audio-only text_free hypothesis; never forced-alignment target phones',
                distance_source=str(DISTANCE), feature_table=str(TABLE))
