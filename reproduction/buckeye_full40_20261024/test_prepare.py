import sys
import unittest
import tempfile
import json
import hashlib
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import classify_phone, frame_span, parse_tier
from reproduction.buckeye_dataset import load_buckeye_rows


class TimestampContractTests(unittest.TestCase):
    def test_missing_label_advances_endpoint(self):
        rows, issues = parse_tier('#\n0.1 122 ah\n0.2 122 \n0.3 122 t\n', 's01', 'phones')
        self.assertEqual(rows[2].start, Decimal('.2'))
        self.assertEqual([i['reason'] for i in issues], ['missing_label'])

    def test_internal_carriage_return_does_not_create_an_endpoint(self):
        rows, issues = parse_tier('#\n0.1 122 the; dh ah\r; dh ah; DT\n0.2 122 end\n', 's01', 'words')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].start, Decimal('.1'))
        self.assertFalse(issues)

    def test_reversal_and_following_uncertain_interval_are_flagged(self):
        _, issues = parse_tier('#\n2 122 a\n1 122 b\n3 122 c\n4 122 d\n', 's01', 'words')
        self.assertEqual([i['reason'] for i in issues], ['time_reversal', 'interval_after_reversal'])

    def test_exact_sample_span_and_relative_precision(self):
        first, last = frame_span(13.082, 13.173013, 1000000)
        self.assertEqual((first, last), (209312, 210769))
        relative = Decimal('13.173013') - Decimal(first) / 16000
        self.assertEqual(relative + Decimal(first) / 16000, Decimal('13.173013'))

    def test_out_of_audio_range_is_clipped_only_to_valid_samples(self):
        self.assertEqual(frame_span(-.001, 1.001, 16000), (0, 16000))

    def test_negative_first_marker_is_not_an_adjacent_reversal(self):
        _, issues = parse_tier('#\n-0.001 122 {B_TRANS}\n0.01 122 SIL\n', 's01', 'phones')
        self.assertNotIn('time_reversal', [i['reason'] for i in issues])
        self.assertIn('negative_endpoint', [i['reason'] for i in issues])


class InventoryContractTests(unittest.TestCase):
    def test_historical_nonidentity_mappings(self):
        for source, expected in {'dx':'d', 'zh':'sh', 'ao':'aa', 'nx':'n', 'tq':'t',
                                 'eng':'ng', 'own':'ow', 'ah+1':'ah'}.items():
            self.assertEqual(classify_phone(source)[:2], (expected, 'speech'))

    def test_full_corpus_variants_are_explicit(self):
        self.assertEqual(classify_phone('ah+1n')[:2], ('ah', 'speech'))
        self.assertEqual(classify_phone('j')[:2], ('jh', 'speech'))

    def test_no_invented_internal_boundaries(self):
        self.assertEqual(classify_phone('ah n')[1], 'composite_without_internal_gold_boundary')

    def test_unknown_does_not_disappear_or_become_silence(self):
        self.assertEqual(classify_phone('id')[1], 'unmapped_phone')

    def test_known_events_and_q_have_distinct_actions(self):
        self.assertEqual(classify_phone('SIL')[:2], ('sil', 'event'))
        self.assertEqual(classify_phone('<EXCLUDE-name>')[:2], ('sil', 'event'))
        self.assertEqual(classify_phone('q')[1], 'historical_q_exclusion')


class EvaluationLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name)
        (self.data/'test.jsonl').write_text(json.dumps(dict(speaker='s01'))+'\n')
        (self.data/'vocab.json').write_text(json.dumps(dict(ah=2)))
        self.complete = dict(dataset_id='buckeye_full40_20261022', source_sha256='source',
            manifests_sha256={'test':hashlib.sha256((self.data/'test.jsonl').read_bytes()).hexdigest()},
            metadata_sha256={'vocab.json':'vocab'})
        self.mock = patch('reproduction.buckeye_dataset.resolve_buckeye', return_value=(self.data,self.complete))
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temporary.cleanup()

    def test_manifest_mutation_is_rejected(self):
        (self.data/'test.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            load_buckeye_rows()

    def test_training_speaker_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            load_buckeye_rows(training_speakers=['s01'])

    def test_inventory_id_mismatch_requires_adapter(self):
        with self.assertRaisesRegex(ValueError, 'inventory/IDs'):
            load_buckeye_rows(model_vocab={'ah':3})


if __name__ == '__main__':
    unittest.main()
