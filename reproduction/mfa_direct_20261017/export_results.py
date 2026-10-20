"""Validate fresh MFA TextGrids and export four inspectable word-input alignments."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    import textgrid
    inputs = json.loads((HERE / 'INPUT_MANIFEST.json').read_text())
    downloads = json.loads((HERE / 'models/DOWNLOAD_MANIFEST.json').read_text())
    references = json.loads((ROOT / 'reproduction/segmentation_figures_20261017/examples.json').read_text())['examples']
    records, sources = [], {}

    def digest(p):
        p = Path(p)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        sources[str(p)] = h
        return h

    for row in inputs:
        found = list((HERE / 'textgrids').rglob(row['id'] + '.TextGrid'))
        assert len(found) == 1, (row['id'], found)
        path = found[0]
        t = textgrid.TextGrid()
        t.read(str(path), round_digits=9)
        tiers = [x for x in t.tiers if x.name.lower().endswith('phones')]
        assert len(tiers) == 1
        expected = references[row['id']]['duration']
        assert abs(t.maxTime - expected) <= .001
        spans, raw_labels = [], []
        for interval in tiers[0]:
            start, end = max(0., interval.minTime), min(expected, interval.maxTime)
            if end <= start:
                continue
            raw = interval.mark.strip()
            phone = re.sub(r'\d+$', '', raw.lower()) or 'sil'
            phone = {'ao': 'aa', 'zh': 'sh'}.get(phone, phone)
            spans.append([phone, start, end])
            raw_labels.append(raw)
        assert len(spans) > 5
        assert all(a[2] <= b[1] + 1e-8 for a, b in zip(spans, spans[1:]))
        records.append(dict(utt=references[row['id']]['original_id'], recording_id=row['id'],
            model='MFA 2.2.17 / english_us_arpa acoustic v3.0.0',
            alignment_type='fresh_mfa_word_transcript_forced_alignment', target='word_transcript',
            duration=expected, phones=[x[0] for x in spans], pred_phones=[x[0] for x in spans],
            pred_starts=[x[1] for x in spans], pred_ends=[x[2] for x in spans], raw_mfa_phone_labels=raw_labels,
            textgrid=str(path), textgrid_sha256=digest(path), transcript=row['transcript'],
            reference_phone_times_used=False, reference_phone_sequence_used=False))
        assert digest(row['wav']) == row['wav_sha256']
        digest(row['transcript_file'])
    for model in downloads:
        assert digest(model['path']) == model['sha256']
    for name in ['INPUT_MANIFEST.json', 'run_align.sh', 'align_api.py', 'ALIGN_API_COMPLETE.json', 'MFA_VERSION.txt', 'export_results.py']:
        digest(HERE / name)
    # Keep the historical raw-PHN scoring closure separate from the figure's
    # shared interval-edge protocol, and never infer TSE for unequal counts.
    scorer_dir = ROOT / 'reproduction/english_suite/mfa_protocol_comparison/scorer_source'
    sys.path.insert(0, str(scorer_dir))
    scorer_file = scorer_dir / 'recompute_timit_raw_phn_tse_rvalue.py'
    spec = importlib.util.spec_from_file_location('frozen_raw_phn', scorer_file)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    digest(scorer_file)
    digest(scorer_dir / 'boundary_rvalue.py')
    timit = next(r for r in records if r['recording_id'].startswith('timit__'))
    phn = Path('/work/gm64/m64000/dataset/TIMIT/timit/test/dr7/fdhc0/si1559.phn')
    digest(phn)
    ref = scorer.load_raw_phn_ref({('FDHC0', 'SI1559'): phn}, timit['utt'], sample_rate=16000., dx_map='d', closure_mode='silence')
    pred = scorer.prediction_rows(timit, dx_map='d', hyp_round_step_ms=0.)
    raw_score, _ = scorer.evaluate_pair(pred, ref, exclude_length_mismatch=True,
        filter_silence_for_metrics=True, ignore_edge_boundaries_for_tse_acc=False)
    (HERE / 'alignments.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    report = dict(completed_at=datetime.datetime.now().astimezone().isoformat(),
                  status='complete', fresh_mfa_alignment=True, utterances=4, expected_utterances=4,
                  scope='four preselected visualization examples only, not full test sets',
                  tool_version=(HERE / 'MFA_VERSION.txt').read_text().strip(),
                  acoustic_model='official english_us_arpa v3.0.0', dictionary='official english_us_arpa v3.0.0',
                  inference_input='audio plus original word transcript; no reference phones or boundary times',
                  no_model_training=True, paper_mfa_3_reproduction_claim=False,
                  records=[dict(recording_id=r['recording_id'], intervals=len(r['pred_phones']), textgrid=r['textgrid']) for r in records],
                  timit_raw_PHN=dict(protocol='historical frozen IF-MDD, dx→d, closure→silence, silence filtered, no time rounding',
                                    boundary_scope='the single selected TIMIT example', metrics=raw_score,
                                    common_TSE_comparison_performed=False),
                  models=downloads, sources_sha256=sources)
    (HERE / 'RESULTS_COMPLETE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Verified four new MFA TextGrids:', [(r['recording_id'], len(r['pred_phones'])) for r in records])
    print('TIMIT raw-PHN length match:', raw_score['length_match'], '; no full-set claim.')


if __name__ == '__main__':
    main()
