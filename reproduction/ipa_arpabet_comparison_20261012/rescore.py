"""Auditable IPA -> canonical ARPAbet rescore of saved audio-only predictions.

No model fitting, time warping, posterior merging, or checkpoint selection.
The benchmark-edit table retains raw TIMIT closures as SIL; the native-reference
table keeps the original closure-merged speech reference and jiwer scorer.
"""
import csv
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

ROOT = Path('/work/gm64/m64000/OTTC')
BENCH = Path('/work/gm64/m64000/fa-bench')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / 'reproduction'))
import jiwer
from fabench.normalize import IPA_TO_39, TIMIT61_TO_39, CANONICAL_SET
from fabench.score.matched import edit_counts

# MFA spellings missing from FA-Bench's Charsiu/espeak IPA table. These are
# declared before scoring and never estimated from recognition errors.
ALIASES = {
    'a': 'aa', 'aː': 'aa', 'aj': 'ay', 'aw': 'aw', 'ej': 'ey', 'eː': 'ey',
    'ow': 'ow', 'ɒː': 'aa', 'ɔj': 'oy', 'əw': 'ow', 'ɛː': 'eh',
    'ɫ̩': 'l', 'ʉː': 'uw', 'ɾ̃': 'n',
}
# English ARPAbet has no separate aspiration/palatalization/labialization,
# dental/retroflex stop, labiodental nasal or approximant-v contrasts.
# Collapsing these is explicit and also reported in a sensitivity table.
ALLOPHONES = {
    'bʲ': 'b', 'dʲ': 'd', 'd̪': 'd', 'fʲ': 'f', 'fʷ': 'f',
    'kʰ': 'k', 'kʷ': 'k', 'mʲ': 'm', 'pʰ': 'p', 'pʲ': 'p', 'pʷ': 'p',
    'tʰ': 't', 'tʲ': 't', 'tʷ': 't', 't̪': 't', 'vʲ': 'v', 'vʷ': 'v',
    'ɡʷ': 'g', 'ɾʲ': 'dx', 'ɖ': 'd', 'ʈ': 't', 'ʈʲ': 't', 'ʈʷ': 't',
    'ɱ': 'm', 'ʋ': 'v',
}
TOKENS_FILE = Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/tokens.txt')
TOKENS = {int(line.rsplit(' ', 1)[1]): line.rsplit(' ', 1)[0]
          for line in TOKENS_FILE.read_text().splitlines()}
MAPS = {'aliases_only': {**IPA_TO_39, **ALIASES},
        'english_collapsed': {**IPA_TO_39, **ALIASES, **ALLOPHONES}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mapped(ids, mode):
    table = MAPS[mode]
    result = []
    for token_id in ids:
        label = TOKENS[token_id]
        if label in table:
            val = table[label]
            if val is not None and val != 'sil':
                result.append(val)
        else:
            # Never discard an unsupported phone, or disguise it as silence.
            result.append('UNMAPPED_IPA:' + label)
    return result


def jiwer_counts(ref, hyp):
    x = jiwer.process_words(' '.join(ref), ' '.join(hyp))
    return x.hits, x.substitutions, x.deletions, x.insertions


def metric(counter):
    n = counter['reference_phones']
    return {**counter, **{key + '_percent': 100 * counter[key] / n if n else None
                        for key in ['substitutions', 'deletions', 'insertions']},
            'PER_percent': 100 * sum(counter[k] for k in
                                   ['substitutions', 'deletions', 'insertions']) / n if n else None}


def add(counter, ref, hyp, scorer):
    m, s, d, i = scorer(ref, hyp)
    assert m + s + d == len(ref)
    assert m + s + i == len(hyp)
    counter.update(utterances=1, reference_phones=len(ref), predicted_phones=len(hyp),
                   matches=m, substitutions=s, deletions=d, insertions=i)
    return {'reference_phones': len(ref), 'predicted_phones': len(hyp),
            'matches': m, 'substitutions': s, 'deletions': d, 'insertions': i}


def official_keys(split):
    keys = set()
    f = BENCH / 'datasets/languages/en/timit/split' / ('dev.list' if split == 'development' else 'core_test.list')
    for line in f.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        speaker, key = line.split()
        sentence = key.split('_')[-1]
        keys.add('timit:' + speaker.lower() + '_' + sentence.lower())
    assert len(keys) == (400 if split == 'development' else 192)
    return keys, f


def sources():
    base = ROOT / 'reproduction/english_suite/mfa_evaluation'
    for name, arm, step in [('CTC', 'ctc', 70014), ('OTTC CE-cost', 'ottc', 70014),
                            ('SP-OTTC CE-cost S3T2', 'spottc', 68868), ('FiLM K4', 'film_k4', 109409)]:
        yield name, base / arm / f'step-{step}', 10
    yield 'OTTC upstream JSD', BENCH / 'data/mfa_ottc_jsd_final_20261011', 10
    yield 'SP-OTTC upstream JSD S3T2', BENCH / 'data/mfa_spottc_jsd_s3t2_10ms_20261009/evaluation', 10
    if (OUT / 's3t2ss_e5/EVAL_COMPLETE.json').exists():
        yield 'SP-OTTC upstream JSD S3T2**', OUT / 's3t2ss_e5', 5


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = None; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in ['td', 'th']: self.cell = ''
    def handle_data(self, text):
        if self.cell is not None: self.cell += text
    def handle_endtag(self, tag):
        if tag in ['td', 'th'] and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row = None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_files = {TOKENS_FILE, Path(__file__),
                   BENCH / 'fabench/normalize/en/maps.py', BENCH / 'fabench/score/matched.py'}
    refs = {}; raw_refs = {}; keys = {}
    for split in ['development', 'test']:
        f = ROOT / 'reproduction/english_suite/mfa_eval_data' / (split + '.jsonl')
        input_files.add(f)
        refs[split] = {x['id']: x for x in map(json.loads, f.open())}
        keys[split], split_file = official_keys(split); input_files.add(split_file)
        assert keys[split] <= refs[split].keys()
        raw_refs[split] = {}
        for key, row in refs[split].items():
            phn = Path(row['wav']).with_suffix('.phn'); input_files.add(phn)
            labs = [line.split()[2].lower() for line in phn.read_text().splitlines()]
            assert all(l in TIMIT61_TO_39 for l in labs), (phn, labs)
            raw_refs[split][key] = [TIMIT61_TO_39[l] for l in labs if TIMIT61_TO_39[l] is not None]
    mapping_rows = []
    for token_id, label in TOKENS.items():
        if not 3 <= token_id <= 102: continue
        row = {'id': token_id, 'ipa': label}
        for mode, table in MAPS.items():
            row[mode] = table.get(label, 'UNMAPPED_IPA:' + label)
        mapping_rows.append(row)
    (OUT / 'mapping.json').write_text(json.dumps({'base': 'FA-Bench IPA_TO_39',
        'aliases': ALIASES, 'allophone_collapses': ALLOPHONES, 'tokens': mapping_rows,
        'unmapped_policy': 'retained as a never-matching token; not deleted',
        'delete_policy': 'glottal stop ʔ/q removed by standard TIMIT-39; silence excluded from saved recognition tokens'},
        ensure_ascii=False, indent=2) + '\n')
    assert all(v is None or v in CANONICAL_SET for table in MAPS.values() for v in table.values())
    results = []; by_utterance = []; mapping_usage = []; validation = []
    for model, directory, epoch in sources():
        mx = json.loads((directory / 'metrics.json').read_text()); input_files.add(directory / 'metrics.json')
        step = mx['progress']['global_step']
        for split in ['development', 'test']:
            f = directory / (split + '_predictions.jsonl'); input_files.add(f)
            predictions = {x['id']: x for x in map(json.loads, f.open())}
            assert predictions.keys() == refs[split].keys(), (model, split, 'missing/extra predictions')
            usage = Counter(TOKENS[t] for x in predictions.values() for t in x['predicted_ids'])
            for label, count in sorted(usage.items()):
                mapping_usage.append({'model': model, 'split': split, 'ipa': label, 'count': count,
                                      'canonical': MAPS['english_collapsed'].get(label, 'UNMAPPED_IPA:' + label)})
            for subset in ['native_full', 'official']:
                selected = predictions.keys() if subset == 'native_full' else keys[split]
                protocols = ['native_ids', 'mapped_native_jiwer_aliases_only',
                             'mapped_native_jiwer_english_collapsed']
                if subset == 'official': protocols += ['raw_phn_fabench_aliases_only', 'raw_phn_fabench_english_collapsed']
                for protocol in protocols:
                    counts = Counter()
                    for key in sorted(selected):
                        ref_ids = refs[split][key]['tokens']; hyp_ids = predictions[key]['predicted_ids']
                        if protocol == 'native_ids':
                            ref, hyp = list(map(str, ref_ids)), list(map(str, hyp_ids)); scorer = jiwer_counts
                        else:
                            mode = 'aliases_only' if protocol.endswith('aliases_only') else 'english_collapsed'
                            hyp = mapped(hyp_ids, mode)
                            ref = raw_refs[split][key] if protocol.startswith('raw_phn') else mapped(ref_ids, mode)
                            scorer = edit_counts if protocol.startswith('raw_phn') else jiwer_counts
                        one = add(counts, ref, hyp, scorer)
                        if subset == 'official':
                            by_utterance.append({'model': model, 'epoch': epoch, 'step': step, 'split': split,
                                'protocol': protocol, 'id': key, **one, 'reference_arpabet': ref,
                                'predicted_arpabet': hyp})
                    result = {'model': model, 'epoch': epoch, 'step': step, 'split': split,
                              'subset': subset, 'protocol': protocol, 'input': 'audio_only', **metric(counts)}
                    results.append(result)
                    if subset == 'native_full' and protocol == 'native_ids':
                        old = mx['splits'][split]['text_free']
                        assert all(counts[k] == old[k] for k in
                                   ['utterances', 'reference_phones', 'substitutions', 'deletions', 'insertions']), (model, split)
                        validation.append({'model': model, 'split': split, 'native_counts_exact_replay': True})
            print(model, split, 'done', flush=True)
    # Published MFA aggregates are reference values; never relabel as a local
    # reproduction, or synthesize unavailable per-utterance outputs.
    pub = BENCH / 'records/202609/en/gold/phone/timit/Details.md'; input_files.add(pub)
    text = pub.read_text(); parser = Tables(); parser.feed(text[text.index('### TIMIT phone editing errors'):])
    mfa = next(r for r in parser.rows if len(r) == 32 and r[1] == 'MFA 3.4')
    reference = []
    for split, offset in [('development', 2), ('test', 17)]:
        s, d, i = map(float, mfa[offset:offset + 3])
        reference.append({'model': 'MFA 3.4 published', 'split': split, 'condition': 'clean',
            'utterances': 400 if split == 'development' else 192, 'input': 'word_transcript',
            'substitutions_percent': s, 'deletions_percent': d, 'insertions_percent': i,
            'PER_percent': 32.7 if split == 'development' else 34.8,
            'local_reproduction': False, 'rounding': 'published 1 decimal; S+D+I may differ by 0.1'})
    report = {'results': results, 'published_mfa_reference': reference, 'validation': validation,
        'backfill': True, 'protocols': {
            'mapped_native_jiwer': 'Original merged speech-only reference and saved native greedy tokens; map both to TIMIT-39, remove q/ʔ; jiwer unchanged. Mapping-only comparison.',
            'raw_phn_fabench': 'Official dev400/core-test192; original raw PHN -> FA-Bench TIMIT-39; closures mapped to SIL and retained, q dropped, no token deduplication; exact FA-Bench edit_counts. Saved native predictions excluded silence before this rescore.',
            'input_limit': 'Our predictions audio-only; published MFA uses words. Shared inventory/scorer does not establish equal-input system ranking.',
            'unknowns': 'Retain unsupported palatal/coarticulated phones as UNMAPPED_IPA symbols; no unreported filtering.',
            'selection': 'Completed models final epoch10; S3T2** fixed epoch5 chosen before inference, not test selection.'}}
    (OUT / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'matrix.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    with (OUT / 'per_utterance.jsonl').open('w') as f:
        for x in by_utterance: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    with (OUT / 'mapping_usage.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping_usage[0])); writer.writeheader(); writer.writerows(mapping_usage)
    (OUT / 'source_manifest.json').write_text(json.dumps({str(f): sha(f) for f in sorted(input_files)}, indent=2) + '\n')
    print('Saved', OUT, 'rows', len(results), flush=True)


if __name__ == '__main__':
    main()
