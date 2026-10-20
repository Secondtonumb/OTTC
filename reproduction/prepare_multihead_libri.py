"""Prepare full LibriSpeech audio and four native phoneset targets, with provenance."""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

from prepare_multihead import dictionary, entries, merge, word_phones

ROOT = Path(__file__).resolve().parent
IFMDD = Path('/home/m64000/work/IF-MDD')
ALIGNMENTS = IFMDD/'data/librispeech_alignments_960/json'
SHAR = Path('/work/gm64/m64000/zipa/data/ipapack_plus_all/ipapack_plus_2/librispeech_shar')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['libri100', 'libri960'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import sentencepiece as spm
    spec = importlib.util.spec_from_file_location('ipapack_reader', IFMDD/'scripts/prepare_librispeech_ipapack127_json.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    dictionary_path = ROOT/'multihead_libri100/assets/english_us_mfa_v3.1.0.dict'
    lexicon, _ = dictionary(dictionary_path)
    vocabs = json.loads((ROOT/'multihead_timit/data/vocabs.json').read_text())
    tokenizer_path = Path('/work/gm64/m64000/zipa/ipa_simplified/unigram_127.model')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    assert {tokenizer.id_to_piece(i): i for i in range(tokenizer.vocab_size())} == vocabs['zipa_ipa']
    tree = ast.parse((ROOT.parent/'ottc/tools/dataset_loader/timit.py').read_text())
    fusion = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                  and any(getattr(t, 'id', None) == 'timit_phoneme_fusion_map' for t in n.targets)))
    train_subsets = ['train-clean-100'] if args.dataset == 'libri100' else ['train-clean-100', 'train-clean-360', 'train-other-500']
    splits = {'train-train': train_subsets, 'train-dev': ['dev-clean', 'dev-other'], 'test': ['test-clean', 'test-other']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'vocabs.json').write_text(json.dumps(vocabs, ensure_ascii=False, indent=2)+'\n')
    metadata = {'dataset': args.dataset, 'dictionary': str(dictionary_path), 'dictionary_sha256': sha256(dictionary_path),
                'head_vocab_sizes': {h: len(v) for h, v in vocabs.items()}, 'seed': 42,
                'duration_filter': None, 'mfa_target_source': 'MFA US IPA v3.1 highest-probability lexical pronunciation; not acoustic alignment',
                'zipa_target_source': 'Native IPA-Pack++ supervision.text and original SentencePiece IPA127 tokenizer',
                'tokenizer_sha256': sha256(tokenizer_path), 'reference_intervals_source': 'Existing LibriSpeech alignment manifest; automatic alignment, not human PHN',
                'oov_policy': 'Keep utterance; mask only MFA head for dictionary-incomplete text', 'splits': {}}
    split_speakers = {}
    for split, subsets in splits.items():
        stats = Counter()
        oov = Counter()
        subset_stats = {}
        speakers = set()
        ids = set()
        sources = {}
        with (args.output/f'{split}.jsonl').open('w') as output, (args.output/f'{split}_excluded.jsonl').open('w') as exclusions:
            for subset in subsets:
                source = ALIGNMENTS/(subset.replace('-', '_')+'.json')
                transcripts = reader.read_shar_transcripts(SHAR, (subset,))
                sources[subset] = {'manifest': str(source), 'sha256': sha256(source),
                                   'ipa_cuts': {str(p): sha256(p) for p in sorted((SHAR/subset).glob('cuts.*.jsonl.gz'))}}
                count = 0
                seconds = 0.
                for e in entries(source):
                    utt = Path(e['wav']).stem
                    if utt in ids:
                        raise ValueError(f'Duplicate utterance: {utt}')
                    ids.add(utt)
                    if not Path(e['wav']).is_file():
                        raise FileNotFoundError(e['wav'])
                    words = re.findall(r"[\w]+(?:['’\-][\w]+)*", e['wrd'].lower())
                    pronunciations = [word_phones(w, lexicon) for w in words]
                    missing = [w for w, p in zip(words, pronunciations) if p is None]
                    if missing or not words:
                        oov.update(missing)
                        stats['masked_mfa_dictionary_oov'] += 1
                        exclusions.write(json.dumps({'utt': utt, 'masked_head': 'mfa_ipa', 'words': missing})+'\n')
                    arp = e['canonical_aligned'].split()
                    original = merge([(p, 0, 0) for p in arp], fusion)
                    ipa = transcripts[utt]
                    phones = {'ottc_arpabet': [p for p, _, _ in original], 'ifmdd_arpabet': arp,
                              'mfa_ipa': [] if missing or not words else ['sil']+[p for seq in pronunciations for p in seq]+['sil'],
                              'zipa_ipa': list(tokenizer.encode(ipa, out_type=str))}
                    targets = {h: [vocabs[h][p] for p in ps] for h, ps in phones.items()}
                    for h, tokens in targets.items():
                        forbidden = {vocabs[h][p] for p in ['<unk>', '<sos/eos>', '<blank>', '<blk>', '<pad>'] if p in vocabs[h]}
                        if forbidden.intersection(tokens):
                            raise ValueError(f'Unexpected special target: {utt}, {h}')
                    rows = [(p['phoneme'], float(p['start']), float(p['end'])) for p in e['phonemes']]
                    references = {'ifmdd_arpabet': rows, 'ottc_arpabet': merge(rows, fusion)}
                    for h, ref in references.items():
                        if [p for p, _, _ in ref] != phones[h]:
                            # Existing manifests insert leading/trailing silence in the sequence.
                            extended = list(ref)
                            if phones[h][0] == 'sil' and extended[0][0] != 'sil':
                                extended.insert(0, ('sil', 0., extended[0][1]))
                            if phones[h][-1] == 'sil' and extended[-1][0] != 'sil':
                                extended.append(('sil', extended[-1][2], float(e['duration'])))
                            if [p for p, _, _ in extended] != phones[h]:
                                raise ValueError(f'Reference sequence mismatch: {utt}, {h}')
                            references[h] = extended
                    record = {'utt': utt, 'wav': e['wav'], 'speaker': e['spk_id'], 'subset': subset,
                              'duration': e['duration'], 'text': e['wrd'], 'targets': targets, 'references': references}
                    output.write(json.dumps(record, ensure_ascii=False)+'\n')
                    speakers.add(e['spk_id'])
                    count += 1
                    seconds += float(e['duration'])
                    stats['kept_utterances'] += 1
                    stats['over_20_seconds'] += int(e['duration'] > 20)
                subset_stats[subset] = {'utterances': count, 'seconds': seconds}
                print(split, subset, count, f'{seconds/3600:.3f}h', flush=True)
        metadata['splits'][split] = {**dict(stats), 'subsets': subset_stats, 'sources': sources,
                                    'oov_words': dict(oov.most_common()), 'speakers': sorted(speakers)}
        split_speakers[split] = speakers
    assert not any(split_speakers[a] & split_speakers[b] for a, b in [('train-train', 'train-dev'), ('train-train', 'test'), ('train-dev', 'test')])
    metadata['speaker_overlap'] = False
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    (args.output/'DATA_READY.json').write_text(json.dumps({'dataset': args.dataset, 'splits': {s: metadata['splits'][s]['kept_utterances'] for s in splits}})+'\n')


if __name__ == '__main__':
    main()
