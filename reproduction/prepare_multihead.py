"""Prepare auditable four-head English targets without fabricating timestamps."""
import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from multihead import HEADS

ROOT = Path(__file__).resolve().parents[1]
IFMDD = Path('/home/m64000/work/IF-MDD')
TIMIT = Path('/work/gm64/m64000/dataset/TIMIT/timit')


def entries(path):
    block = []
    with path.open() as f:
        assert f.readline().strip() == '{'
        for line in f:
            if not block and line.strip() == '}':
                break
            if not block:
                assert line.startswith('  "') and line.rstrip().endswith(': {')
            block.append(line)
            if line.rstrip() in ('  },', '  }'):
                yield next(iter(json.loads('{'+''.join(block).rstrip().removesuffix(',')+'}').values()))
                block = []
    assert not block


def dictionary(path):
    lexicon, phones = {}, set()
    for line in path.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) not in {2,6}:
            raise ValueError(f'Expected MFA probability dictionary: {line[:80]}')
        word, prob = parts[0], float(parts[1]) if len(parts)==6 else 1.0
        pronunciation = parts[-1].split()
        phones.update(pronunciation)
        # Stable tie break: first dictionary entry with maximal probability.
        if word not in lexicon or prob > lexicon[word][0]:
            lexicon[word] = (prob, pronunciation)
    return {w:p for w,(_,p) in lexicon.items()}, phones


def timit_entries(split,path,index):
    if split!='train-train':
        yield from entries(path)
        return
    # Match original OTTC train split: all 4,620 utterances including SA1/SA2.
    for phn in sorted(p for p in index.values() if p.relative_to(TIMIT).parts[0]=='train'):
        lines=[line.split() for line in phn.read_text().splitlines()]
        mapping_path=IFMDD/'data/phones.60-48-39.map'
        mapping={fields[0]:fields[-1] if len(fields)>1 else None for fields in (line.split() for line in mapping_path.read_text().splitlines())}
        mapping.update(dx='d')
        rows=[{'phoneme':mapping.get(p,p),'start':int(s)/16000,'end':int(e)/16000} for s,e,p in lines if mapping.get(p,p) is not None]
        speaker,sent=phn.parent.name.upper(),phn.stem.upper()
        yield {'wav':str(phn.with_suffix('.wav')),'duration':int(lines[-1][1])/16000,
               'wrd':' '.join(phn.with_suffix('.txt').read_text().split()[2:]),
               'canonical_aligned':' '.join(r['phoneme'] for r in rows),'phonemes':rows,
               'raw_id':(speaker,sent)}


def word_phones(word, lexicon):
    word = word.lower().replace('’', "'").replace('‘', "'")
    if word in lexicon:
        return lexicon[word]
    if word.strip("'") in lexicon:
        return lexicon[word.strip("'")]
    if '-' in word:
        chunks = word.split('-')
        if all(c in lexicon for c in chunks):
            return [p for c in chunks for p in lexicon[c]]
    return None


def parse_encoder(path):
    return {ast.literal_eval(s.split(' => ')[0]):int(s.split(' => ')[1])
            for s in path.read_text().split('================')[0].splitlines() if ' => ' in s}


def merge(rows, mapping):
    result = []
    for p,start,end in rows:
        p = mapping.get(p,p)
        if result and result[-1][0] == p:
            result[-1] = (p,result[-1][1],end)
        else:
            result.append((p,start,end))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['libri100','timit'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,default=ROOT/'reproduction/multihead_libri100/assets/english_us_mfa_v3.1.0.dict')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lexicon,mfa_phones=dictionary(args.dictionary)
    fusion=ast.literal_eval(next(n.value for n in ast.parse((ROOT/'ottc/tools/dataset_loader/timit.py').read_text()).body
                               if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='timit_phoneme_fusion_map' for t in n.targets)))
    vocabs={
        'ottc_arpabet':json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/vocab_used.json').read_text()),
        'ifmdd_arpabet':parse_encoder(IFMDD/'data/librispeech_context_phone/monophone/label_encoder.txt'),
        'mfa_ipa':{'<blank>':0, **{p:i+1 for i,p in enumerate(sorted(mfa_phones|{'sil'}))}},
        'zipa_ipa':{s.split()[0]:int(s.split()[1]) for s in Path('/work/gm64/m64000/zipa/ipa_simplified/tokens.txt').read_text().splitlines()},
    }
    (args.output/'vocabs.json').write_text(json.dumps(vocabs,ensure_ascii=False,indent=2)+'\n')
    metadata={'dataset':args.dataset,'dictionary':str(args.dictionary),'dictionary_sha256':hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
              'mfa_target_source':'Official MFA US IPA v3.1 probability dictionary; most probable pronunciation; no acoustic variant selection',
              'mfa_intervals_available':False,'zipa_target_source':'IPA-Pack++ supervision.text via existing IF-MDD IPA127 manifest' if args.dataset=='libri100' else 'Explicit ARPAbet→IPA mapping',
              'same_audio_encoder_all_heads':True,'oov_policy':'Keep audio and other three heads; mask MFA loss for dictionary-incomplete utterances; preserve list',
              'original_ottc_head_note':'Original fused TIMIT inventory and mapping; original character-ASR LibriSpeech recipe is a different task',
              'head_vocab_sizes':{h:len(v) for h,v in vocabs.items()},'splits':{}}
    if args.dataset=='libri100':
        files={s:IFMDD/f'data/librispeech100_ipapack127_json/{s}.json' for s in ['train-train','train-dev','test']}
    else:
        files={s:IFMDD/f'data/timit_alignments_39phone_unrounded/json/{f}.json' for s,f in [('train-train','train'),('train-dev','dev'),('test','test')]}
    phn_index={(p.parent.name.upper(),p.stem.upper()):p for p in TIMIT.rglob('*.phn')} if args.dataset=='timit' else {}
    mapping_tree=ast.parse((IFMDD/'zipa_try/eval_zipa_cmu_monophone.py').read_text())
    ipa_map=ast.literal_eval(next(n.value for n in mapping_tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id',None)=='ARPABET_TO_IPA_BASE'))
    ipa_map['ah']=('ə',)
    for split,path in files.items():
        stats=Counter();counts={h:Counter() for h in HEADS};oov=Counter()
        with (args.output/f'{split}.jsonl').open('w') as output, (args.output/f'{split}_excluded.jsonl').open('w') as excluded:
            for e in timit_entries(split,path,phn_index) if args.dataset=='timit' else entries(path):
                stats['source_utterances']+=1
                text=e.get('wrd',e.get('text',''))
                words=re.findall(r"[\w]+(?:['’\-][\w]+)*",text.lower())
                pronunciations=[word_phones(w,lexicon) for w in words]
                missing=[w for w,p in zip(words,pronunciations) if p is None]
                if missing or not words:
                    oov.update(missing)
                    stats['masked_mfa_dictionary_oov']+=1
                    excluded.write(json.dumps({'wav':e['wav'],'words':missing,'text':text})+'\n')
                rows=[(p['phoneme'],float(p['start']),float(p['end'])) for p in e['phonemes']]
                arp=e.get('canonical_aligned','').split()
                if args.dataset=='libri100':
                    # The IPA127 manifest preserved original ARPAbet interval labels.
                    arp=[p for p,_,_ in rows]
                    if not arp or arp[0]!='sil':arp.insert(0,'sil')
                    if arp[-1]!='sil':arp.append('sil')
                else:
                    speaker,sent=e['raw_id'] if 'raw_id' in e else Path(e['wav']).stem.upper().split('_',1)
                    raw=[]
                    for line in phn_index[speaker,sent].read_text().splitlines():
                        start,end,p=line.split();raw.append((p,int(start)/16000,int(end)/16000))
                original=merge(raw if args.dataset=='timit' else [(p,0,0) for p in arp],fusion)
                mfa=[] if missing or not words else ['sil']+[p for seq in pronunciations for p in seq]+['sil']
                zipa=e['ipa_pieces'] if args.dataset=='libri100' else [p for ph in arp if ph!='sil' for p in ipa_map[ph]]
                phones={'ottc_arpabet':[p for p,_,_ in original],'ifmdd_arpabet':arp,'mfa_ipa':mfa,'zipa_ipa':zipa}
                if any(p not in vocabs[h] or p in {'<unk>','<sos/eos>','<bos>','<eos>'} for h,ps in phones.items() for p in ps):
                    stats['excluded_unknown_token']+=1
                    excluded.write(json.dumps({'wav':e['wav'],'reason':'unknown/special token'})+'\n')
                    continue
                targets={h:[vocabs[h][p] for p in ps] for h,ps in phones.items()}
                utt='_'.join(e['raw_id']) if 'raw_id' in e else Path(e['wav']).stem
                record={'utt':utt,'wav':e['wav'],'duration':e['duration'],'text':text,'targets':targets,
                        'references':{'ifmdd_arpabet':rows,'ottc_arpabet':original if args.dataset=='timit' else merge(rows,fusion)}}
                output.write(json.dumps(record,ensure_ascii=False)+'\n')
                stats['kept_utterances']+=1
                for h,ps in phones.items():counts[h].update(ps)
        metadata['splits'][split]={'source':str(path),**dict(stats),'oov_words':dict(oov.most_common()),'target_tokens':{h:sum(c.values()) for h,c in counts.items()},
                                    'phone_counts':{h:dict(c) for h,c in counts.items()}}
        print(split,dict(stats),'top OOV',oov.most_common(8),flush=True)
    (args.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
