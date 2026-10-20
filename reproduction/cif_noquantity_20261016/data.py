"""Audited phone-occurrence supervision for CIF without quantity loss."""
import json
from pathlib import Path

from reproduction.cif_timit_20261016.data import ROOT, VOCAB, collate, read_audio, sha


def _read(dataset):
    vocab = json.loads(VOCAB.read_text())
    rows, hashes, speakers, seen, paths = {}, {}, {}, {}, {}
    if dataset == 'buckeye':
        directory = ROOT/'reproduction/buckeye_singlehead_20261014/data'
        files = [('train','train',10927),('dev','dev',1524),('test','test',1840)]
    else:
        directory = ROOT/'reproduction/multihead_libri100/data'
        files = [('train','train-train',28538),('dev','train-dev',5567),('test','test',5558)]
    for split, name, expected in files:
        path = directory/(name+'.jsonl')
        hashes[split] = sha(path)
        output=[]; speakers[split]=set(); seen[split]=set(); paths[split]=set()
        with path.open() as stream:
            for line in stream:
                x=json.loads(line)
                reference=x['reference'] if dataset=='buckeye' else x['references']['ottc_arpabet']
                ids=[vocab[p] for p,_,_ in reference]
                assert ids and not set(ids)&{0,41}, x['utt']
                if dataset=='libri100':
                    assert ids==x['targets']['ottc_arpabet'], x['utt']
                samples=x.get('samples',round(x['duration']*16000))
                frames=(samples-400)//320+1
                assert len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))<=frames, x['utt']
                assert all(0<=s<=e<=samples/16000+.001 and (s<e or (p=='sil' and (s==0 or abs(e-samples/16000)<.001))) for p,s,e in reference), x['utt']
                assert x['utt'] not in seen[split]
                seen[split].add(x['utt']);speakers[split].add(x['speaker']);paths[split].add(x['wav'])
                output.append(dict(utt=x['utt'],wav=x['wav'],speaker=x['speaker'],duration=samples/16000,
                    samples=samples,subset=x.get('subset',split),
                    references={'ottc_arpabet':reference},targets={'ottc_arpabet':ids}))
        assert len(output)==expected,(dataset,split,len(output))
        rows[split]=output
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not speakers[a]&speakers[b] and not seen[a]&seen[b] and not paths[a]&paths[b]
    return rows,vocab,dict(manifests_sha256=hashes,vocab_sha256=sha(VOCAB),
        shared_data_sha256=sha(ROOT/'reproduction/cif_timit_20261016/data.py'),
        corpus='Buckeye original-human partial19' if dataset=='buckeye' else 'LibriSpeech train-clean-100 automatic phone alignments; aligner identity unverified',
        target_protocol='phone occurrences; Buckeye retains repeated human phones and no artificial edge sil; Libri reuses canonical ottc_arpabet occurrences including source sil; zero-duration edge sil retained in targets and excluded from boundary scoring')


def load_rows(dataset, train_limit=0, eval_limit=0):
    rows,vocab,contract=_read(dataset)
    if train_limit: rows['train']=rows['train'][:train_limit]
    if eval_limit:
        rows['dev']=rows['dev'][:eval_limit];rows['test']=rows['test'][:eval_limit]
    return rows,vocab,contract


def benchmark_rows(dataset, rows):
    # Same benchmark references for both training corpora. These are never used
    # to select a checkpoint or tune boundaries across corpora.
    other='libri100' if dataset=='buckeye' else 'buckeye'
    heldout,_,_=_read(other)
    libri=rows if dataset=='libri100' else heldout
    buckeye=rows if dataset=='buckeye' else heldout
    result={'buckeye_dev':buckeye['dev'],'buckeye_test':buckeye['test']}
    for split in ['dev','test']:
        for subset in ['clean','other']:
            name=split+'_'+subset
            result['libri_'+name]=[r for r in libri[split] if r['subset']==split+'-'+subset]
    # Own full-set dev/test already evaluated by train.py; named subsets are
    # additionally retained to make clean/other and cross-corpus comparisons explicit.
    return result
