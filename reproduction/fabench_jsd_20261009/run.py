"""Word-input FA-Bench evaluation of the completed Libri100 four-head JSD.

Inference reads jobs only. Reference phones and times enter the scorer only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path('/work/gm64/m64000/fa-bench')
OUT = Path(__file__).resolve().parent
MODEL = ROOT / 'reproduction/multihead_libri100/models/jsd_upstream'
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT/'reproduction'), str(ROOT),
                str(BENCH/'.eval-deps310'), str(BENCH), str(BENCH/'analysis')]
import numpy as np
import soundfile as sf
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from multihead import MultiPhoneModel
from ifmdd_k2_alignment import load_backend, align
from alignment_decoders import native_ot_spans

HEADS = ('ottc_arpabet', 'ifmdd_arpabet')
CELLS = {
    'timit': BENCH/'data/ottc_wordinput_20261009',
    'buckeye_processed': BENCH/'data/ottc_buckeye_processed_20261009',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def targets(job, vocab):
    # Match the 39-phone training inventory, independent of reference phones.
    # Keep each G2P occurrence, including repeats, to preserve word ownership.
    fold = {'ao': 'aa', 'zh': 'sh'}
    groups = [[fold.get(p, p) for p in g] for g in job['g2p_groups']]
    phones = ['sil'] + [p for g in groups for p in g] + ['sil']
    assert all(p in vocab and vocab[p] != 0 for p in phones), phones
    assert all(groups) and len(groups) == len(job['g2p_words'])
    return phones, [vocab[p] for p in phones], groups


def intervals(spans, phones, groups, words):
    assert len(spans) == len(phones)
    ph = [dict(label=p, start=float(s), end=float(e)) for p, (_, s, e) in zip(phones, spans)]
    wd, cursor = [], 1
    for word, group in zip(words, groups):
        selected = ph[cursor:cursor+len(group)]
        wd.append(dict(label=word, start=selected[0]['start'], end=selected[-1]['end']))
        cursor += len(group)
    assert cursor == len(ph)-1
    assert all(np.isfinite([p['start'],p['end']]).all() and 0 <= p['start'] <= p['end'] for p in ph)
    return ph, wd


def infer(cell, smoke=False):
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU required'
    complete = json.loads((MODEL/'TRAIN_COMPLETE.json').read_text())
    assert complete['epochs'] == 40 and not complete['smoke']
    jobs_path = CELLS[cell]/'jobs.jsonl'
    jobs = [json.loads(x) for x in jobs_path.read_text().splitlines()]
    assert len(jobs) == len({j['utt_id'] for j in jobs})
    assert all(not set(j) & {'phones', 'references', 'targets', 'gold', 'words'} for j in jobs)
    if smoke:
        # Includes a long and an adjacent-repeated-phone example.
        chosen = [jobs[0], max(jobs,key=lambda j:j['duration_s'])]
        repeated = next((j for j in jobs if any(a == b for a,b in zip(
            [p for g in j['g2p_groups'] for p in g], [p for g in j['g2p_groups'] for p in g][1:]))), jobs[1])
        chosen.append(repeated)
        jobs = list({j['utt_id']:j for j in chosen}.values())
    output = OUT/cell/('smoke' if smoke else 'full')
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha = digest(MODEL/'final/pytorch_model.bin')
    signature = dict(checkpoint_sha256=checkpoint_sha, jobs_sha256=digest(jobs_path),
                     evaluator_sha256=digest(__file__), smoke=smoke)
    if (output/'INFERENCE_COMPLETE.json').exists():
        saved = json.loads((output/'INFERENCE_COMPLETE.json').read_text())
        assert saved['signature'] == signature and saved['n_jobs'] == len(jobs)
        print('Already complete',cell,smoke,flush=True)
        return
    if (output/'provenance.json').exists():
        assert json.loads((output/'provenance.json').read_text())['signature'] == signature
    model, info = MultiPhoneModel.from_pretrained(MODEL/'final', output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), info
    model = model.eval().cuda()
    assert model.alignment_encoder is not None and model.config.objective == 'jsd_upstream'
    backend, backend_prov = load_backend()
    (output/'k2_source_snapshot.py').write_text(backend_prov.pop('source_snapshot')+'\n')
    provenance = dict(signature=signature, checkpoint=str(MODEL/'final'), training_completion=complete,
        training_run=json.loads((MODEL/'wandb_run.json').read_text()), backend=backend_prov,
        heads=HEADS, decoders=['k2','native_ot'], input_contract='reference words -> fixed Charsiu G2P -> ao/zh fold -> ordered phone occurrences',
        acoustic_input='same utterance normalization as training; audio-only forward; BF16 autocast',
        special_tokens='leading/trailing sil, blank=0', gold_used_by_inference=False,
        native_ot_time='fractional inverse-CDF frame boundaries * 0.02; end clipped to audio duration',
        native_ot_scope='equal target mass per supplied G2P occurrence, including repeated phones',
        k2_time='original IF-MDD duration/frame conversion, blank_policy=previous',
        gpu=torch.cuda.get_device_name(), sources={str(p):digest(p) for p in [ROOT/'reproduction/multihead.py',
            ROOT/'reproduction/alignment_decoders.py', ROOT/'reproduction/ifmdd_k2_alignment.py']})
    write(output/'provenance.json',provenance)
    hyp = output/'hyp.jsonl'
    seen = set()
    if hyp.exists():
        for line in hyp.read_text().splitlines():
            row=json.loads(line); assert row['utt_id'] not in seen; seen.add(row['utt_id'])
    assert seen <= {j['utt_id'] for j in jobs}
    started = time.monotonic()
    with hyp.open('a') as f, torch.no_grad():
        for i,job in enumerate(jobs):
            if job['utt_id'] in seen:
                continue
            x,sr=sf.read(job['audio_path'],dtype='float32')
            assert sr == 16000 and x.ndim == 1 and abs(len(x)/sr-job['duration_s']) < 1/sr
            x=(x-x.mean())/np.sqrt(x.var()+1e-7)
            values=torch.from_numpy(x).unsqueeze(0).cuda()
            mask=torch.ones_like(values,dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(input_values=values,attention_mask=mask)
                mh=model.alignment_encoder(values,attention_mask=mask).last_hidden_state
            length=int(pred['lengths'][0]); record=dict(utt_id=job['utt_id'],split=job['split'],modes={})
            for head in HEADS:
                phones,ids,groups=targets(job,model.config.vocabs[head])
                logits=pred['logits'][head][0,:length].float()
                assert torch.isfinite(logits).all()
                aligned,path=align(logits.log_softmax(-1),ids,job['duration_s'],backend)
                k2=[(s['token_id'],s['start'],s['end']) for s in aligned]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    mass=model.mass_heads[head](mh)[0,:length,0].float().softmax(-1).cpu().numpy()
                ot=[(v,s*.02,min(e*.02,job['duration_s'])) for v,s,e in native_ot_spans(mass,ids)]
                for decoder,spans in [('k2',k2),('native_ot',ot)]:
                    ph,wd=intervals(spans,phones,groups,job['g2p_words'])
                    record['modes'][head+'_'+decoder]=dict(phones=ph,words=wd)
            f.write(json.dumps(record)+'\n');f.flush()
            if i%25 == 0:
                print(cell,i+1,'/',len(jobs),'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(hyp.read_text().splitlines()) == len(jobs)
    assert digest(MODEL/'final/pytorch_model.bin') == checkpoint_sha
    write(output/'INFERENCE_COMPLETE.json',dict(signature=signature,n_jobs=len(jobs),failures=0,
          smoke=smoke,elapsed_this_segment_s=time.monotonic()-started))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=CELLS,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();infer(args.cell,args.smoke)
