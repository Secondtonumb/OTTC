"""Audio-only recognition and segmentation of the final README checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bootstrap
from bootstrap import ROOT
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT)]
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import Wav2Vec2Processor
from model import WavLMForOTTC as wvForOTTC
from segmentation_metrics import r_value_percent
from recipe import suffix as ce_suffix
from ottc.models.sequence.wav2vec2_ctc import wvForCTC
from ottc.tools.dataset_loader.timit import merge_intervals, timit_phoneme_fusion_map
from run_exp import latest_checkpoint
from ottc.utils.wandb_tracking import publish_benchmark

SILENCE = {'sil', '<sil>', 'sp', 'spn', 'pau', 'h#', 'epi',
           'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl'}
PHONE_MAP = dict(timit_phoneme_fusion_map, dx='d')


def normalize(sequence):
    return [PHONE_MAP.get(p.lower().rstrip('012'), p.lower().rstrip('012'))
            for p in sequence if p.lower().rstrip('012') not in SILENCE | {'q'}]


def edit_alignment(ref, hyp):
    """Levenshtein edits and matching token index pairs, including repeats."""
    dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref)+1)
    dp[0, :] = np.arange(len(hyp)+1)
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+(a != b))
    i, j = len(ref), len(hyp)
    pairs, counts = [], Counter()
    while i or j:
        if i and j and dp[i, j] == dp[i-1, j-1]+(ref[i-1] != hyp[j-1]):
            if ref[i-1] == hyp[j-1]:
                pairs.append((i-1, j-1))
            else:
                counts['substitutions'] += 1
            i -= 1
            j -= 1
        elif i and dp[i, j] == dp[i-1, j]+1:
            counts['deletions'] += 1
            i -= 1
        else:
            counts['insertions'] += 1
            j -= 1
    return int(dp[-1, -1]), pairs[::-1], dict(counts)


def matches(ref, pred, tolerance):
    i = j = n = 0
    while i < len(ref) and j < len(pred):
        if abs(ref[i]-pred[j]) <= tolerance+1e-9:
            n += 1
            i += 1
            j += 1
        elif pred[j] < ref[i]:
            j += 1
        else:
            i += 1
    return n


def prf(counts):
    match, ref, pred = counts
    p = match/pred if pred else 0
    r = match/ref if ref else 0
    os = pred/ref-1 if ref else 0
    return dict(matches=match, reference=ref, predicted=pred, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0,
                r_value=r_value_percent(100*p,100*r)/100,
                over_segmentation=os)


def original_timit():
    speakers = {'mdab0','mwbt0','felc0','mtas1','mwew0','fpas0','mjmp0','mlnt0',
                'fpkt0','mlll0','mtls0','fjlm0','mbpm0','mklt0','fnlp0','mcmj0',
                'mjdh0','fmgd0','mgrt0','mnjm0','fdhc0','mjln0','mpam0','fmld0'}
    rows = []
    for wav in sorted(Path('/work/gm64/m64000/dataset/TIMIT/timit/test').rglob('*.wav')):
        if wav.parent.name not in speakers:
            continue
        entries = [x.split() for x in wav.with_suffix('.phn').read_text().splitlines()]
        phones, starts, ends = merge_intervals([x[2] for x in entries],
            [int(x[0])/16000 for x in entries], [int(x[1])/16000 for x in entries],
            timit_phoneme_fusion_map)
        rows.append((str(wav.relative_to(wav.parents[3])), dict(wav=str(wav),
                     phonemes=[dict(phoneme=p, start=s, end=e)
                               for p, s, e in zip(phones, starts, ends)])))
    assert len(rows) == 240
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', choices=['ctc', 'ottc', 'softdtw', 'softdtweuc', 'softdtw_1', 'softdtw_0.01'], required=True)
    parser.add_argument('--bins-metric', choices=['crossentropy', 'jsd', 'jsd_standard', 'euclidian'], default='crossentropy')
    parser.add_argument('--cross-domain', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size',type=int,choices=[128,256],required=True)
    args = parser.parse_args()
    if args.loss!='ottc' or args.bins_metric!='crossentropy':parser.error('This entry evaluates the OT-weighted CE experiment')
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    suffix = ('smoke_' if args.smoke else 'runexp_')+args.loss+'_fuse_phoneme'
    if args.bins_metric != 'crossentropy':
        if args.loss != 'ottc':
            parser.error('--bins-metric applies only to OTTC')
        suffix += '_' + args.bins_metric
    suffix=ce_suffix(args.batch_size)
    model_dir = ROOT/'reproduction/models/large_models_results/wavlm-large'/suffix
    completed=ROOT/'reproduction/results'/suffix
    if (completed/'EVAL_COMPLETE.json').exists():
        required={'readme_kaldi_test_240','timit_core_test_192','timit_development_400'}
        if args.cross_domain:required.update({'buckeye_test','l2arctic_test'})
        if required<=json.loads((completed/'metrics.json').read_text())['splits'].keys():
            print('Already evaluated:',completed/'metrics.json');return
    checkpoint = latest_checkpoint(model_dir)
    state = json.loads((checkpoint/'trainer_state.json').read_text())
    if not args.smoke:
        assert state['epoch'] == 40, state['epoch']
    model, info = (wvForOTTC if args.loss == 'ottc' else wvForCTC).from_pretrained(
        checkpoint, output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
    if args.loss == 'ottc':
        assert model.config.bins_metric == args.bins_metric
    else:
        assert model.config.loss_type == args.loss
    model = model.eval().cuda()
    model.step = int(checkpoint.name[11:])
    processor = Wav2Vec2Processor.from_pretrained(model_dir/'processor')
    cases = {'readme_kaldi_test_240': (None, original_timit())}
    data = Path('/work/gm64/m64000/IF-MDD/data')
    manifests = {'timit_core_test_192': data/'timit_alignments_39phone_unrounded/json/test.json',
                 'timit_development_400': data/'timit_alignments_39phone_unrounded/json/dev.json'}
    if args.cross_domain:
        manifests.update(buckeye_test=data/'buckeye_raw_alignments/json/test.json',
                         l2arctic_test=data/'test_times.json')
    for name, path in manifests.items():
        cases[name] = (path, sorted(json.loads(path.read_text()).items()))
    output = ROOT/'reproduction/results'/suffix
    results = dict(checkpoint=str(checkpoint), epoch=state['epoch'], global_step=state['global_step'],
                   loss_type=args.loss,
                   softdtw=dict(gamma=model.softdtw.gamma, cost=model.softdtw.cost,
                                normalize=model.softdtw.normalize) if args.loss.startswith('softdtw') else None,
                   bins_metric=args.bins_metric if args.loss == 'ottc' else None,
                   checkpoint_loading_info=info,
                   evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   rvalue_source_sha256=hashlib.sha256((ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py').read_bytes()).hexdigest(),
                   audio_only=True, raw_segmentation='argmax class transitions including blank; no time shift',
                   interval_metric='duration overlap of correctly recognized DP-matched tokens / reference duration',
                   boundary_contract='all interval endpoints, excluding 20ms utterance edges; one-to-one matching',
                   splits={})
    begin = time.monotonic()
    examples = []
    with torch.inference_mode():
        for name, (manifest, rows) in cases.items():
            if args.smoke:
                rows = rows[:4]
            total = Counter()
            boundary = {str(ms): [0, 0, 0] for ms in [10, 20, 30]}
            ref_duration = overlap = error_time = 0.0
            pairs_n = 0
            with (output/(name+'.jsonl')).open('w') as handle:
                for key, row in rows:
                    wav = row['wav'].replace('/home/m64000/work/', '/work/gm64/m64000/')
                    audio, sr = sf.read(wav, dtype='float32')
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != 16000:
                        common = math.gcd(sr, 16000)
                        audio = resample_poly(audio, 16000//common, sr//common)
                        sr = 16000
                    duration = len(audio)/sr
                    values = processor(audio, sampling_rate=sr, return_tensors='pt',
                                       return_attention_mask=True).to('cuda')
                    out = model(**values)  # No reference labels are supplied.
                    ids = out.logits[0].argmax(-1).cpu().numpy()
                    if args.loss == 'ottc':
                        alpha = out.hidden_states[0].cpu().numpy().reshape(-1)
                        ids[alpha == 0] = processor.tokenizer.pad_token_id
                        ids[ids == processor.tokenizer.unk_token_id] = processor.tokenizer.pad_token_id
                    raw_hyp = processor.decode(ids).split()
                    changes = np.flatnonzero(ids[1:] != ids[:-1])+1
                    starts, ends = np.r_[0, changes], np.r_[changes, len(ids)]
                    spans = [(processor.tokenizer.convert_ids_to_tokens(int(ids[s])), float(s)*.02,
                              min(float(e)*.02, duration)) for s, e in zip(starts, ends)
                             if ids[s] != processor.tokenizer.pad_token_id]
                    if 'phonemes' in row:
                        refs = [(x['phoneme'], float(x['start']), float(x['end'])) for x in row['phonemes']]
                    else:
                        phones = row['perceived_train_target'].split()
                        refs = list(zip(phones, row['target_starts'], row['target_ends']))
                    ref = [p for p, _, _ in refs]
                    err, _, counts = edit_alignment(ref, raw_hyp)
                    total['raw_errors'] += err
                    total['raw_phones'] += len(ref)
                    total.update({'raw_'+k: v for k, v in counts.items()})
                    normalized_ref, normalized_hyp = normalize(ref), normalize(raw_hyp)
                    nerr, _, ncounts = edit_alignment(normalized_ref, normalized_hyp)
                    total['nosil_errors'] += nerr
                    total['nosil_phones'] += len(normalized_ref)
                    total.update({'nosil_'+k: v for k, v in ncounts.items()})
                    rb = sorted({float(t) for _, s, e in refs for t in [s, e] if .02 < t < duration-.02})
                    pb = [int(t)*.02 for t in changes if .02 < int(t)*.02 < duration-.02]
                    for ms in boundary:
                        c = boundary[ms]
                        c[0] += matches(rb, pb, int(ms)/1000)
                        c[1] += len(rb)
                        c[2] += len(pb)
                    _, pairs, _ = edit_alignment(ref, [p for p, _, _ in spans])
                    ref_duration += sum(e-s for _, s, e in refs)
                    for i, j in pairs:
                        _, rs, re = refs[i]
                        _, ps, pe = spans[j]
                        overlap += max(0, min(re, pe)-max(rs, ps))
                        error_time += abs(rs-ps)+abs(re-pe)
                    pairs_n += len(pairs)
                    handle.write(json.dumps(dict(utt=key, wav=wav, reference=refs, hypothesis=raw_hyp,
                         predicted_spans=spans, reference_boundaries=rb, predicted_boundaries=pb,
                         raw_errors=err, normalized_errors=nerr))+'\n')
                    if name == 'readme_kaldi_test_240' and len(examples) < 3:
                        examples.append(dict(utt=key, logits=out.logits[0].float().cpu().numpy(),
                                             predicted_ids=ids, reference=refs))
            metrics = dict(utterances=len(rows), counts=dict(total),
                  raw_PER=total['raw_errors']/total['raw_phones'],
                  normalized_nosil_PER=total['nosil_errors']/total['nosil_phones'],
                  boundaries={ms: prf(c) for ms, c in boundary.items()},
                  matched_interval_overlap_ratio=overlap/ref_duration,
                  matched_token_TSE_ms=error_time/pairs_n*1000 if pairs_n else None)
            if manifest:
                metrics.update(manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            results['splits'][name] = metrics
            print(name, json.dumps(metrics), flush=True)
    results['elapsed_seconds'] = time.monotonic()-begin
    results['trainer_eval'] = json.loads((model_dir/'eval_results.json').read_text())
    (output/'metrics.json').write_text(json.dumps(results, indent=2))
    np.savez_compressed(output/'examples.npz', **{
        f'{i}_{k}': np.asarray(v) for i, ex in enumerate(examples) for k, v in ex.items()})
    (output/'EVAL_COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(checkpoint))))
    publish_benchmark(results, model_dir)


if __name__ == '__main__':
    main()
