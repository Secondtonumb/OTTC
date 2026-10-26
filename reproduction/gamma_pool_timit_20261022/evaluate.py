"""Matched frozen-checkpoint TIMIT recognition: greedy vs gamma pooling."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import time

from speechbrain.utils.metric_stats import ErrorRateStats
import torch

from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, HERE as TRAINING, build_model, load_data, Collator, phones_from_ids,
    normalize, sha, write, scorer,
)
from ottc.utils.gamma_decode import gamma_decode, greedy_token_count

HERE = Path(__file__).resolve().parent
MODES = ("greedy", "gamma_greedy_length", "gamma_reference_length", "gamma_posterior_greedy_length", "gamma_posterior_reference_length")


def speechbrain_stats(refs, hyps, ids=None):
    stats = ErrorRateStats()
    stats.append(ids=ids if ids is not None else [str(i) for i in range(len(refs))],
                 target=[r.split() for r in refs], predict=[h.split() for h in hyps])
    return stats


def score(refs, hyps):
    summary = speechbrain_stats(refs, hyps).summarize()
    return dict(utterances=len(refs), reference_phones=summary['num_scored_tokens'],
                errors=summary['num_edits'], substitutions=summary['substitutions'],
                deletions=summary['deletions'], insertions=summary['insertions'],
                PER_percent=summary['WER'])


def report(path, records, mode, totals):
    refs = [r['reference'] for r in records]
    hyps = [r['hypotheses'][mode] for r in records]
    stats = speechbrain_stats(refs, hyps, [r['utt'] for r in records])
    summary = stats.summarize()
    assert summary['num_edits'] == totals['errors']
    assert summary['insertions'] == totals['insertions']
    assert summary['deletions'] == totals['deletions']
    assert summary['substitutions'] == totals['substitutions']
    with path.open('w', encoding='utf-8') as stream:
        # SpeechBrain supplies both edit alignment and column padding, exactly
        # as ErrorRateStats.write_stats(), rather than a semicolon-only imitation.
        stats.write_stats(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--costs', nargs='+', choices=['jsd_upstream', 'ce'], default=['jsd_upstream', 'ce'])
    parser.add_argument('--limit', type=int, default=0, help='0 means full dev400/test192')
    parser.add_argument('--output', type=Path, default=HERE / 'results_aggregation_comparison')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device == 'cuda':
        # Match historical standalone FP32 and PyTorch's default backend flags.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = [Path(__file__), HERE/'test_gamma_decode.py', ROOT/'ottc/utils/gamma_decode.py',
               ROOT/'ottc/utils/eval_metrics.py', ROOT/'run_exp.py', TRAINING/'common.py',
               ROOT/'reproduction/native_stride_ottc_20261018/common.py',
               ROOT/'ottc/models/sequence/wav2vec2_ottc.py']
    provenance = dict(device=args.device, scorer='SpeechBrain ErrorRateStats; independent frozen PER verification',
        precision='FP32 batch1; AMP disabled; matmul TF32 false; cudnn TF32 true (historical defaults)',
        torch=torch.__version__, python=platform.python_version(),
        GPU=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        source_sha256={str(p):sha(p) for p in sources}, PBS_JOBID=os.environ.get('PBS_JOBID'),
        protocol='same frozen TIMIT manifests/vocab/normalize; remove sil/q, map dx to d; micro-averaged PER',
        decoder='uniform beta; same fractional CDF overlap and column-normalized gamma for logits and posterior averaging; one output per position; no CTC collapse',
        length_modes={'greedy':'frame argmax then historical run collapse',
            'gamma_greedy_length':'number of non-pad/non-unk greedy runs, including silence; no reference input',
            'gamma_reference_length':'oracle number of training target positions, including silence; no reference identities/times used',
            'gamma_posterior_greedy_length':'same greedy M; weighted arithmetic posterior average',
            'gamma_posterior_reference_length':'same oracle M; weighted arithmetic posterior average'},
        optimizer_updated=False, uploads=False, checkpoints={}, limit=args.limit)
    results = {}
    parity = {}
    helpers = scorer()
    for cost in args.costs:
        name = f'wav2vec2__timit__10ms__mono__{cost}'
        model_dir = TRAINING/'models'/name
        config = json.loads((model_dir/'run_config.json').read_text())
        assert config['head'] == 'mono' and config['cost'] == cost
        selection = json.loads((model_dir/'DEV_SELECTION.json').read_text())
        assert selection['selected'] == 'best_dev' and not selection['test_used']
        checkpoint = model_dir/'best_dev.pt'
        data, rows, vocab, processor, data_info = load_data(config)
        del data
        model, topo = build_model(config, vocab, 73)
        assert topo is None
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert saved['recipe'] == name
        model.load_state_dict(saved['model'], strict=True)
        model.step = saved['step']
        provenance['checkpoints'][cost] = dict(path=str(checkpoint), sha256=sha(checkpoint),
            epoch=saved['epoch'], step=saved['step'], training_config_sha256=sha(model_dir/'run_config.json'),
            data=data_info, prior_dev_selection=selection)
        del saved
        model.to(args.device).eval()
        collator = Collator(processor)
        results[cost] = {}
        for split in ('dev', 'test'):
            selected_rows = rows[split][:args.limit] if args.limit else rows[split]
            records = []
            timer = time.monotonic()
            with torch.inference_mode():
                for i, row in enumerate(selected_rows):
                    batch = collator([dict(wav=row['wav'], labels=[])])
                    batch.pop('labels')
                    out = model(**{k:v.to(args.device) for k,v in batch.items()})
                    mask = out.hidden_states[1][0] > 0
                    logits = out.logits[0][mask].float()
                    alpha = out.hidden_states[0][0][mask].float()
                    frame_ids = logits.argmax(-1).tolist()
                    m = greedy_token_count(frame_ids, (0, 41))
                    # Read only the target count for the separately labeled oracle arm.
                    oracle_m = len(row['training_target_ids'])
                    pooled = gamma_decode(logits, alpha, m).tolist()
                    oracle = gamma_decode(logits, alpha, oracle_m).tolist()
                    pooled_p = gamma_decode(logits, alpha, m, aggregation='posterior').tolist()
                    oracle_p = gamma_decode(logits, alpha, oracle_m, aggregation='posterior').tolist()
                    hyps = {'greedy': normalize(phones_from_ids(frame_ids, vocab)),
                            'gamma_greedy_length': normalize(phones_from_ids(pooled, vocab, collapse=False)),
                            'gamma_reference_length': normalize(phones_from_ids(oracle, vocab, collapse=False)),
                            'gamma_posterior_greedy_length': normalize(phones_from_ids(pooled_p, vocab, collapse=False)),
                            'gamma_posterior_reference_length': normalize(phones_from_ids(oracle_p, vocab, collapse=False))}
                    records.append(dict(utt=row['utt'], reference=normalize([p for p,_,_ in row['reference']]),
                        hypotheses=hyps, frames=logits.shape[0], greedy_length=m, reference_length=oracle_m,
                        gamma_greedy_ids=pooled, gamma_reference_ids=oracle,
                        gamma_posterior_greedy_ids=pooled_p, gamma_posterior_reference_ids=oracle_p))
                    if (i+1) % 100 == 0:
                        print(json.dumps(dict(cost=cost, split=split, processed=i+1)), flush=True)
            refs = [r['reference'] for r in records]
            results[cost][split] = {}
            for mode in MODES:
                hyps = [r['hypotheses'][mode] for r in records]
                totals = score(refs, hyps)
                # Independently verify the frozen historical PER implementation.
                assert abs(totals['PER_percent'] - helpers['phone_per'](refs, hyps)) < 1e-10
                totals['reference_length_used'] = mode.endswith('reference_length')
                results[cost][split][mode] = totals
                report(output/f'{cost}_{split}_{mode}.wer.txt', records, mode, totals)
            # Require exact utterance transcripts, not just aggregate baseline parity.
            historical = model_dir/f'benchmark_best_dev_{split}.predictions.jsonl'
            prior_records = {r['utt']:r for r in (json.loads(line) for line in historical.open())}
            for record in records:
                assert record['reference'] == normalize([p for p,_,_ in prior_records[record['utt']]['reference']])
            mismatches = [dict(utt=r['utt'], historical=normalize(prior_records[r['utt']]['text_free']['hypothesis']),
                               current=r['hypotheses']['greedy']) for r in records
                          if normalize(prior_records[r['utt']]['text_free']['hypothesis']) != r['hypotheses']['greedy']]
            parity[f'{cost}/{split}'] = dict(utterances=len(records), mismatches=mismatches,
                historical_sha256=sha(historical), current_PER=results[cost][split]['greedy']['PER_percent'],
                historical_PER=score(refs, [normalize(prior_records[r['utt']]['text_free']['hypothesis']) for r in records])['PER_percent'])
            write(output/'HISTORICAL_BASELINE_CHECK.json', parity)
            # All decoder comparisons above share the exact same new forward.
            # Report historical numeric drift explicitly rather than hiding it.
            if mismatches:
                print(json.dumps(dict(historical_baseline_drift=parity[f'{cost}/{split}'])), flush=True)
            with (output/f'{cost}_{split}.predictions.jsonl').open('w') as stream:
                for record in records:
                    stream.write(json.dumps(record) + '\n')
            write(output/'PROGRESS.json', dict(cost=cost, split=split, utterances=len(records),
                historical_greedy_mismatches=len(mismatches), seconds=time.monotonic()-timer))
            write(output/'RESULTS.json', results)
            print(json.dumps(dict(cost=cost, split=split, results=results[cost][split])), flush=True)
        del model
        gc.collect()
        if args.device == 'cuda': torch.cuda.empty_cache()
    write(output/'PROVENANCE.json', provenance)
    with (output/'RESULTS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cost','split','decoder',*results[args.costs[0]]['dev']['greedy']])
        writer.writeheader()
        for cost, splits in results.items():
            for split, modes in splits.items():
                for mode, values in modes.items(): writer.writerow(dict(cost=cost, split=split, decoder=mode, **values))
    write(output/'COMPLETE.json', dict(complete=True, full_sets=args.limit == 0,
        costs=args.costs, historical_greedy_transcripts_match=all(not p['mismatches'] for p in parity.values()),
        seconds=time.monotonic()-started, results_sha256=sha(output/'RESULTS.json'),
        provenance_sha256=sha(output/'PROVENANCE.json')))


if __name__ == '__main__':
    main()
