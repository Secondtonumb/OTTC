"""Matched, evaluation-only TIMIT transfer of existing Libri100/960 weights."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT/'reproduction'), str(ROOT)]
CONT = ROOT/'reproduction/libri960_continue5_20261017'
PARENTS = ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'
OBJECTIVES = {'jsd_upstream': 'jsd', 'ce': 'crossentropy', 'jsd_standard': 'jsd_standard'}
VARIANTS = ['libri100_epoch40', 'best_dev', 'final', 'average_last5']
SPLITS = {'dev400': ('train-dev.jsonl', 400), 'core_test192': ('test.jsonl', 192)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def prepare():
    source = ROOT/'reproduction/multihead_timit/data'
    old = ROOT/'reproduction/results/runexp_ottc_fuse_phoneme_jsd/timit_core_test_192.jsonl'
    historical = {Path(r['utt']).stem: r for r in map(json.loads, old.open())}
    vocab = json.loads((CONT/'models/jsd_upstream/vocab_used.json').read_text())
    report = dict(dataset='TIMIT dev400 and core-test192; no SA', optimizer_updated=False,
                  evaluation_only=True, source_sha256={str(old): digest(old)}, splits={})
    for split, (filename, expected) in SPLITS.items():
        path = source/filename
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == len({r['utt'] for r in rows}) == expected
        prepared = []
        for row in rows:
            reference = row['references']['ifmdd_arpabet']
            assert all(p in vocab and p != 'q' and 0 <= s < e for p, s, e in reference)
            if split == 'core_test192':
                prior = historical[row['utt']]
                assert prior['wav'] == row['wav'] and prior['reference'] == reference
            assert Path(row['wav']).is_file()
            prepared.append(dict(utt=row['utt'], wav=row['wav'], duration=row['duration'],
                                 reference=reference, loss_target_ids=row['targets']['ottc_arpabet']))
        target = HERE/'data'/(split+'.jsonl')
        target.parent.mkdir(parents=True, exist_ok=True)
        contents = ''.join(json.dumps(r)+'\n' for r in prepared)
        if target.exists():
            assert target.read_text() == contents, 'Frozen evaluation manifest changed'
        else:
            target.write_text(contents)
        report['source_sha256'][str(path)] = digest(path)
        report['splits'][split] = dict(utterances=expected, manifest_sha256=digest(target),
            reference='unrounded human PHN; closure as sil; q omitted preserving gaps; dx to d; sil retained')
    dev = [json.loads(l) for l in (HERE/'data/dev400.jsonl').open()]
    test = [json.loads(l) for l in (HERE/'data/core_test192.jsonl').open()]
    assert not {r['utt'].split('_')[0] for r in dev} & {r['utt'].split('_')[0] for r in test}
    save(HERE/'PREPARE_COMPLETE.json', report)
    return report


def checkpoint_for(objective, variant):
    directory = CONT/'models'/objective
    if variant == 'libri100_epoch40':
        checkpoint = PARENTS/f'runexp_libri100_singlehead_{objective}_20261014/checkpoint-17840'
        stage_epoch, stage_step = 0, 0
    elif variant == 'best_dev':
        info = json.loads((directory/'BEST_DEV.json').read_text())
        checkpoint = Path(info['checkpoint'])
        stage_epoch, stage_step = info['stage_epoch'], info['stage_step']
    elif variant == 'average_last5':
        info = json.loads((directory/'AVERAGE_LAST5.json').read_text())
        assert info['member_count'] == 5
        assert [m['stage_epoch'] for m in info['members']] == [1, 2, 3, 4, 5]
        checkpoint = Path(info['checkpoint'])
        stage_epoch, stage_step = 5, 21970
    else:
        info = json.loads((directory/'TRAIN_COMPLETE.json').read_text())
        assert info['epoch'] == 5 and info['global_step'] == 21970
        checkpoint = Path(info['checkpoint'])
        stage_epoch, stage_step = 5, 21970
    return directory, checkpoint, dict(stage_epoch=stage_epoch, stage_step=stage_step,
        parent_epoch=40, parent_step=17840, cumulative_optimizer_step=17840+stage_step,
        average_members=(info['members'] if variant == 'average_last5' else None),
        selected_by='fixed parent / final / last5; best-dev inherited from Libri development; no TIMIT selection')


def scoring_helpers():
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from segmentation_metrics import r_value_percent
    namespace = dict(np=np, Counter=Counter, Levenshtein=Levenshtein, r_value_percent=r_value_percent)
    for path, names in [(ROOT/'reproduction/libri100_singlehead_20261014/train.py',
                         ['phone_per', 'boundary_matches']),
                        (CONT/'evaluate.py', ['edges', 'new_counts', 'score', 'summary'])]:
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def evaluate(objective, variant, smoke):
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    transformers.utils.import_utils._peft_available = False
    transformers.utils.import_utils._bitsandbytes_available = False
    from transformers import Wav2Vec2Processor
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
    from ifmdd_k2_alignment import align, load_backend
    from segmentation_metrics import SOURCE as R_SOURCE
    assert torch.cuda.is_available(), 'Use an allocated GPU node'
    torch.set_num_threads(4)
    directory, checkpoint, identity = checkpoint_for(objective, variant)
    output = HERE/('smoke' if smoke else 'evaluation')/objective/variant
    if (output/'EVAL_COMPLETE.json').exists():
        done = json.loads((output/'EVAL_COMPLETE.json').read_text())
        assert done['metrics_sha256'] == digest(output/'metrics.json')
        assert done['checkpoint_sha256'] == digest(checkpoint/'model.safetensors')
        print('REUSE', objective, variant, flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    weights = checkpoint/'model.safetensors'
    weight_stat = weights.stat()
    weight_hash = digest(weights)
    experiment = json.loads((directory/'experiment.json').read_text())
    for relative in ['ottc/models/sequence/wav2vec2_ottc.py', 'ottc/ops_loss/ot_loss.py']:
        path = ROOT/relative
        assert digest(path) == experiment['identity']['source_sha256'][str(path)]
    model, loading = wvForOTTC.from_pretrained(checkpoint, output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    assert model.config.bins_metric == OBJECTIVES[objective] and model.config.vocab_size == 42
    model.eval().cuda()
    model.step = identity['cumulative_optimizer_step']
    processor = Wav2Vec2Processor.from_pretrained(directory/'processor')
    vocab = json.loads((directory/'vocab_used.json').read_text())
    inverse = {v: k for k, v in vocab.items()}
    backend, provenance = load_backend()
    (output/'ifmdd_k2_source_snapshot.py').write_text(provenance.pop('source_snapshot')+'\n')
    (output/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    helpers = scoring_helpers()
    result = dict(objective=objective, variant=variant, checkpoint=str(checkpoint),
        checkpoint_sha256=weight_hash, checkpoint_loading_info=loading, identity=identity,
        evaluator_sha256=digest(Path(__file__)), smoke=smoke,
        protocol=dict(corpus='TIMIT dev400/core-test192, zero-shot Libri-trained checkpoints',
            training_on_timit=False, optimizer_updated=False, inference='FP32 batch1 audio-only argmax',
            evaluation_loss='Separate labelled forward; raw OTTC utterance objective averaged over utterances; different objectives not comparable',
            reference='Frozen historical human PHN intervals; closures as sil, q removed, gaps preserved, sil retained; dx to d for recognition',
            boundary_protocol='unique interval endpoints; exclude utterance edges20ms; one-to-one time-only matching at10/20/30ms',
            text_free_clock='fixed .02 seconds per frame; raw argmax changes including silence',
            forced_input='Exact reference phone sequence including sil, without reference timestamps',
            forced_clock='duration/frame_count; blank previous; no offset', k2=provenance,
            r_value_source_sha256=digest(R_SOURCE),
            selection='Preserve Libri dev choice; TIMIT dev and test never select checkpoints',
            failure_policy='Full-set P/R/F1/R includes failed FA with zero matches; MAE only successful exact-target alignments'),
        host=platform.node(), pbs_job=os.environ.get('PBS_JOBID'), splits={})
    torch.cuda.reset_peak_memory_stats()
    checked = False
    with torch.inference_mode():
        for split in SPLITS:
            manifest = HERE/'data'/(split+'.jsonl')
            rows = [json.loads(line) for line in manifest.open()]
            if smoke:
                ordered = sorted(rows, key=lambda r: r['duration'])
                rows = [ordered[-1], ordered[0], ordered[len(ordered)//2]]
            totals = {mode: helpers['new_counts']() for mode in ['text_free', 'forced_k2']}
            losses = []
            with (output/(split+'_predictions.jsonl')).open('w') as stream:
                for i, row in enumerate(rows):
                    audio, sr = sf.read(row['wav'], dtype='float32')
                    assert sr == 16000 and audio.ndim == 1
                    duration = len(audio)/sr
                    assert abs(duration-row['duration']) < .001
                    batch = processor(audio, sampling_rate=sr, return_tensors='pt', return_attention_mask=True).to('cuda')
                    out = model(**batch)
                    length = int(out.hidden_states[1][0].sum())
                    logits = out.logits[0, :length].float()
                    assert torch.isfinite(logits).all()
                    labels = torch.tensor([row['loss_target_ids']], dtype=torch.long, device='cuda')
                    loss_out = model(**batch, labels=labels)
                    assert torch.isfinite(loss_out.loss)
                    if not checked:
                        torch.testing.assert_close(out.logits, loss_out.logits, atol=0, rtol=0)
                        checked = True
                    losses.append(float(loss_out.loss))
                    ids = logits.argmax(-1).cpu().numpy()
                    ids[ids == 41] = 0
                    hypothesis = processor.decode(ids).split()
                    reference = row['reference']
                    count = totals['text_free']; count['utterances'] += 1
                    count['refs'].append(' '.join('d' if p == 'dx' else p for p, _, _ in reference if p not in ('sil', 'q')))
                    count['hyps'].append(' '.join('d' if p == 'dx' else p for p in hypothesis if p not in ('sil', 'q')))
                    changes = np.flatnonzero(ids[1:] != ids[:-1])+1
                    helpers['score'](count, reference, [int(t)*.02 for t in changes], duration)
                    record = dict(utt=row['utt'], wav=row['wav'], duration=duration, reference=reference,
                        raw_loss=losses[-1], text_free=dict(hypothesis=hypothesis, frame_token_ids=ids.tolist()),
                        reference_times_used_by_aligner=False)
                    count = totals['forced_k2']; count['utterances'] += 1
                    targets = [vocab[p] for p, _, _ in reference]
                    try:
                        segments, path = align(logits.log_softmax(-1), targets, duration, backend, blank_policy='previous')
                        spans = [[inverse[s['token_id']], s['start'], s['end']] for s in segments]
                        assert [p for p, _, _ in spans] == [p for p, _, _ in reference]
                        helpers['score'](count, reference, [t for _, s, e in spans for t in (s, e)], duration)
                        for (_, rs, re), (_, ps, pe) in zip(reference, spans):
                            err = count['errors']; err['tokens'] += 1
                            err['start'] += abs(rs-ps); err['end'] += abs(re-pe)
                            err['start_hit20'] += abs(rs-ps) <= .02+1e-9
                            err['end_hit20'] += abs(re-pe) <= .02+1e-9
                        record['forced_k2'] = dict(predicted_spans=spans, supplied_target_ids=targets, frame_token_ids=path)
                    except ValueError as exc:
                        count['alignment_failures'] += 1
                        helpers['score'](count, reference, [], duration)
                        record['forced_k2'] = dict(failure=str(exc), supplied_target_ids=targets)
                    stream.write(json.dumps(record)+'\n')
                    if (i+1) % 100 == 0 or i+1 == len(rows):
                        print('PROGRESS', objective, variant, split, i+1, '/', len(rows), flush=True)
            result['splits'][split] = {mode: helpers['summary'](c) for mode, c in totals.items()}
            result['splits'][split]['raw_evaluation_loss'] = sum(losses)/len(losses)
            result.setdefault('manifest_sha256', {})[split] = digest(manifest)
    result.update(label_independence_verified=checked, elapsed_seconds=time.monotonic()-begun,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    assert (weights.stat().st_mtime_ns, weights.stat().st_size) == (weight_stat.st_mtime_ns, weight_stat.st_size)
    save(output/'metrics.json', result)
    save(output/'EVAL_COMPLETE.json', dict(checkpoint_sha256=weight_hash, metrics_sha256=digest(output/'metrics.json'),
        prediction_sha256={s: digest(output/(s+'_predictions.jsonl')) for s in SPLITS},
        utterances={s: result['splits'][s]['text_free']['utterances'] for s in SPLITS},
        checkpoint_unchanged=True, smoke=smoke, elapsed_seconds=result['elapsed_seconds']))
    print('COMPLETE', objective, variant, result['elapsed_seconds'], flush=True)
    del model, processor
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectives', nargs='+', choices=list(OBJECTIVES), default=list(OBJECTIVES))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    for objective in args.objectives:
        for variant in args.variants:
            evaluate(objective, variant, args.smoke)


if __name__ == '__main__':
    main()
