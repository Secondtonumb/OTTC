"""Read-only checkpoint evaluation through the original S3T2** phone-loop FSA."""
import gc
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / 'reproduction/zapformer_s3t2ss_native_20261012'
sys.path.insert(0, str(NATIVE))
import runtime
import torch
import k2
from task_model import S3T2
from ifmdd_losses import build_s3t2_phone_loop, subphonetic_greedy_complete
sys.path.insert(0, str(ROOT / 'reproduction/english_suite'))
from mfa_metrics import audio_forward, boundary_scores, summarize
sys.path.insert(0, str(ROOT / 'reproduction/ipa_arpabet_comparison_20261012'))
from rescore import mapped
import jiwer
import numpy as np

DATA = ROOT / 'reproduction/spottc_jsd_s3t2_framece_libri100_20261012/eval_data'
ARMS = {
    'jsd_fs': ROOT / 'reproduction/zapformer_jsd_charsiu_fs_20261013/training',
    **{arm: ROOT / 'reproduction/zapformer_libri100_ablation_20261013/models' / arm
       for arm in ('jsd_only', 'ce_fs', 'ce_only')},
}
TOPOLOGY = 's3_t2_star_star'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def graph(device='cpu'):
    arcs, outputs, n = build_s3t2_phone_loop(104, TOPOLOGY, 0)
    lines = [f'{a.src} {a.dst} {a.label} {outputs.get(i, 0)} 0.0'
             for i, a in enumerate(arcs)]
    lines.append(f'0 {n} -1 -1 0.0')
    text = '\n'.join(sorted(lines, key=lambda s: tuple(map(int, s.split()[:3]))) + [str(n)]) + '\n'
    fsa = k2.arc_sort(k2.Fsa.from_str(text, acceptor=False).to(device))
    values = fsa.arcs.values()
    fsa.original_src = values[:, 0].clone()
    fsa.original_dst = values[:, 1].clone()
    return k2.Fsa.from_fsas([fsa]), text


def decode(scores, vector):
    """Exact dense intersection; no search beam, LM, transcript or phone penalty."""
    frames = len(scores)
    dense = k2.DenseFsaVec(scores.log_softmax(-1).unsqueeze(0),
                           torch.tensor([[0, 0, frames]], dtype=torch.int32))
    lattice = k2.intersect_dense(vector, dense, output_beam=8., frame_idx_name='frame_idx')
    best = k2.shortest_path(lattice, use_double_scores=True)
    labels = best.labels.cpu().tolist()
    outputs = best.aux_labels.cpu().tolist()
    indices = best.frame_idx.cpu().tolist()
    srcs = best.original_src.cpu().tolist()
    dsts = best.original_dst.cpu().tolist()
    current = 0
    path = []
    hyp = []
    for label, output, index, src, dst in zip(labels, outputs, indices, srcs, dsts):
        assert src == current, (src, current)
        current = dst
        if label < 0:
            assert label == -1 and src == 0
            continue
        assert index == len(path), (index, len(path))
        path.append(label)
        if output > 0:
            assert src == 0 and (label - 1) % 3 == 0
            hyp.append(output)
        if label == 0:
            assert src == dst == 0, 'Blank inside an unfinished phone'
    assert len(path) == frames
    return hyp, path, float(best.scores.double().sum())


def fixtures():
    vector, text = graph()
    # Current greedy removes this internal blank; the legal FSA must repair it.
    path = [7, 0, 9]
    scores = torch.full((3, 313), -30.)
    scores[torch.arange(3), torch.tensor(path)] = 0.
    hyp, legal, _ = decode(scores, vector)
    assert hyp == [3] and legal != path
    assert subphonetic_greedy_complete(torch.tensor([path]), torch.tensor([3]), TOPOLOGY) == [[3]]
    for path, expected in [([7, 9], [3]), ([7, 7, 8, 8, 9, 9], [3]),
                           ([7, 9, 7, 9], [3, 3]), ([0, 7, 9, 0, 10, 12], [3, 4])]:
        scores = torch.full((len(path), 313), -30.)
        scores[torch.arange(len(path)), torch.tensor(path)] = 0.
        hyp, legal, _ = decode(scores, vector)
        assert hyp == expected and legal == path, (path, hyp, legal)
    (HERE / 'phone_loop.fst.txt').write_text(text)
    return {'status': 'passed', 'internal_blank_rejected': True,
            'legal_paths_and_repeated_phone_occurrences': True}


def edit_counts(count, ref, hyp):
    edits = jiwer.process_words(' '.join(ref), ' '.join(hyp))
    count.update(reference_phones=len(ref), hypothesis_phones=len(hyp), utterances=1,
                 substitutions=edits.substitutions, insertions=edits.insertions,
                 deletions=edits.deletions, phone_errors=edits.substitutions + edits.insertions + edits.deletions)


def edits_summary(count):
    n = count['reference_phones']
    return {**dict(count), 'PER_percent': 100 * count['phone_errors'] / n,
            **{k + '_percent': 100 * count[k] / n for k in ('substitutions', 'deletions', 'insertions')}}


def init_run(out, arm, config):
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    receipt = out / 'wandb_run.json'
    opts = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
                mode=os.environ.get('WANDB_MODE', defaults.get('mode', 'online')),
                name=f'{arm}_s3t2ss_topology_decode_epoch50', job_type='evaluation',
                group='zapformer_topology_decode_20261013', dir=str(out), config=config,
                settings=wandb.Settings(init_timeout=45, disable_code=True))
    if receipt.exists():
        opts.update(id=json.loads(receipt.read_text())['id'], resume='allow')
    try:
        run = init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode'] = 'offline'
        run = init_evaluation(**opts)
    receipt.write_text(json.dumps(dict(id=run.id, url=run.url, mode=run.settings.mode)) + '\n')
    run.define_metric('checkpoint_optimizer_step')
    run.define_metric('evaluation/*', step_metric='checkpoint_optimizer_step')
    return run


def main():
    torch.set_num_threads(4)
    fixture = fixtures()
    (HERE / 'FIXTURE_PASS.json').write_text(json.dumps(fixture, indent=2) + '\n')
    if '--fixtures' in sys.argv:
        print(fixture, flush=True)
        return
    assert torch.cuda.is_available(), 'Use a GPU compute node for acoustic inference'
    sources = [Path(__file__), NATIVE / 'ifmdd_losses.py', NATIVE / 'task_model.py',
               ROOT / 'reproduction/english_suite/mfa_metrics.py',
               ROOT / 'reproduction/ipa_arpabet_comparison_20261012/rescore.py',
               ROOT / 'reproduction/segmentation_metrics.py',
               ROOT / 'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py',
               ROOT / 'reproduction/provenance/mfa_timit_metrics_source/evaluate_l2arctic_fa_timestamps.py']
    sources = [p for p in sources if p.exists()]
    protocol = dict(training_performed=False, original_experiments_modified=False,
                    topology=TOPOLOGY, decoder='exact k2 dense intersection and Viterbi shortest path',
                    graph='free phone loop; blank only at phone boundaries; all phone substates retained',
                    graph_source=str(NATIVE / 'ifmdd_losses.py'),
                    graph_constructor='build_s3t2_phone_loop -> add_subphonetic_phone_arcs',
                    search_pruning=False, output_beam=8., language_model=False,
                    supplied_phones=False, supplied_timestamps=False, insertion_penalty=0.,
                    source_hashes={str(p): sha(p) for p in sources},
                    reference='same frozen speech-only references and IPA -> English-collapsed ARPAbet/TIMIT39',
                    inference_precision='BF16 acoustic forward, float32 state log-softmax',
                    segmentation='phone-level changes including blank on decoded legal state path; original 10ms grid and boundary matching; substate changes excluded',
                    metrics_timestamp='new evaluation; checkpoint epoch/step retained; no training timestamp backfill',
                    test_checkpoint_selection=False, optimizer_step=5300, epoch=50,
                    fixture=fixture, dataset_hashes={s: sha(DATA / f'{s}.jsonl') for s in ('development', 'test')})
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows_by_split = {s: [json.loads(x) for x in (DATA / f'{s}.jsonl').read_text().splitlines()]
                     for s in ('development', 'test')}
    assert len(rows_by_split['development']) == 400 and len(rows_by_split['test']) == 192
    vector, _ = graph('cuda')
    caches = {}
    all_results = []
    for arm, experiment in ARMS.items():
        out = HERE / arm
        out.mkdir(exist_ok=True)
        checkpoint = experiment / 'step-000005300.pt'
        checkpoint_hash = sha(checkpoint)
        prior = json.loads((experiment / 'wandb_run.json').read_text())
        config = dict(protocol, arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                      training_run=prior, training_config=json.loads((experiment / 'run_config.json').read_text()))
        run = init_run(out, arm, config)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert saved['progress']['global_step'] == 5300 and saved['progress']['epoch'] == 50
        assert saved['contract']['topology'] == TOPOLOGY
        net = S3T2('jsd', topology=TOPOLOGY).cuda().eval()
        arm_results = []
        for variant, state_key in [('final', 'model'), ('cumulative_average', 'model_avg')]:
            net.load_state_dict(saved[state_key], strict=True)
            for split, rows in rows_by_split.items():
                start = time.monotonic()
                historical = {}
                eval_dir = 'step-000005300' if variant == 'final' else 'averaged-step-000005300'
                for file in (experiment / 'evaluations' / eval_dir / split).glob(f'rank-*/{split}_predictions.jsonl'):
                    for line in file.open():
                        row = json.loads(line)
                        assert row['id'] not in historical
                        historical[row['id']] = row['predicted_ids']
                assert len(historical) == len(rows), (arm, variant, split, len(historical))
                mapped_count = Counter()
                native_count = Counter()
                greedy_count = Counter()
                changed_frames = total_frames = replay_mismatch = 0
                with (out / f'{variant}_{split}_predictions.jsonl').open('w') as stream, torch.no_grad():
                    for i, row in enumerate(rows):
                        path = row['feature_file']
                        if path not in caches:
                            caches[path] = np.memmap(path, dtype=np.float32, mode='r')
                        features = np.array(caches[path][row['feature_offset']:row['feature_offset'] + row['frames'] * 80]).reshape(-1, 80)
                        x = torch.from_numpy(features).unsqueeze(0).cuda()
                        lengths = torch.tensor([len(features)], device='cuda')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits, lengths, _ = audio_forward(net, 'spottc', x, lengths)
                        scores = logits[0].float()
                        hyp_all, state_path, path_score = decode(scores, vector)
                        hyp = [p for p in hyp_all if 3 <= p <= 102]
                        raw = scores.argmax(-1)
                        greedy = [p for p in subphonetic_greedy_complete(raw.unsqueeze(0), lengths, TOPOLOGY)[0] if 3 <= p <= 102]
                        replay_mismatch += int(greedy != historical[row['id']])
                        edit_counts(mapped_count, mapped(row['tokens'], 'english_collapsed'), mapped(hyp, 'english_collapsed'))
                        edit_counts(greedy_count, mapped(row['tokens'], 'english_collapsed'), mapped(greedy, 'english_collapsed'))
                        edit_counts(native_count, list(map(str, row['tokens'])), list(map(str, hyp)))
                        phones = [0 if p == 0 else (p - 1) // 3 + 1 for p in state_path]
                        edges = [j * .01 for j in range(1, len(phones)) if phones[j] != phones[j - 1]]
                        boundary_scores(native_count, row['references'], edges, row['duration'])
                        raw_list = raw.cpu().tolist()
                        changed_frames += sum(a != b for a, b in zip(raw_list, state_path))
                        total_frames += len(state_path)
                        stream.write(json.dumps(dict(id=row['id'], predicted_ids=hyp, all_phone_ids=hyp_all,
                                                     greedy_ids=greedy, state_argmax=raw_list, legal_state_path=state_path,
                                                     path_score=path_score, audio_only=True, path_legality_verified=True)) + '\n')
                        if (i + 1) % 100 == 0:
                            print(arm, variant, split, i + 1, '/', len(rows), flush=True)
                assert replay_mismatch == 0, f'Acoustic/greedy replay mismatch in {arm}/{variant}/{split}: {replay_mismatch}'
                result = dict(arm=arm, checkpoint_variant=variant, split=split, optimizer_step=5300, epoch=50,
                              mapped_arpabet=edits_summary(mapped_count), greedy_mapped_arpabet=edits_summary(greedy_count),
                              text_free_topology=summarize(native_count),
                              raw_argmax_changed_frame_percent=100 * changed_frames / total_frames,
                              greedy_replay_mismatch_utterances=replay_mismatch,
                              legal_paths_verified=len(rows), elapsed_seconds=time.monotonic() - start)
                arm_results.append(result)
                all_results.append(result)
                (out / 'results.json').write_text(json.dumps(arm_results, indent=2) + '\n')
                (HERE / 'results.json').write_text(json.dumps(all_results, indent=2) + '\n')
                payload = {'checkpoint_optimizer_step': 5300}
                for mode in ('mapped_arpabet', 'greedy_mapped_arpabet', 'text_free_topology'):
                    payload.update({f'evaluation/{variant}/{split}/{mode}/{k}': v for k, v in result[mode].items()
                                    if isinstance(v, (int, float))})
                run.log(payload)
                run.summary.update(payload)
                print('RESULT', json.dumps(result), flush=True)
        import wandb
        artifact = wandb.Artifact(f'{arm}-s3t2ss-topology-decode', type='evaluation')
        artifact.add_file(str(out / 'results.json'))
        artifact.add_file(str(HERE / 'protocol.json'))
        artifact.add_file(str(HERE / 'phone_loop.fst.txt'))
        run.log_artifact(artifact)
        run.summary['evaluation/complete'] = True
        run.finish()
        (out / 'COMPLETE.json').write_text(json.dumps(dict(arm=arm, checkpoint_sha256=checkpoint_hash,
            model_and_average_evaluated=True, greedy_replay='exact', completed_at=datetime.now(timezone.utc).isoformat())) + '\n')
        del net, saved
        gc.collect()
        torch.cuda.empty_cache()
    (HERE / 'COMPLETE.json').write_text(json.dumps(dict(results=len(all_results), training_performed=False,
        completed_at=datetime.now(timezone.utc).isoformat(), protocol=protocol), indent=2) + '\n')


if __name__ == '__main__':
    main()
