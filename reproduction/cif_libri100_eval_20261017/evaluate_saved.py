"""Evaluate frozen best-dev Libri100 CIF checkpoints without changing training."""
from collections import Counter
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import time

import torch

from ottc.models.sequence.wav2vec2_cif import wvCIFConfig, wvForCIF
from reproduction.cif_quantity_20261016.data import ROOT, load_rows, sha
from reproduction.cif_quantity_20261016.evaluate import evaluate
from reproduction.cif_timit_20261016.evaluate import edits
from reproduction.segmentation_metrics import SOURCE, _scorer


HERE = Path(__file__).resolve().parent
TRAINING = ROOT / 'reproduction/cif_quantity_20261016'
VARIANTS = ('ottc_jsd_0p5', 'ce', 'ctc_0p5')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def subset_metrics(predictions, ids):
    selected = [x for x in predictions if x['utt'] in ids]
    assert len(selected) == len(ids) == len({x['utt'] for x in selected})
    totals = Counter()
    counts = {mode: {t: Counter() for t in (10, 20, 30)}
              for mode in ('text_free', 'text_free_frame')}
    normalize = lambda xs: ['d' if x == 'dx' else x for x in xs if x not in ('sil', 'q')]
    for row in selected:
        ref, hyp = row['reference'], row['hypothesis']
        totals.update(errors=edits(ref, hyp), phones=len(ref),
                      normalized_errors=edits(normalize(ref), normalize(hyp)),
                      normalized_phones=len(normalize(ref)))
        for mode, by_tolerance in counts.items():
            for tolerance, c in by_tolerance.items():
                c.update(_scorer.precision_recall_counts(
                    row['boundaries'][mode], row['reference_boundaries'], tolerance / 1000))
    result = dict(utterances=len(selected), raw_PER_percent=100 * totals['errors'] / totals['phones'],
                  normalized_nosil_PER_percent=100 * totals['normalized_errors'] / totals['normalized_phones'],
                  counts=dict(totals), audio_only=True, forced_alignment=False,
                  rvalue_source_sha256=sha(SOURCE),
                  provenance='subset re-score of complete saved predictions; no new inference or subset loss estimate')
    for mode, by_tolerance in counts.items():
        result[mode] = {str(t): _scorer.finalize_counts(
            c['precision_matches'], c['recall_matches'],
            c['predicted_boundaries'], c['reference_boundaries']) for t, c in by_tolerance.items()}
    return result


def tracking(output, config):
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    identity = output / 'wandb_run.json'
    saved = json.loads(identity.read_text()) if identity.exists() else {}
    options = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                   entity=os.environ.get('WANDB_ENTITY', defaults.get('entity')),
                   name='libri100_' + output.name + '_bestdev_interim_evaluation',
                   group='cif_libri100_interim_evaluation_20261017', config=config,
                   dir=str(output), id=saved.get('id'), mode=os.environ.get('WANDB_MODE', 'online'),
                   settings=wandb.Settings(init_timeout=35))
    if saved and options['mode'] == 'online':
        options['resume'] = 'allow'
    try:
        run = init_evaluation(**options)
    except wandb.errors.CommError:
        options.update(mode='offline')
        options.pop('resume', None)
        run = init_evaluation(**options)
    write(identity, dict(id=run.id, entity=run.entity, project=run.project, mode=run.settings.mode))
    return run


def main():
    assert torch.cuda.is_available(), 'Run this evaluator on an allocated GPU node'
    torch.set_num_threads(4)
    libri, vocab, data_contract = load_rows('libri100')
    buckeye, buck_vocab, buck_contract = load_rows('buckeye')
    assert vocab == buck_vocab
    assert len(libri['dev']) == 5567 and len(libri['test']) == 5558 and len(buckeye['test']) == 1840
    for variant in VARIANTS:
        output = HERE / 'runs' / variant
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'EVAL_COMPLETE.json').exists():
            print('ALREADY COMPLETE', variant, flush=True)
            continue
        source = TRAINING / 'runs' / f'libri100_{variant}_q1_seed42'
        contract = json.loads((source / 'run_config.json').read_text())
        for file, digest in contract['sources_sha256'].items():
            assert sha(ROOT / file) == digest, file
        assert contract['manifests_sha256'] == data_contract['manifests_sha256']
        snapshot = output / 'checkpoint.pt'
        if not snapshot.exists():
            # Training writes best_dev_r.pt via atomic rename. A hardlink pins
            # this immutable inode even if a later epoch replaces the source.
            os.link(source / 'best_dev_r.pt', snapshot)
        checkpoint_hash = sha(snapshot)
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
        assert checkpoint['model_config']['quantity_weight'] == 1
        selected_epoch, selected_step = checkpoint['epoch'], checkpoint['step']
        dev_path = source / f'dev_epoch_{selected_epoch:03d}.json'
        dev = json.loads(dev_path.read_text())
        assert abs(dev['text_free']['20']['r_value_pct'] - checkpoint['dev_R_value_20ms']) < 1e-8
        shutil.copyfile(dev_path, output / 'selected_dev.json')
        config = dict(evaluation_only=True, interim_training_evaluation=True, trained_epochs_target=40,
                      selected_epoch=selected_epoch, selected_step=selected_step,
                      selection='maximum dev text-free R-value@20ms available before test evaluation',
                      checkpoint_sha256=checkpoint_hash, source_checkpoint=str(source / 'best_dev_r.pt'),
                      frozen_snapshot=str(snapshot), source_run=json.loads((source / 'wandb_run.json').read_text()),
                      model_config=checkpoint['model_config'], quantity_weight=1,
                      libri_data=data_contract, buckeye_data=buck_contract,
                      evaluator_sha256=sha(Path(__file__)),
                      evaluation_source_sha256=sha(ROOT / 'reproduction/cif_quantity_20261016/evaluate.py'),
                      rvalue_source_sha256=sha(SOURCE), training_sources_sha256=contract['sources_sha256'],
                      test_coverage=5558, buckeye_test_coverage=1840,
                      boundaries='text-free, fractional duration/T and right-cell grid, frozen IF-MDD scorer, 10/20/30ms',
                      reference_quality='Libri automatic alignments of unverified aligner identity; Buckeye human phone boundaries',
                      host=platform.node(), pbs_job=os.environ.get('PBS_JOBID'),
                      created_at=datetime.now().astimezone().isoformat())
        write(output / 'run_config.json', config)
        run = tracking(output, config)
        success = False
        started = time.monotonic()
        try:
            print('LOADING', variant, 'selected epoch', selected_epoch, flush=True)
            model = wvForCIF(wvCIFConfig(**checkpoint['model_config']))
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            model.eval().to('cuda')
            torch.cuda.reset_peak_memory_stats()
            longest = sorted(libri['test'], key=lambda r: r['duration'], reverse=True)[:16]
            probe = evaluate(model, longest, vocab, output / 'longest16_probe.json', torch.bfloat16, 8)
            write(output / 'PROBE_PASS.json', dict(utterances=16, partial=True,
                  max_duration=max(r['duration'] for r in longest),
                  peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                  elapsed_seconds=probe['elapsed_seconds']))
            for name, rows in [('libri_test', libri['test']), ('buckeye_test', buckeye['test'])]:
                result = evaluate(model, rows, vocab, output / (name + '.json'), torch.bfloat16, 8)
                records = [json.loads(x) for x in (output / (name + '.predictions.jsonl')).open()]
                assert len(records) == len({x['utt'] for x in records}) == len(rows)
                assert {x['utt'] for x in records} == {x['utt'] for x in rows}
                result.update(partial=False, interim_training_evaluation=True,
                              selected_epoch=selected_epoch, selected_step=selected_step,
                              checkpoint_sha256=checkpoint_hash, checkpoint_selection=config['selection'])
                write(output / (name + '.json'), result)
                run.log({name + '/PER_nosil': result['normalized_nosil_PER_percent'],
                         name + '/PER_raw': result['raw_PER_percent'],
                         **{name + '/' + k + '_20ms': result['text_free']['20'][k + '_pct']
                            for k in ('precision', 'recall', 'f1', 'r_value')}})
                print('RESULT', variant, name, json.dumps(result), flush=True)
                if name == 'libri_test':
                    for subset in ('clean', 'other'):
                        ids = {r['utt'] for r in rows if r['subset'] == 'test-' + subset}
                        assert len(ids) == (2620 if subset == 'clean' else 2938)
                        sub = subset_metrics(records, ids)
                        sub.update(interim_training_evaluation=True, selected_epoch=selected_epoch,
                                   selected_step=selected_step, checkpoint_sha256=checkpoint_hash)
                        write(output / ('libri_test_' + subset + '.json'), sub)
                        run.log({'libri_test_' + subset + '/PER_nosil': sub['normalized_nosil_PER_percent'],
                                 **{'libri_test_' + subset + '/' + k + '_20ms': sub['text_free']['20'][k + '_pct']
                                    for k in ('precision', 'recall', 'f1', 'r_value')}})
            assert sha(snapshot) == checkpoint_hash
            write(output / 'EVAL_COMPLETE.json', dict(
                evaluation_only=True, interim_training_evaluation=True,
                full_test_coverage=True, selected_epoch=selected_epoch, selected_step=selected_step,
                checkpoint_sha256=checkpoint_hash, wall_seconds=time.monotonic() - started,
                finished_at=datetime.now().astimezone().isoformat(), pbs_job=os.environ.get('PBS_JOBID')))
            import wandb
            artifact = wandb.Artifact('libri100_' + variant + '_interim_evaluation', type='evaluation')
            for path in output.glob('*.json'):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary['evaluation_complete'] = True
            success = True
            del model
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            run.finish(exit_code=0 if success else 1)


if __name__ == '__main__':
    main()
