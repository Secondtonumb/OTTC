"""Local-only FP32 evaluation of the three original dev-selected GW weights."""
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from reproduction.timit_gw_20261026.common import ROOT, build_model, load_data, sha, write
from reproduction.timit_gw_20261026.evaluate import average, evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'reproduction/timit_gw_20261026/models/monotone_w0.1__seed43'
SELECTION = ROOT / 'reproduction/timit_gw_outputfix_20261027/JOB3714901_BEST3_DEV.json'


def main():
    assert os.environ.get('OTTC_EVAL_UPLOAD') == '0'
    torch.set_num_threads(4)
    config = json.loads((SOURCE / 'run_config.json').read_text())
    selected = json.loads(SELECTION.read_text())
    members = selected['members']
    assert [m['epoch'] for m in members] == [26, 34, 30]
    paths = [Path(m['path']) for m in members]
    for member, path in zip(members, paths):
        assert sha(path) == member['sha256']
    current_sources = {p: dict(expected=h, actual=sha(Path(p)))
                       for p, h in config['source_sha256'].items() if Path(p).exists()}
    critical = ['ottc/models/sequence/wav2vec2_ottc.py',
                'ottc/models/sequence/wav2vec2_ottc_gw.py',
                'ottc/ops_loss/gw_transport.py',
                'reproduction/timit_gw_20261026/evaluate.py',
                'reproduction/timit_ssl_native10_20261019/evaluate.py',
                'reproduction/ifmdd_k2_alignment.py']
    for relative in critical:
        entry = current_sources[str(ROOT / relative)]
        assert entry['expected'] == entry['actual'], relative
    weights = HERE / 'best3_dev_average.pt'
    average(paths, weights, config)
    saved = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    assert [m['sha256'] for m in saved['members']] == [m['sha256'] for m in members]
    assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
    write(HERE / 'PROVENANCE.json', dict(
        source_job='3714901.opbs', evaluation_job=os.environ.get('PBS_JOBID'),
        created_at=datetime.now(timezone.utc).isoformat(), members=members,
        selection=selected['selection'], precision_note=selected['precision_note'],
        test_used_for_selection=False, evaluation_uploads=False,
        average_dtype='float32', model_dtype='float32', weights_sha256=sha(weights),
        config_sha256=sha(SOURCE / 'run_config.json'),
        evaluation_source_sha256=sha(Path(__file__)), original_sources=current_sources))
    _, rows, vocab, processor, audit = load_data(config)
    assert config['data'] == audit
    model, topo = build_model(config, vocab, config['planned_steps_per_epoch'])
    model.load_state_dict(saved['model'], strict=True)
    del saved
    gc.collect()
    model.cuda().eval()
    model.step = 2920
    assert all(p.dtype == torch.float32 for p in model.parameters())
    results = {}
    for split in ('dev', 'test'):
        target = HERE / f'benchmark_best3_{split}.json'
        metrics = json.loads(target.read_text()) if target.exists() else evaluate(
            config, model, rows[split], vocab, processor, topo, target)
        assert metrics['utterances'] == len(rows[split])
        assert metrics['modes']['forced_alignment']['alignment_failures'] == 0
        predictions = target.with_suffix('.predictions.jsonl')
        records = [json.loads(line) for line in predictions.open()]
        assert len(records) == len({r['utt'] for r in records}) == len(rows[split])
        expected = {r['utt']: r for r in rows[split]}
        for row in records:
            assert row['reference'] == expected[row['utt']]['reference']
        pfer = score_predictions(predictions, metrics)
        write(HERE / f'pfer_{split}.json', pfer)
        results[split] = dict(metrics=metrics, pfer=pfer,
                              predictions_sha256=sha(predictions))
    write(HERE / 'RESULTS.json', results)
    write(HERE / 'EVAL_COMPLETE.json', dict(
        source_job='3714901.opbs', evaluation_job=os.environ.get('PBS_JOBID'),
        finished_at=datetime.now(timezone.utc).isoformat(),
        epochs=[m['epoch'] for m in members], test_used_for_selection=False,
        weights_sha256=sha(weights), results_sha256=sha(HERE / 'RESULTS.json'),
        coverage={s: len(rows[s]) for s in results}, FP32_verified=True,
        evaluation_uploads=False))
    print('EVAL_COMPLETE', json.dumps({s: dict(
        PER=r['pfer']['PER_crosscheck_percent'], PFER=r['pfer']['PFER_percent'])
        for s, r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
