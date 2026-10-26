"""Offline repeat-collapse diagnostic; never changes the position decoder."""
import json
from pathlib import Path

from reproduction.gamma_pool_timit_20261022.evaluate import HERE, ROOT, score, report, write
from reproduction.timit_ssl_native10_20261019.common import phones_from_ids, normalize


def main():
    output = HERE/'results'
    complete = json.loads((output/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['full_sets']
    vocab = json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json').read_text())
    diagnostics = output/'repeat_collapse_diagnostic'
    diagnostics.mkdir(exist_ok=True)
    totals = {}
    examples = []
    selected = {'FDHC0_SI2189', 'FDHC0_SX29', 'FELC0_SI1386'}
    for cost in complete['costs']:
        totals[cost] = {}
        for split in ['dev', 'test']:
            records = [json.loads(l) for l in (output/f'{cost}_{split}.predictions.jsonl').open()]
            totals[cost][split] = {}
            for mode, key in [('gamma_greedy_length', 'gamma_greedy_ids'),
                              ('gamma_reference_length', 'gamma_reference_ids')]:
                collapsed = []
                for r in records:
                    hypothesis = normalize(phones_from_ids(r[key], vocab, collapse=True))
                    collapsed.append(dict(utt=r['utt'], reference=r['reference'], hypotheses={mode:hypothesis}))
                    if split == 'test' and r['utt'] in selected:
                        examples.append(dict(cost=cost, utt=r['utt'], mode=mode,
                            reference=r['reference'], greedy=r['hypotheses']['greedy'],
                            gamma=r['hypotheses'][mode], collapsed=hypothesis,
                            greedy_length=r['greedy_length'], reference_length=r['reference_length'],
                            scores={m:score([r['reference']], [h]) for m,h in
                                    [('greedy', r['hypotheses']['greedy']), ('gamma', r['hypotheses'][mode]), ('collapsed',hypothesis)]}))
                values = score([r['reference'] for r in collapsed], [r['hypotheses'][mode] for r in collapsed])
                totals[cost][split][mode+'_collapse'] = values
                report(diagnostics/f'{cost}_{split}_{mode}_collapse.wer.txt', collapsed, mode, values)
    write(diagnostics/'RESULTS.json', totals)
    write(diagnostics/'EXAMPLES.json', examples)
    write(diagnostics/'PROVENANCE.json', dict(operation='collapse consecutive equal token IDs before special-token filtering; diagnostic only',
        inference_rerun=False, core_decoder_changed=False, reference_inputs_unchanged=True,
        caveat='May remove legitimate repeated targets; cannot recover missing distinct phones'))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
