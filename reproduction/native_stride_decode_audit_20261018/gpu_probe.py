"""Reproduce evaluation-only failures without modifying old recipes or weights."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / 'native_stride_ottc_20261018'
sys.path.insert(0, str(EXPERIMENT))
from common import ROOT, topology, write
import copy
import json
import torch
from graphs import graph, decode


def main():
    torch.set_num_threads(4)
    vocab = json.loads((EXPERIMENT/'models/wav2vec2__timit__10ms__s3t2ss__ce/vocab_used.json').read_text())
    topo = topology(vocab)
    record = json.loads(next((EXPERIMENT/'models/wav2vec2__timit__10ms__s3t2ss__ce/benchmark_final_test.predictions.jsonl').open()))
    phones = [vocab[p] for p, _, _ in record['reference']]
    results = {}
    for mode in [False, True]:
        for supplied, targets in [('repeat', [vocab['b'],vocab['b']]), ('real_silence', phones)]:
            key = f'inference_{mode}_{supplied}'
            try:
                with torch.inference_mode(mode):
                    fsa = graph(topo, vocab, targets)
                results[key] = dict(passed=True, arcs=int(fsa.num_arcs))
            except Exception as exc:
                results[key] = dict(passed=False, error_type=type(exc).__name__, error=str(exc).splitlines()[0][:180])
    normal = torch.nn.Linear(2, 2)
    with torch.inference_mode():
        copied = copy.deepcopy(normal)
    results['deepcopy_in_inference_creates_inference_parameter'] = copied.weight.is_inference()
    try:
        copied.load_state_dict(normal.state_dict())
        results['load_into_inference_parameter'] = dict(passed=True)
    except RuntimeError as exc:
        results['load_into_inference_parameter'] = dict(passed=False, error=str(exc).splitlines()[0])
    fresh = torch.nn.Linear(2,2);fresh.load_state_dict(normal.state_dict())
    results['fresh_model_load_succeeds'] = True
    write(HERE/'GPU_PROBE.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__': main()
