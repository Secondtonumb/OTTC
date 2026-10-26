import itertools
import json
from pathlib import Path
import numpy as np
from reproduction.buckeye_full40_exactfa_20261026.exact import exact_ctc


def main():
    rng=np.random.default_rng(42)
    checked=0
    for targets in ([1],[1,2],[1,1]):
        for frames in (3,4,5):
            lp=rng.normal(size=(frames,3)); best=-np.inf
            for path in itertools.product(range(3),repeat=frames):
                collapsed=[p for i,p in enumerate(path) if p and (i==0 or p!=path[i-1])]
                if collapsed==targets:
                    best=max(best,sum(lp[t,p] for t,p in enumerate(path)))
            path,score=exact_ctc(lp,targets)
            assert abs(score-best)<1e-10
            assert abs(sum(lp[t,p] for t,p in enumerate(path))-score)<1e-10
            checked+=1
    try:
        exact_ctc(np.zeros((2,3)),[1,1])
        raise AssertionError('Infeasible repetition accepted')
    except ValueError:
        checked+=1
    root=Path(__file__).resolve().parents[2]
    probe=root/'artifacts/buckeye_fa_mae_audit_20261026'
    vocab=json.loads((root/'reproduction/buckeye_full40_seed_sweep_20261025/models/ag__seed43/processor/vocab.json').read_text())
    for case in json.loads((probe/'EXACT_PROBE.json').read_text()):
        lp=np.load(probe/(Path(case['utterance']).name+'.log_probs.npy'))
        _,score=exact_ctc(lp,[vocab[p] for p,_,_ in case['reference']])
        assert abs(score-case['exact_path_score'])<1e-6
        checked+=1
    print(json.dumps(dict(passed=checked,coverage='exhaustive short paths, repeats, infeasibility, two real long-utterance optima')))


if __name__=='__main__': main()
