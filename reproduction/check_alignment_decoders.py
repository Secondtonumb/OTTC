"""Compare constrained Viterbi to exhaustive paths and OT to exact quantiles."""
import itertools
import numpy as np
from alignment_decoders import ctc_viterbi,native_ot_spans


def collapse(path):
    return [v for i,v in enumerate(path) if v and (not i or v!=path[i-1])]


def main():
    rng=np.random.default_rng(42)
    for targets in [[1],[1,2],[1,1],[1,2,1]]:
        scores=rng.normal(size=(5,3))
        logp=scores-np.log(np.exp(scores).sum(-1,keepdims=True))
        result=ctc_viterbi(logp,targets)
        all_scores=[sum(logp[t,v] for t,v in enumerate(path))
                    for path in itertools.product(range(3),repeat=5) if collapse(path)==targets]
        np.testing.assert_allclose(result['log_probability'],max(all_scores),rtol=0,atol=1e-10)
        assert collapse(result['frame_token_ids'])==targets
    try:ctc_viterbi(np.zeros((2,3)),[1,1])
    except ValueError:pass
    else:raise AssertionError('Repeated tokens require a blank frame')
    np.testing.assert_allclose(native_ot_spans([.25]*4,[1,2]),[(1,0,2),(2,2,4)])
    np.testing.assert_allclose(native_ot_spans([.1,.6,.3],[1,2]),[(1,0,1+2/3),(2,1+2/3,3)])
    np.testing.assert_allclose(native_ot_spans([0,.5,0,.5],[1,2]),[(1,0,2),(2,2,4)])
    import torch
    from ifmdd_k2_alignment import load_backend,align
    backend,_=load_backend()
    for targets in [[1],[1,2],[1,1],[1,2,1]]:
        scores=torch.tensor(rng.normal(size=(5,3)),dtype=torch.float32).log_softmax(-1)
        spans,path=align(scores,targets,.1,backend)
        assert collapse(path)==targets
        assert [s['token_id'] for s in spans]==targets
        assert spans[0]['start']==0 and spans[-1]['end']==.1
        expected=ctc_viterbi(scores.numpy(),targets)['log_probability']
        score=sum(float(scores[t,v]) for t,v in enumerate(path))
        np.testing.assert_allclose(score,expected,rtol=0,atol=1e-6)
    print('PASS exact IF-MDD K2-FSA paths match exhaustive objective, repeated targets, previous-blank timestamps, OT quantiles')


if __name__=='__main__':main()
