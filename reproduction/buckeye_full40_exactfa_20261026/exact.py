"""Unpruned CTC Viterbi. Inputs are frame scores and target IDs, never gold times."""
import numpy as np
from numba import njit


@njit(cache=True)
def _viterbi(logp, targets):
    frames, _ = logp.shape
    states = len(targets)*2+1
    labels = np.zeros(states, dtype=np.int64)
    labels[1::2] = targets
    previous = np.full(states, -np.inf)
    previous[0] = logp[0, 0]
    previous[1] = logp[0, targets[0]]
    back = np.zeros((frames, states), dtype=np.uint8)
    for t in range(1, frames):
        current = np.full(states, -np.inf)
        for s in range(states):
            value, jump = previous[s], 0
            if s > 0 and previous[s-1] > value:
                value, jump = previous[s-1], 1
            if s > 1 and labels[s] != 0 and labels[s] != labels[s-2] and previous[s-2] > value:
                value, jump = previous[s-2], 2
            current[s] = value + logp[t, labels[s]]
            back[t, s] = jump
        previous = current
    state = states-1 if previous[-1] > previous[-2] else states-2
    score = previous[state]
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames-1, -1, -1):
        path[t] = labels[state]
        if t:
            state -= back[t, state]
    return path, score


def exact_ctc(logp, targets):
    logp = np.ascontiguousarray(logp, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logp.ndim != 2 or not len(logp) or not len(targets):
        raise ValueError('Nonempty frame scores and targets required')
    if not np.isfinite(logp).all() or np.any(targets <= 0) or np.any(targets >= logp.shape[1]):
        raise ValueError('Invalid scores or supplied targets')
    path, score = _viterbi(logp, targets)
    if not np.isfinite(score):
        raise ValueError('CTC target cannot be covered by available frames')
    collapsed = [int(p) for i,p in enumerate(path) if p != 0 and (i == 0 or p != path[i-1])]
    if collapsed != targets.tolist():
        raise ValueError('Exact path does not cover supplied target occurrences')
    return path, float(score)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(n=len(a), MAE_ms=float(a.mean()), median_ms=float(np.median(a)),
                p90_ms=float(np.quantile(a,.9)), p95_ms=float(np.quantile(a,.95)),
                p99_ms=float(np.quantile(a,.99)), max_ms=float(a.max()),
                capped100_mean_ms=float(np.minimum(a,100).mean()),
                **{f'over_{t}ms_percent':float((a>t).mean()*100) for t in (100,200,500)})
