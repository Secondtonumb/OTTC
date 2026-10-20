"""Reference-conditioned decoders; neither receives reference timestamps."""
import numpy as np


def ctc_viterbi(log_probs,targets,blank=0):
    """Maximum-probability blank-augmented path for a fixed token sequence.

    The common decoder can be applied to any frame posterior, including OTTC.
    Applying it to OTTC does not turn its training objective into CTC.
    """
    log_probs=np.asarray(log_probs,dtype=np.float64)
    targets=np.asarray(targets,dtype=np.int64)
    frames,vocab=log_probs.shape
    if not len(targets) or np.any(targets==blank) or np.any(targets<0) or np.any(targets>=vocab):
        raise ValueError('Expected nonempty valid nonblank targets')
    repeats=int(np.sum(targets[1:]==targets[:-1]))
    if frames<len(targets)+repeats:raise ValueError('Insufficient frames for the supplied sequence')
    states=np.full(2*len(targets)+1,blank,dtype=np.int64);states[1::2]=targets
    skip=np.zeros(len(states),dtype=bool)
    skip[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
    previous=np.full(len(states),-np.inf);previous[:2]=log_probs[0,states[:2]]
    moves=np.zeros((frames,len(states)),dtype=np.uint8)
    for t in range(1,frames):
        advance=np.r_[-np.inf,previous[:-1]]
        leap=np.r_[-np.inf,-np.inf,previous[:-2]]
        leap[~skip]=-np.inf
        candidates=np.stack([previous,advance,leap])
        choice=candidates.argmax(0);moves[t]=choice
        previous=candidates[choice,np.arange(len(states))]+log_probs[t,states]
    last=len(states)-1 if previous[-1]>=previous[-2] else len(states)-2
    score=float(previous[last])
    if not np.isfinite(score):raise ValueError('No finite alignment path')
    path=np.empty(frames,dtype=np.int64);path[-1]=last
    for t in range(frames-1,0,-1):path[t-1]=path[t]-int(moves[t,path[t]])
    spans=[]
    for i,token in enumerate(targets):
        indices=np.flatnonzero(path==2*i+1)
        if not len(indices):raise AssertionError('A target token was skipped')
        spans.append((int(token),int(indices[0]),int(indices[-1]+1)))
    return {'spans':spans,'frame_token_ids':states[path].tolist(),'state_path':path.tolist(),
            'log_probability':score}


def native_ot_spans(weights,targets):
    """Inverse-CDF transport to equal target masses, with fractional frames.

    Known token count and ordered labels are supplied; frame masses are learned
    from audio. No target duration, true boundary or tuned time shift is used.
    """
    weights=np.asarray(weights,dtype=np.float64)
    if weights.ndim!=1 or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('Expected a nonnegative frame measure')
    if not len(targets):raise ValueError('Expected nonempty target sequence')
    weights=weights/weights.sum();cumulative=weights.cumsum();cumulative[-1]=1.
    boundaries=[0.]
    for q in np.arange(1,len(targets))/len(targets):
        frame=int(np.searchsorted(cumulative,q,side='left'))
        before=cumulative[frame-1] if frame else 0.
        boundaries.append(frame+(q-before)/weights[frame])
    boundaries.append(float(len(weights)))
    return [(int(token),float(boundaries[i]),float(boundaries[i+1])) for i,token in enumerate(targets)]
