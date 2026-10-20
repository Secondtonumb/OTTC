"""Bound padded activation size while preserving the optimizer batch and sampler."""
LIMITS = {'ctc': (16000, 24000000), 'ottc': (16000, 24000000),
          'spottc': (14000, 20000000), 'film_k4': (8000, 10000000)}

def plan(lengths, arm):
    linear, quadratic = LIMITS[arm]
    groups=[];group=[];longest=0
    for i,n in enumerate(lengths):
        n=int(n);assert n>0
        candidate=max(longest,n);size=len(group)+1
        if group and (size*candidate>linear or size*candidate*candidate>quadratic):
            groups.append(group);group=[];longest=0
        group.append(i);longest=max(longest,n)
    if group:groups.append(group)
    assert [i for group in groups for i in group]==list(range(len(lengths)))
    return groups

def tensors(cpu, indices):
    import torch
    x,lengths,targets,target_lengths=cpu
    ends=target_lengths.cumsum(0);starts=ends-target_lengths
    assert indices==list(range(indices[0],indices[-1]+1))
    first,last=indices[0],indices[-1]+1
    target=targets[int(starts[first]):int(ends[last-1])]
    longest=int(lengths[first:last].max())
    return x[first:last,:longest],lengths[first:last],target,target_lengths[first:last]
