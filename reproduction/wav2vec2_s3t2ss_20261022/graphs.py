"""Use the frozen IF-MDD S3T2** arcs for audio-only and supplied-phone paths."""
import torch
import k2


def graph(topo, vocab, supplied=None, device='cuda'):
    # K2 builds mutable FSA tensors even when its caller uses inference_mode.
    with torch.inference_mode(False),torch.no_grad():
        return _graph(topo,vocab,supplied,device)


def _graph(topo, vocab, supplied=None, device='cuda'):
    lines = []
    if supplied is None:
        text = topo['token_fst_text'](topo['phones'], 's3_t2_star_star', split_silence=False, blank_label='<blank>')
        labels = {label:i for i,label in enumerate(topo['labels'])}; maximum = 0
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 1: continue
            src,dst = map(int,parts[:2]); maximum=max(maximum,src,dst)
            lines.append((src,dst,labels[parts[2]],0 if parts[3]=='<eps>' else vocab[parts[3]]))
        final, super_final = 0,maximum+1
    else:
        n,arcs,_,final = topo['build_arcs'](supplied,topo['mapping'],'s3_t2_star_star',blank_index=0,
            repeat_separator='optional_blank',silence_self_loops='none')
        boundary=0; next_free=1; entries={}
        for occurrence,phone in enumerate(supplied,1):
            entries[(boundary,topo['mapping'][phone][0])]=occurrence
            boundary=next_free+len(topo['mapping'][phone]); next_free=boundary+1
        assert boundary==final
        lines=[(a.src,a.dst,a.label,entries.get((a.src,a.label),0)) for a in arcs]
        super_final=n
    lines.append((final,super_final,-1,-1))
    text='\n'.join(f'{s} {d} {i} {o} 0.0' for s,d,i,o in sorted(lines))+f'\n{super_final}\n'
    fsa=k2.arc_sort(k2.Fsa.from_str(text,acceptor=False).to(device))
    values=fsa.arcs.values(); fsa.original_src=values[:,0].clone(); fsa.original_dst=values[:,1].clone()
    return k2.Fsa.from_fsas([fsa])


def decode(scores, vector):
    with torch.inference_mode(False),torch.no_grad():
        scores=scores.detach().clone()
        dense=k2.DenseFsaVec(scores.log_softmax(-1).unsqueeze(0),torch.tensor([[0,0,len(scores)]],dtype=torch.int32))
        lattice=k2.intersect_dense(vector,dense,output_beam=8.,frame_idx_name='frame_idx')
        best=k2.shortest_path(lattice,use_double_scores=True)
        path=[]; outputs=[]; starts=[]
        for label,output,frame in zip(best.labels.tolist(),best.aux_labels.tolist(),best.frame_idx.tolist()):
            if label==-1: continue
            assert frame==len(path); path.append(label)
            if output>0: outputs.append(output); starts.append(frame)
        if len(path)!=len(scores): raise ValueError('No complete S3T2** path')
    return outputs,path,starts


def spans_from_starts(tokens, starts, duration, frames):
    return [[p,0. if i==0 else starts[i]*duration/frames,
             starts[i+1]*duration/frames if i+1<len(starts) else duration] for i,p in enumerate(tokens)]
