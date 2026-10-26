#!/usr/bin/env python3
"""Convert one or more aligned NumPy [T,K] outputs into Posterior Lab JSON.
Requires NumPy. Run --help for arguments. Does not upload anything.
"""
import argparse,json
from pathlib import Path
import numpy as np


def convert(args):
    phones=json.loads(Path(args.phones).read_text(encoding='utf8'))
    if not isinstance(phones,list) or len(phones)<2 or any(not isinstance(x,str) or not x for x in phones) or len(set(phones))!=len(phones):
        raise ValueError('phones.json must be an array of unique, non-empty labels.')
    methods=[];frames=None
    for label,filename in args.posterior:
        path=Path(filename)
        loaded=np.load(path,allow_pickle=False)
        if isinstance(loaded,np.lib.npyio.NpzFile):
            try:p=np.array(loaded[args.npz_key],dtype=np.float64)
            finally:loaded.close()
        else:p=np.asarray(loaded,dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=len(phones) or p.shape[0]<2:
            raise ValueError(f'{path}: expected [T,{len(phones)}], T>=2; got {p.shape}.')
        if frames is not None and p.shape[0]!=frames:raise ValueError('Methods must have identical time grids. Do not silently truncate or resample.')
        frames=p.shape[0]
        if not np.isfinite(p).all():raise ValueError(f'{path}: NaN or infinity.')
        if args.input_kind=='logits':
            p=np.exp(p-p.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        if (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=.001,rtol=0):
            raise ValueError(f'{path}: input is not a normalized probability matrix. Supply the full class set, or specify --input-kind logits.')
        # 9 decimal places keep normalization errors well below validation tolerance.
        methods.append({'id':label,'label':label,'posterior':np.round(p,9).tolist()})
    if len({m['id'] for m in methods})!=len(methods):raise ValueError('Method IDs must be unique.')
    if args.times:
        times=np.asarray(np.load(args.times,allow_pickle=False),dtype=float)
        if times.shape!=(frames,) or not np.isfinite(times).all() or (times<0).any() or not (np.diff(times)>0).all():raise ValueError('--times must be a strictly increasing, non-negative [T] array in milliseconds.')
    else:
        if args.frame_hop_ms<=0 or args.start_ms<0:raise ValueError('Use a positive frame hop and a non-negative start offset.')
        times=args.start_ms+np.arange(frames)*args.frame_hop_ms
    data={'schema_version':1,'source_type':args.source_type,'utterance_id':args.utterance_id,'text':args.text,'phones':phones,'time_ms':times.tolist(),'methods':methods}
    if args.segments:data['reference_segments']=json.loads(Path(args.segments).read_text(encoding='utf8'))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(f'Wrote {output}: {len(methods)} methods, {frames} frames, {len(phones)} phones. No upload was performed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--posterior',nargs=2,action='append',required=True,metavar=('METHOD','NPY_OR_NPZ'),help='Repeat once per method, e.g. --posterior ce ce.npy')
    p.add_argument('--phones',required=True,help='JSON array listing ALL columns in exact output order')
    p.add_argument('--npz-key',default='posterior')
    p.add_argument('--input-kind',choices=['probabilities','logits'],default='probabilities')
    p.add_argument('--times',help='Optional [T] .npy timestamps in ms; otherwise use frame hop')
    p.add_argument('--frame-hop-ms',type=float,default=10)
    p.add_argument('--start-ms',type=float,default=0)
    p.add_argument('--source-type',choices=['model','simulated'],default='model')
    p.add_argument('--utterance-id',required=True)
    p.add_argument('--text',default='')
    p.add_argument('--segments',help='Optional JSON reference segment list')
    p.add_argument('--output',default='posterior.json')
    args=p.parse_args()
    try:convert(args)
    except (ValueError,KeyError,OSError) as e:p.error(str(e))

if __name__=='__main__':main()
