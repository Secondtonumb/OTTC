"""Paired test-utterance bootstrap, conditional on fixed dev-selected models."""
import json
from pathlib import Path
import numpy as np
from summarize import boundary_helpers

HERE=Path(__file__).resolve().parent


def records(name,candidate,mode,helpers):
    rows={}
    for line in (HERE/'results'/name/f'benchmark_{candidate}_test.predictions.jsonl').open():
        row=json.loads(line);ref=row['reference'];duration=row['duration']
        counts=helpers['new_counts']();error=0.;tokens=0
        if mode=='FA':
            spans=row['forced_alignment']['predicted_spans']
            predicted=[t for _,s,e in spans for t in (s,e)]
            tokens=len(ref)
            error=sum(abs(s-ps)+abs(e-pe) for (_,s,e),(_,ps,pe) in zip(ref,spans))
        elif '__s3t2ss__' in name:
            path=row['fsa_text_free'];frames=len(path['frame_token_ids'])
            starts=[round(s*frames/duration) for _,s,_ in path['predicted_spans']]
            predicted=[start*.01 for start in starts[1:]]
        else:
            path=np.asarray(row['text_free']['frame_token_ids']);path[path==41]=0
            predicted=((np.flatnonzero(path[1:]!=path[:-1])+1)*.01).tolist()
        helpers['score'](counts,ref,predicted,duration);b=counts['bounds']['20']
        rows[row['utt']]=[b['matches'],b['reference'],b['predicted'],error,tokens]
    return rows


def metrics(values,helpers):
    matches,reference,predicted,error,tokens=values.T
    precision=100*matches/predicted;recall=100*matches/reference
    return dict(F1=200*matches/(reference+predicted),
        R_value=np.array([helpers['r_value_percent'](p,r) for p,r in zip(precision,recall)]),
        MAE_ms=np.divide(500*error,tokens,out=np.zeros_like(error),where=tokens>0))


def main():
    selection=json.loads((HERE/'BOUNDARY_SELECTION_CHECK.json').read_text())['experiments']
    helpers=boundary_helpers();replicates=4000;seed=20261018;rng=np.random.default_rng(seed)
    result=dict(method='Paired utterance bootstrap, 4000 replicates; percentile95% CI; fixed dev-selected candidates; delta=S3-minus-mono; does not include training-seed/model-selection uncertainty',
                seed=seed,replicates=replicates,experiments={})
    for cost in ['jsd_upstream','ce','jsd_standard']:
        mono=f'wav2vec2__timit__10ms__mono__{cost}';sub=f'wav2vec2__timit__10ms__s3t2ss__{cost}';output={}
        for title,key,mode in [('nonFA_dev_R','dev_R_selected_nonFA','nonFA'),
                               ('FA_dev_R','dev_R_selected_FA','FA'),('FA_dev_MAE','dev_MAE_selected_FA','FA')]:
            left=records(mono,selection[mono][key]['selected'],mode,helpers)
            right=records(sub,selection[sub][key]['selected'],mode,helpers)
            ids=sorted(left);assert set(ids)==set(right) and len(ids)==192
            x=np.asarray([left[i] for i in ids],dtype=float);y=np.asarray([right[i] for i in ids],dtype=float)
            indices=rng.integers(0,len(ids),size=(replicates,len(ids)))
            bx=metrics(x[indices].sum(axis=1),helpers);by=metrics(y[indices].sum(axis=1),helpers)
            px=metrics(x.sum(axis=0,keepdims=True),helpers);py=metrics(y.sum(axis=0,keepdims=True),helpers)
            output[title]={}
            for key in ['F1','R_value']+(['MAE_ms'] if mode=='FA' else []):
                delta=by[key]-bx[key]
                output[title][key]=dict(delta=float(py[key][0]-px[key][0]),
                    CI95_low=float(np.percentile(delta,2.5)),CI95_high=float(np.percentile(delta,97.5)))
        result['experiments'][cost]=output
    (HERE/'PAIRED_BOOTSTRAP.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
