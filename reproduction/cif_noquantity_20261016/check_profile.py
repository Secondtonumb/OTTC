"""Gate six exact-recipe profiles and exercise longest held-out inputs."""
import datetime
import json
import math
import os
from pathlib import Path

import torch

from ottc.models.sequence.wav2vec2_cif import wvForCIF
from reproduction.cif_noquantity_20261016.data import ROOT,load_rows,sha
from reproduction.cif_noquantity_20261016.evaluate import evaluate


def main():
    root=Path(os.environ['CIF_NQ_ROOT']);probe=Path(os.environ['CIF_NQ_PROBE']);report={}
    for dataset in ['buckeye','libri100']:
        for variant in ['ce','ctc_0p5','ottc_jsd_0p5']:
            p=probe/(dataset+'_'+variant);marker=json.loads((p/'PROBE_COMPLETE.json').read_text());config=json.loads((p/'run_config.json').read_text())
            assert marker['progress']['step']==4 and config['quantity_weight']==0
            h=[json.loads(l) for l in (p/'history.jsonl').read_text().splitlines()];h=[x for x in h if 'train_loss' in x]
            assert len(h)==4 and all(math.isfinite(x['train_loss']) and math.isfinite(x['gradient_norm']) for x in h)
            assert max(x['peak_allocated_GiB'] for x in h)<105
            assert max(x['max_audio_seconds'] for x in h)>(26 if dataset=='buckeye' else 24)
            assert all(json.loads((p/f'final_{s}.json').read_text())['utterances']==16 for s in ['dev','test'])
            report[dataset+'_'+variant]=dict(config=config,probe=str(p),
                peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in h),
                warm_batch_seconds=sum(x['batch_seconds'] for x in h[1:])/3,
                warm_audio_seconds_per_wall_second=sum(x['audio_seconds'] for x in h[1:])/sum(x['batch_seconds'] for x in h[1:]))
    # All variants share the exact same recognition architecture. Check the
    # longest dev/test bucket (including35.155s Libri) in the full encoder.
    for dataset in ['buckeye','libri100']:
        rows,vocab,_=load_rows(dataset)
        torch.manual_seed(42)
        model=wvForCIF.from_encoder_pretrained(ROOT/'reproduction/models/wav2vec2-large-lv60',
            quantity_weight=0).cuda().eval()
        cases=sorted(rows['dev']+rows['test'],key=lambda x:x['duration'],reverse=True)[:8]
        torch.cuda.reset_peak_memory_stats()
        metrics=evaluate(model,cases,vocab,probe/(dataset+'_longest_eval.json'),torch.bfloat16)
        assert metrics['utterances']==8 and math.isfinite(metrics['ce_per_phone'])
        report[dataset+'_longest_eval']={'max_seconds':cases[0]['duration'],'elapsed_seconds':metrics['elapsed_seconds'],
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
        del model;torch.cuda.empty_cache()
    result={'created_at':datetime.datetime.now().astimezone().isoformat(),'batch_size':64,'quantity_weight':0,'profiles':report}
    (root/'GPU_PROFILE_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL SIX NO-QUANTITY PROFILES PASS',flush=True)


if __name__=='__main__':main()
