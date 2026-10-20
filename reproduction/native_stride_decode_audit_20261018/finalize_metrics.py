"""Validate completed standalone evaluations; record AMP/FP32 differences honestly."""
import json
import math
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent/'native_stride_ottc_20261018'


def main():
    for cost in ['jsd_upstream','ce','jsd_standard']:
        name=f'wav2vec2__timit__10ms__s3t2ss__{cost}'
        out=HERE/'results'/name;old=EXPERIMENT/'models'/name
        done=json.loads((out/'BENCHMARK_COMPLETE.json').read_text())
        selection=json.loads((out/'DEV_SELECTION.json').read_text())
        assert not selection['test_used'] and not done['test_used_for_selection']
        candidates=list(done['candidates']);assert len(candidates)==4
        dev={};comparisons={}
        for candidate in candidates:
            for split,expected in [('dev',400),('test',192)]:
                file=f'benchmark_{candidate}_{split}.json';value=json.loads((out/file).read_text())
                assert value['utterances']==expected and math.isfinite(value['raw_loss_per_utterance'])
                for mode,metrics in value['modes'].items():
                    assert metrics['utterances']==expected and metrics['alignment_failures']==0,(file,mode)
                predictions=out/file.replace('.json','.predictions.jsonl')
                ids=[json.loads(line)['utt'] for line in predictions.open()]
                assert len(ids)==len(set(ids))==expected
                previous=json.loads((old/file).read_text())
                comparisons[file]={mode:dict(old_PER=previous['modes'][mode]['nosil_PER_percent'],
                    standalone_FP32_PER=value['modes'][mode]['nosil_PER_percent']) for mode in ['text_free','fsa_text_free']}
                if split=='dev':dev[candidate]=value['modes']['fsa_text_free']['nosil_PER_percent']
        assert done['selected']==selection['selected']==min(candidates,key=dev.get)
        (out/'CORRECTION_VALIDATION.json').write_text(json.dumps(dict(
            all_forced_alignments_successful=True,unique_prediction_coverage_verified=True,
            dev_only_selection_verified=True,no_new_training=True,
            original_vs_standalone_FP32_metrics=comparisons,
            precision_note='Fresh FP32 model replaces training-end Trainer AMP-wrapped inference; differences are preserved, not suppressed.'),indent=2)+'\n')
        (out/'CORRECTED_EVALUATION_COMPLETE.json').write_text(json.dumps(dict(recipe=name,selected=done['selected'],
            all_FA_successful=True,training_restarted=False,wandb_artifact_status='pending_publication',
            validated_at_UTC=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        print(name,'validated',done['selected'])


if __name__=='__main__':main()
