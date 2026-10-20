"""Inventory the 24 authorized experiment slots and submit independent jobs."""
from common import HERE,ROOT,write,sha,recipe
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone


def matrix():
    names=[]
    for head in ['mono','s3t2ss']:
        for cost in ['ce','jsd_upstream','jsd_standard']:
            names.append(f'wav2vec2__timit__10ms__{head}__{cost}')
    for corpus in ['timit','buckeye','libri100']:
        for frame in ['10ms','20ms']:
            for cost in ['ce','jsd_upstream','jsd_standard']:
                names.append(f'wavlm__{corpus}__{frame}__mono__{cost}')
    return names


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    parser.add_argument('--submit-libri-after-review',action='store_true');args=parser.parse_args()
    path=HERE/'dispatch.json';state=json.loads(path.read_text()) if path.exists() else dict(experiments={})
    state.update(updated_at_UTC=datetime.now(timezone.utc).isoformat(),authorized_slots=24,
        default_job_walltime='02:00:00',
        costs=['OTTC transport CE','original upstream JSD','standard JSD'],upsample=False,
        caveat='Wav2Vec2 TIMIT pair is a 10-epoch transfer test from the same Libri960 JSD parent; WavLM is a 40-epoch official-SSL initialization study. Do not mix their initialization or training histories.')
    # Old TIMIT targets match, but actual audio sample lengths differ for 4452
    # of 4620 examples. Preserve that model as a historical reference, rather
    # than silently presenting it as this batch's matched control.
    complete=ROOT/'reproduction/results/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013/TRAIN_COMPLETE.json'
    old=ROOT/'reproduction/models/large_models_results/wavlm-large/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013'
    cfg=json.loads((old/'run_config.json').read_text());done=json.loads(complete.read_text())
    assert done['epoch']==40 and cfg['batch_size']==64 and cfg['bins_metric']=='jsd'
    assert cfg['num_train_epochs']==40 and cfg['learning_rate']==2e-4 and cfg['frozen_epochs_for_regfrozen']==30
    state['reuse_audit']=dict(historical_JSD_configuration=cfg,
        existing_CE='WavLM 20ms TIMIT CE used batch128, not this batch64 control; preserve it as an additional historical run. New batch64 CE is a separately identified matched-batch experiment.',
        historical_selection='Old 20ms JSD uses fixed final epoch40, old kaldi-test240 monitoring and different raw-audio lengths; new runs use the frozen IF-MDD audio manifests and dev400 selection. This is a distinct recipe, not a restart to retrofit averaging.',
        actual_data_match_audit=str(HERE/'DATA_MATCH_AUDIT.json'))
    state['historical_references']={
        'wavlm_timit20_JSD':dict(checkpoint=str(old/'checkpoint-2920'),training_complete=done,used_as_matched_control=False)}
    libri_allowed=False
    if args.submit_libri_after_review:
        review=json.loads((HERE/'LIBRI_GATE.json').read_text())
        assert review['passed'] and review['numerical_checks_passed'] and review['recognition_reasonable'] and review['segmentation_reasonable']
        expected=[name for name in matrix() if '__libri100__' not in name]
        assert set(review['reviewed_benchmarks'])==set(expected)
        for name in expected:
            marker=HERE/'models'/name/'BENCHMARK_COMPLETE.json'
            assert marker.exists() and sha(marker)==review['reviewed_benchmarks'][name]
        libri_allowed=True
    for name in matrix():
        record=state['experiments'].setdefault(name,dict(status='prepared',recipe=recipe(name)))
        if not record.get('job'):record['recipe']=recipe(name)
        if record.get('status')=='reused_completed_training':
            for key in ['checkpoint','checkpoint_sha256','training_complete','results','averaging_available']:record.pop(key,None)
            record['status']='prepared_distinct_audio_manifest'
        if '__libri100__' in name and not libri_allowed:
            assert not record.get('job'),'Libri100 must not already be submitted before the result gate'
            record.update(status='held_until_TIMIT_and_Buckeye_results_reviewed',gate='LIBRI_GATE.json: full training/evaluation and outcome review required; profiling alone is insufficient')
            continue
        if (args.submit or libri_allowed) and not record.get('job'):
            log=HERE/(name+'.log')
            result=subprocess.run(['/opt/pbs/bin/qsub','-l','walltime=02:00:00','-v','NATIVE_OTTC_RECIPE='+name,'-o',str(log),str(HERE/'run_one.pbs')],capture_output=True,text=True)
            if result.returncode:
                record.update(status='submission_failed',error=result.stderr.strip());write(path,state)
                raise RuntimeError(name+': '+result.stderr)
            record.update(status='submitted_profile_gate_then_training',job=result.stdout.strip(),log=str(log),walltime='02:00:00')
        write(path,state)
    write(path,state)
    print(json.dumps({name:record['status'] for name,record in state['experiments'].items()},indent=2))


if __name__=='__main__': main()
