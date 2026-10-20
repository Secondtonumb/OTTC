"""Rebuild the original external TIMIT split and frozen MFA mapping on local audio."""
import hashlib
import json
import sys
from pathlib import Path

BASE=Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa')
sys.path.insert(0,str(BASE/'experiments/mfa_zapformer'))
from evaluation_phone_mapping import map_intervals

OUT=Path(__file__).resolve().parent/'mfa_eval_data'
AUDIO=Path('/work/gm64/m64000/dataset/TIMIT/timit')
SPLITS=Path('/home/m64000/work/IF-MDD/data/timit_proccessed')
TOKENS=BASE/'outputs/mfa_corpus_audit_20260921/training_inventory/tokens.txt'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import torch
    import soundfile as sf
    from torchaudio.compliance import kaldi
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    inventory=dict(line.split() for line in TOKENS.read_text().splitlines())
    inventory={p:int(i) for p,i in inventory.items()}
    files={}
    for path in AUDIO.rglob('*.phn'):
        key=(path.parent.name+'_'+path.stem).lower()
        if key in files:raise ValueError('Duplicate '+key)
        files[key]=path
    roles={'development':['train','dev'],'test':['test']}
    speakers={};report={'splits':{},'tokens_sha256':sha(TOKENS),'mapping_source_sha256':sha(BASE/'experiments/mfa_zapformer/evaluation_phone_mapping.py'),'protocol_sha256':sha(BASE/'experiments/mfa_zapformer/evaluation_protocol.json'),'split_sha256':{},'feature_config':json.loads(Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json').read_text())['feature_config']}
    for role,parts in roles.items():
        rows=[];speakers[role]=set()
        with (OUT/(role+'.f32')).open('wb') as blob:
            offset=0
            for part in parts:
                source=SPLITS/(part+'_clean')/'utt2spk'
                report['split_sha256'][part]=sha(source)
                for line in source.read_text().splitlines():
                    utt,speaker=line.lower().split();speakers[role].add(speaker)
                    phn=files[utt];raw=[]
                    for interval in phn.read_text().splitlines():
                        start,end,label=interval.split();raw.append({'start':int(start)/16000,'end':int(end)/16000,'label':label})
                    mapped=map_intervals(raw,'timit',inventory)
                    assert all(r['mfa_id'] is not None for r in mapped)
                    wav=phn.with_suffix('.wav');audio,sr=sf.read(wav,dtype='float32')
                    assert sr==16000 and audio.ndim==1
                    feature=kaldi.fbank(torch.from_numpy(audio).unsqueeze(0)*32768,sample_frequency=16000,num_mel_bins=80,frame_shift=10.,frame_length=25.,snip_edges=False,dither=0.,energy_floor=0.)
                    feature.numpy().astype('<f4').tofile(blob)
                    rows.append({'id':'timit:'+utt,'wav':str(wav),'speaker':speaker,'source_split':part,'duration':len(audio)/16000,'feature_file':str((OUT/(role+'.f32')).resolve()),'feature_offset':offset,'frames':len(feature),'tokens':[r['mfa_id'] for r in mapped if r['score_speech']],'phone_intervals':mapped,'references':[(r['mfa_id'],r['start'],r['end']) for r in mapped if r['score_speech']],'gradient_training_allowed':False})
                    offset+=feature.numel()
                    if len(rows)%500==0:print(role,len(rows),flush=True)
        target=OUT/(role+'.jsonl');target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report['splits'][role]={'count':len(rows),'hours':sum(r['duration'] for r in rows)/3600,'speakers':len(speakers[role]),'manifest_sha256':sha(target),'features_sha256':sha(OUT/(role+'.f32'))}
    assert not speakers['development']&speakers['test']
    assert report['splits']['development']['count']==4096 and report['splits']['test']['count']==192
    (OUT/'PREPARE_COMPLETE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['splits']),flush=True)

if __name__=='__main__':main()
