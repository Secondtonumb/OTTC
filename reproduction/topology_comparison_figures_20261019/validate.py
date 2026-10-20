"""Validate the exported evidence and actual offline browser interactions."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

HERE=Path(__file__).resolve().parent
sources=json.loads((HERE/'source_manifest.json').read_text())
for p,expected in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected,p
metrics=json.loads((HERE/'metrics.json').read_text())
assert len(metrics['rows'])==23 and metrics['utterances']==192
assert len((HERE/'per_utterance_metrics.jsonl').read_text().splitlines())==23*192
for r in metrics['rows']:
    for b in r['boundaries'].values():assert b['reference_boundaries']==7149
for mode in ['recognition','fa']:
    e=json.loads((HERE/f'{mode}_examples.json').read_text())['examples']['timit__FDHC0_SI1559']
    assert len(e['models'])==17
    for m in e['models'].values():
        assert len(m['phones'])==len(m['starts'])==len(m['ends'])
        assert all(0<=a<b<=e['duration']+1e-6 for a,b in zip(m['starts'],m['ends']))
    tg=HERE/'textgrids'/mode/'timit__FDHC0_SI1559.TextGrid'
    text=tg.read_text();assert 'Object class = "TextGrid"' in text and 'size = 36' in text
assert len(list((HERE/'interactive/audio').glob('*.wav')))==1
print('Data, hashes, coverage, TextGrid and intervals verified',flush=True)

profile=Path('/tmp/ottc_topology_figures_firefox');profile.mkdir(exist_ok=True)
(profile/'user.js').write_text('user_pref("marionette.port", 28319);\nuser_pref("media.autoplay.default", 0);\n')
env=dict(os.environ,XDG_CACHE_HOME=str(HERE/'browser_cache'),MOZ_CRASHREPORTER_DISABLE='1',
         MOZ_DISABLE_CONTENT_SANDBOX='1',MOZ_DISABLE_GMP_SANDBOX='1',MOZ_WEBRENDER='0',LIBGL_ALWAYS_SOFTWARE='1')
log=(HERE/'browser.log').open('w')
proc=subprocess.Popen(['firefox','--headless','--offline','--no-remote','--profile',str(profile),'--marionette'],stdout=log,stderr=log,env=env)
connection=None
try:
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        try:connection=socket.create_connection(('127.0.0.1',28319),timeout=1);break
        except OSError:time.sleep(.2)
    assert connection is not None,'Firefox did not start'
    connection.settimeout(30);stream=connection.makefile('rb')
    def receive():
        prefix=b''
        while not prefix.endswith(b':'):
            chunk=stream.read(1)
            if not chunk:raise RuntimeError('Browser connection closed')
            prefix+=chunk
        return json.loads(stream.read(int(prefix[:-1])))
    receive();serial=0
    def command(name,params):
        global serial
        serial+=1;data=json.dumps([0,serial,name,params]).encode();connection.sendall(str(len(data)).encode()+b':'+data)
        result=receive()
        if result[2]:raise RuntimeError(result[2])
        return result[3].get('value',result[3]) if isinstance(result[3],dict) else result[3]
    command('WebDriver:NewSession',{'capabilities':{'alwaysMatch':{'acceptInsecureCerts':True}}})
    result={}
    for mode,filename in [('recognition','recognition.html'),('fa','forced_alignment.html')]:
        command('WebDriver:Navigate',{'url':(HERE/'interactive'/filename).as_uri()})
        script=r'''
const done=arguments[arguments.length-1], mode=arguments[0];
(async()=>{
 const payload=JSON.parse(document.getElementById('data').textContent),ex=Object.values(payload.examples)[0];
 const a=document.getElementById('audio');a.muted=true;
 if(a.readyState<1)await new Promise(r=>a.addEventListener('loadedmetadata',r,{once:true}));
 if(payload.page.mode!==mode||a.duration<3.39||a.duration>3.40)throw Error('Wrong mode/audio');
 const boxes=[...document.querySelectorAll('#models input')];
 if(boxes.length!==17||boxes.filter(c=>c.checked).length!==6)throw Error('Wrong default tracks');
 const tier='10ms S3/K2 / upstream-JSD',m=ex.models[tier];
 if(m.text_free!==(mode==='recognition'))throw Error('FA/recognition condition mixed');
 const i=m.starts.findIndex((s,j)=>s>.5&&s<1.5&&m.ends[j]-s>.06);
 if(i<0)throw Error('No interval for playback check');
 document.querySelector(`.interval[data-tier="${tier}"][data-index="${i}"]`).dispatchEvent(new MouseEvent('click',{bubbles:true}));
 let played=false,stop=performance.now()+3500;
 while(performance.now()<stop){played=played||!a.paused;if(played&&a.paused&&Math.abs(a.currentTime-m.ends[i])<.004)break;await new Promise(r=>setTimeout(r,10));}
 if(!played||!a.paused||Math.abs(a.currentTime-m.ends[i])>=.004)throw Error('Interval playback did not stop correctly');
 document.getElementById('detail').click();
 if(document.querySelector('.interval.selected')?.dataset.tier!==tier)throw Error('Zoom lost selection');
 const old=[...document.querySelectorAll('#models label')].find(l=>l.textContent.startsWith('Historical / OTTC upstream-JSD'));
 old.querySelector('input').click();
 if(!document.querySelector('.interval[data-tier="'+old.textContent+'"]'))throw Error('Historical checkbox failed');
 document.getElementById('clear-selection').click();
 if(document.querySelector('.interval.selected'))throw Error('Clear failed');
 return {ok:true,mode,audio_duration:a.duration,models:boxes.length,default_models:6,interval_playback:true,zoom:true,historical_toggle:true};
})().then(done,e=>done({ok:false,error:String(e),stack:e.stack}));
'''
        value=command('WebDriver:ExecuteAsyncScript',{'script':script,'args':[mode],'newSandbox':False});assert value['ok'],value
        result[mode]=value
        screen=command('WebDriver:TakeScreenshot',{'id':None,'highlights':[],'full':False,'scroll':False})
        (HERE/f'{mode}_browser_preview.png').write_bytes(base64.b64decode(screen))
    command('Marionette:Quit',{'flags':['eForceQuit']})
    result.update(source_hashes_verified=len(sources),scored_model_modes=23,scored_records=23*192,textgrids=2)
    (HERE/'VALIDATION_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
finally:
    if connection:connection.close()
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill()
    log.close()
