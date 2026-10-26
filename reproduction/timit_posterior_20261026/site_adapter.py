"""Prepare Posterior Lab v1 data and an offline, single-scene browser viewer."""
import base64
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / 'site_integration'
METHODS = [('ce', 'ce', 'CE'), ('jsd_standard', 'jsd', 'JSD'),
           ('ag', 'ag-div', 'AG-DIV'), ('jdiv', 'j-div', 'J-DIV')]
FILENAME = 'timit_FADG0_SI1279_seed43.posterior-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def export_site_json(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    reference = json.loads((output / 'reference.json').read_text())
    methods = []
    sources = []
    for cost, key, label in METHODS:
        raw = json.loads((output / f'{cost}.json').read_text())
        assert raw['labels'] == manifest['labels']
        assert len(raw['posterior']) == 180
        assert len(raw['posterior'][0]) == 42
        methods.append(dict(id=key, label=label, posterior=raw['posterior']))
        member = next(m for m in manifest['methods'] if m['cost'] == cost)
        sources.append(dict(id=key, run=f'{cost}__seed43', epoch=member['epoch'], step=member['step'],
                            checkpoint_sha256=member['checkpoint_sha256'],
                            config_sha256=member['config_sha256'], selection=member['selection']))
    # Exact analytic CNN centers from receptive field 400, hop 160, sample rate 16000.
    # No interpolation, probability changes, or stretching to waveform duration.
    times = [12.46875 + i * 10 for i in range(180)]
    payload = dict(schema_version=1, source_type='model', utterance_id=manifest['utterance'],
                   text=manifest['text'], phones=manifest['labels'], time_ms=times, methods=methods,
                   reference_segments=[dict(start_ms=s*1000, end_ms=e*1000, phone=p)
                                       for p, s, e in reference['references']['ottc_arpabet']],
                   metadata=dict(dataset='TIMIT', split='dev', seed=43,
                                 experiment_family='timit_seed_sweep_20261022',
                                 frontend='wav2vec2-large-lv60; native10ms; monophone42',
                                 frame_hop_ms=10, time_convention='CNN receptive field centers',
                                 time_offset_ms=12.46875, sample_rate_hz=16000,
                                 receptive_field_samples=400, frame_hop_samples=160,
                                 historical_plot_clock='frame-start convention: subtract12.46875ms from time_ms',
                                 inference_dtype='float32', reference_conditioned_inference=False,
                                 reference_tier='mapped human PHN: ottc_arpabet, original exported intervals unchanged',
                                 reference_times_quantized=False, probability_transform='none; original FP32 softmax values',
                                 complete_vocabulary=True, subset_renormalized=False,
                                 methods=sources, omitted_methods=['KL: user requested completed matched models only'],
                                 audio_sha256=manifest['audio_sha256']))
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    assert '/work/' not in text and '/home/' not in text
    (output / FILENAME).write_text(text)
    # Compare every value with the original interchange file, without rounding.
    recovered = json.loads((output / FILENAME).read_text())
    for (cost, _, _), method in zip(METHODS, recovered['methods']):
        original = json.loads((output / f'{cost}.json').read_text())
        assert method['posterior'] == original['posterior']
        assert all(abs(t - center*1000) < .001
                   for t, center in zip(times, original['receptive_center_seconds']))
    for name in ['posterior-format.md', 'posterior.schema.json', 'convert_posterior.py']:
        (output / name).write_bytes((CONTRACT / name).read_bytes())
    return payload


def standalone_viewer(output, payload):
    html = (CONTRACT / 'site_index.html').read_text()
    # Drop the hosting provider's challenge injection from the local page.
    html = html.split('<script>(function(){function c()')[0]
    html = html.replace('</body></html>', '')
    html = re.sub(r'<script[^>]*src="assets/[^\"]+"[^>]*></script>', '', html)
    css = (CONTRACT / 'assets/style.css').read_text()
    css += '\n#overview{margin-top:24px;padding:20px;background:white;border:1px solid #e0e7ee;border-radius:14px}'
    css += '\n#overview img{display:block;width:100%;height:auto;max-width:100%}#overview h2{font-size:20px}'
    css += '\n#render-status{padding:12px;color:#216c57}audio{max-width:100%;margin-top:12px}'
    html = html.replace('<link rel="stylesheet" href="assets/style.css">', f'<style>{css}</style>')
    html = html.replace('正在加载模拟示例…', '正在读取真实 TIMIT 输出…').replace('恢复模拟示例', '恢复本句真实数据')
    html = html.replace('SIMULATED', 'MODEL OUTPUT').replace('当前为模拟数据', '真实模型输出')
    html = html.replace('>模拟数据<', '>真实模型输出<')
    html = html.replace('href="example-posterior.json"', f'href="{FILENAME}"').replace('下载示例 JSON', '下载本句真实 JSON')
    html = html.replace('完整模拟示例', '本句真实数据')
    audio = base64.b64encode((output / 'audio.wav').read_bytes()).decode()
    html = html.replace('<section class="visual-section"', f'<audio controls src="data:audio/wav;base64,{audio}"></audio><section class="visual-section"')
    overview = base64.b64encode((output / 'posterior_3d_comparison.png').read_bytes()).decode()
    html = html.replace('<details class="contract"', '<section id="overview"><h2>四模型同视角总览</h2>'
                        '<p>上方切换模型并旋转；下方始终保留 CE / JSD / AG-DIV / J-DIV 四模型对比。总览使用历史帧起点时间轴，交互图使用感受野中心时间轴，相差12.46875ms。</p>'
                        f'<img id="overview-image" src="data:image/png;base64,{overview}" alt="CE JSD AG-DIV J-DIV real TIMIT posterior comparison">'
                        '</section><details class="contract"')
    html = html.replace('</footer>', '</footer><p id="render-status" role="status">正在渲染…</p>')
    validator = (CONTRACT / 'assets/format.js').read_text().replace('export ', '')
    app = (CONTRACT / 'assets/app.js').read_text()
    app = re.sub(r'^import .*?;\n', '', app, count=1)
    app = app.replace("async function demo(){try{const response=await fetch('example-posterior.json');if(!response.ok)throw new Error(`加载示例失败：${response.status}`);setData(await response.json());$('file').value=''}catch(e){showError(e)}}",
                      "async function demo(){try{setData(JSON.parse(document.getElementById('bundled-dataset').textContent));$('file').value=''}catch(e){showError(e)}}")
    assert "fetch(" not in app
    app = app.replace("methodIndex=Math.max(0,d.methods.findIndex(m=>m.id==='ag-div'||m.id==='ag'));", 'methodIndex=0;')
    # The page has one live GL scene, while the four-way overview is a normal image.
    # There is no lower WebGL row or shared, fixed-height multi-scene clipping region.
    app = app.replace("if(version===renderVersion){$('reset-camera').disabled=heat;timeline();}",
                      "if(version===renderVersion){$('reset-camera').disabled=heat;timeline();"
                      "$('render-status').textContent='已渲染 '+m.label+' · '+(heat?'2D 热图':'3D 视图')+' · 下方四模型总览可直接查看';"
                      "window.posteriorRenderReady=true;window.dispatchEvent(new Event('posterior-rendered'));}")
    fallback = """
    window.addEventListener('error',e=>{const s=document.getElementById('render-status');if(s)s.textContent='渲染错误: '+e.message;});
    function supportsGL(){const c=document.createElement('canvas');try{const gl=c.getContext('webgl2')||c.getContext('webgl');if(!gl)return false;const lose=gl.getExtension('WEBGL_lose_context');if(lose)lose.loseContext();return true}catch(e){return false}}
    if(!supportsGL())document.getElementById('view').value='heatmap';
    """
    selftest = """
    if(location.hash==='#selftest'){
      let started=false;
      window.addEventListener('posterior-rendered',async()=>{
        if(started)return;started=true;
        const report={method_renders:[],overview_visible:false,full42:false};
        try{
          for(let mi=0;mi<dataset.methods.length;mi++){methodIndex=mi;await render();report.method_renders.push(dataset.methods[mi].id)}
          $('top-k').value='all';$('view').value='heatmap';await render();report.full42=$('plot').data[0].z.length===42;
          const img=$('overview-image');await img.decode();report.overview_visible=img.naturalWidth>0&&img.getBoundingClientRect().height>0;
          methodIndex=0;$('method').value='0';$('top-k').value='12';await render();
          report.pass=report.method_renders.length===4&&report.full42&&report.overview_visible;
          window.posteriorSelftest=report;document.documentElement.dataset.selftest=report.pass?'passed':'failed';
          $('render-status').textContent='SELFTEST '+(report.pass?'PASSED':'FAILED')+' '+JSON.stringify(report);
          if(typeof dump==='function')dump('POSTERIOR_SELFTEST '+JSON.stringify(report)+'\\n');
        }catch(e){report.error=String(e);window.posteriorSelftest=report;if(typeof dump==='function')dump('POSTERIOR_SELFTEST '+JSON.stringify(report)+'\\n');showError(e)}
      });
    }
    """
    plotly = (CONTRACT / 'assets/plotly.min.js').read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    html += f'<script id="bundled-dataset" type="application/json">{encoded}</script>'
    html += '<script>' + plotly.replace('</script', '<\\/script') + '</script>'
    html += '<script>' + validator + fallback + selftest + app + '</script></body></html>'
    assert 'type="module"' not in html and 'src="assets/' not in html
    (output / 'posterior_3d_interactive.html').write_text(html)
    return dict(renderer='single Plotly scene with four selectable methods plus embedded four-model overview',
                offline=True, embedded_audio=True, public_site_modified=False,
                source_site='https://phonetic-posterior-lab-kg.kevingenghaopeng.chatgpt.site/',
                source_contract_sha256=sha(CONTRACT / 'posterior-format.md'),
                source_validator_sha256=sha(CONTRACT / 'assets/format.js'))


def build(output):
    output = Path(output)
    payload = export_site_json(output)
    viewer = standalone_viewer(output, payload)
    write(output / 'SITE_INTEGRATION.json', dict(schema_version=1, file=FILENAME,
          methods=[m['id'] for m in payload['methods']], shape=[4,180,42], posterior_values_unchanged=True,
          time_offset_ms=12.46875, private_server_paths=False, viewer=viewer,
          file_sha256=sha(output / FILENAME), adapter_sha256=sha(__file__)))
    print('SITE_EXPORT_OK', output / FILENAME, flush=True)


if __name__ == '__main__':
    build(HERE / 'seed43/FADG0_SI1279')
