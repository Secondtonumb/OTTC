"""Split saved speech inspections into recognition and supplied-text FA pages.

Reuses saved predictions only. Never converts a recognition path into an FA path.
Run: python -m reproduction.cif_comparison_20261016.segmentation_subpages
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproduction/segmentation_figures_20261017'
COLORS = {'CTC': '#475569', 'OTTC CE': '#277da8', 'OTTC upstream-JSD': '#248543',
          'OTTC upstream-JSD (α)': '#006d77', 'OTTC standard-JSD': '#8e6bbd', 'WavLM OTTC CE': '#14647d',
          'WavLM OTTC upstream-JSD': '#1b7c54', 'CIF CE+Quantity': '#d79015',
          'CIF + CTC': '#c84b31', 'CIF + OTTC': '#c15d9a',
          'Zapformer JSD (native 10ms)': '#646464',
          'Archived CTM (ID unverified)': '#0f766e', 'MFA 2.2.17 (word FA)': '#0369a1'}
OBJECTIVES = [('OTTC upstream-JSD', 'jsd_upstream'), ('OTTC CE', 'ce'),
              ('OTTC standard-JSD', 'jsd_standard')]


def main():
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    from reproduction.segmentation_metrics import _scorer, SOURCE

    sources, cache = {}, {}

    def track(p):
        p = Path(p)
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def load(p):
        return json.loads(track(p).read_text())

    def record(p, uid):
        p = Path(p)
        if p not in cache:
            cache[p] = {Path(r['utt']).stem: r for r in
                        (json.loads(s) for s in track(p).read_text().splitlines())}
        return cache[p][uid]

    base = load(OUT / 'examples.json')['examples']
    alpha_probe = ROOT / 'reproduction/ottc_alpha_decode_20261017'
    alpha_results = {}
    if (alpha_probe / 'SCORE_COMPLETE.json').exists():
        alpha_status = load(alpha_probe / 'SCORE_COMPLETE.json')
        assert alpha_status['status'] == 'complete' and alpha_status['utterances'] == 4
        alpha_results = {r['recording_id']: r for r in (json.loads(s) for s in track(alpha_probe / 'scored_predictions.jsonl').read_text().splitlines())}
        track(alpha_probe / 'DECODE_COMPLETE.json')
        track(alpha_probe / 'RESULTS.json')
    cif_probe = ROOT / 'reproduction/cif_fa_probe_20261017'
    cif_results = []
    if (cif_probe / 'EVAL_COMPLETE.json').exists():
        cif_status = load(cif_probe / 'EVAL_COMPLETE.json')
        assert cif_status['status'] == 'complete' and cif_status['records'] == 20
        cif_results = [json.loads(s) for s in track(cif_probe / 'predictions.jsonl').read_text().splitlines()]
        track(cif_probe / 'RESULTS.json')
    manifest = load(OUT / 'timestamp_manifest.json')
    track(SOURCE)
    # Exactly the same one-to-one matcher as the existing shared-reference plots.
    matcher_path = track(ROOT / 'reproduction/buckeye_singlehead_20261014/train.py')
    tree = ast.parse(matcher_path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boundary_matches')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(matcher_path), 'exec'), namespace)
    match = namespace['boundary_matches']
    frozen_scorer = ROOT / 'reproduction/english_suite/mfa_protocol_comparison/scorer_source/recompute_timit_raw_phn_tse_rvalue.py'
    track(frozen_scorer)
    track(ROOT / 'reproduction/mfa_direct_20261017/RESULTS_COMPLETE.json')
    template = track(Path(__file__).with_name('segmentation_explorer.html')).read_text()
    spec = importlib.util.spec_from_file_location('speech_exporter', track(OUT / 'export_segmentation_bundle.py'))
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    pages = {'recognition': {}, 'fa': {}}
    coverage = []

    def fa_model(ex, spans, protocol, clock):
        dur = ex['duration']
        assert all(0 <= a < b <= dur + 1e-8 for _, a, b in spans)
        assert all(a[2] <= b[1] + 1e-8 for a, b in zip(spans, spans[1:]))
        bounds = sorted({float(t) for _, a, b in spans for t in (a, b) if .02 < t < dur - .02})
        metrics = {}
        for ms in [10, 20, 25]:
            hits = match(ex['reference']['boundaries'], bounds, ms / 1000)
            metrics[str(ms)] = _scorer.finalize_counts(hits, hits, len(bounds), len(ex['reference']['boundaries']))
        return dict(phones=[p for p, _, _ in spans], starts=[a for _, a, _ in spans],
                    ends=[b for _, _, b in spans], boundaries=bounds, metrics=metrics,
                    text_free=False, input_protocol=protocol, alignment_input='supplied_phone_sequence', clock=clock)

    for rid, example in base.items():
        recognition, fa = copy.deepcopy(example), copy.deepcopy(example)
        recognition['models'] = {n: m for n, m in recognition['models'].items() if m['text_free']}
        recognition['auxiliary_note'] = '本页全部为音频独立预测；参考音素与时间仅用于展示和评分。'
        if rid in alpha_results:
            result = alpha_results[rid]
            spans = [list(x) for x in result['native_alpha_spans']]
            if spans[-1][2] < example['duration'] - 1e-7:
                spans.append(['<tail>', spans[-1][2], example['duration']])
            model = dict(phones=[p for p,_,_ in spans], starts=[a for _,a,_ in spans], ends=[b for _,_,b in spans],
                boundaries=sorted(set(t for t in result['native_alpha_boundaries'] if .02<t<example['duration']-.02)),
                metrics=result['metrics']['alpha_fractional'], text_free=True,
                boundary_method='alpha_quantile_fractional', alignment_input='audio_only_predicted_phone_sequence',
                input_protocol='预测α＋自身识别音素数量Û，β=1/Û，逆CDF帧内插值；不输入参考音素/数量/时间，不新增静音。识别序列与原argmax完全相同。',
                source_predictions=str(alpha_probe / 'scored_predictions.jsonl'), checkpoint=result['checkpoint'],
                checkpoint_sha256=result['checkpoint_sha256'],
                clock='fixed20ms frame cells, uniform within-frame alpha interpolation; final display tail unscored')
            ordered = {}
            for name, old in recognition['models'].items():
                ordered[name] = old
                if name == 'OTTC upstream-JSD': ordered['OTTC upstream-JSD (α)'] = model
            recognition['models'] = ordered
            recognition['auxiliary_note'] += ' α行固定原识别序列，使用预测质量重新分段；全句边界可能移动，分数帧不保证更准确。'
        fa['models'] = {}
        uid, split = example['original_id'], example['split']
        for label, objective in OBJECTIVES:
            if split == 'timit':
                directory = ROOT / 'reproduction/timit_singlehead_k2_fa_20261013/full' / objective
                path = directory / 'predictions.jsonl'
            else:
                dataset = 'buckeye' if split == 'buckeye' else 'libri100'
                directory = ROOT / f'reproduction/{dataset}_singlehead_20261014/evaluation' / objective
                name = 'test' if split == 'buckeye' else 'test_' + split.split('_')[1]
                path = directory / f'{name}_predictions.jsonl'
            row = record(path, uid)
            metadata = load(directory / 'metrics.json')
            assert row['reference_times_used_by_aligner'] is False
            assert abs(row['duration'] - example['duration']) < 1e-8
            expected = list(zip(example['reference']['phones'], example['reference']['starts'], example['reference']['ends']))
            assert len(row['reference']) == len(expected)
            for supplied, ref in zip(row['reference'], expected):
                assert supplied[0] == ref[0] and np.allclose(supplied[1:], ref[1:], rtol=0, atol=1e-8)
            aligned = row if split == 'timit' else row['forced_k2']
            assert 'failure' not in aligned
            spans = aligned['predicted_spans']
            assert [p for p, _, _ in spans] == example['reference']['phones']
            model = fa_model(example, spans,
                '给定参考音素序列的 K2 FA；包含语料既定的静音/音素映射；未输入参考时间；不是词转写 G2P 条件。',
                'saved K2 duration/frame_count grid; blank previous; no fitted shift')
            model.update(source_predictions=str(path), checkpoint=metadata['checkpoint'],
                         checkpoint_sha256=metadata['checkpoint_sha256'])
            if split == 'timit':
                # The FA evaluator verified replay of the exact recognition manifest.
                original = Path(metadata['original_prediction_manifest'])
                assert metadata['original_prediction_manifest_sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
                assert row['text_free_replay_identical'] is True
            fa['models'][label] = model
        zap = 'Zapformer JSD (native 10ms)'
        if split == 'timit':
            path = ROOT / 'reproduction/zapformer_timit_20261013/runs/jsd_seed42/final_test.predictions.jsonl'
            row = record(path, uid)
            boundaries = row['boundaries']['forced_k2']
            # The saved FA contains all successive supplied-token starts. Preserve
            # those points exactly; utterance endpoints complete the display only.
            data_path = ROOT / 'reproduction/zapformer_timit_20261013/data/test.jsonl'
            data = next(r for r in (json.loads(s) for s in track(data_path).read_text().splitlines()) if r['id'] == uid)
            assert row['reference'] == data['tokens']
            phones = [ref[0] for ref in data['references']]
            assert ['d' if p == 'dx' else p for p in phones] == example['reference']['phones']
            assert len(boundaries) + 1 == len(phones) == len(row['reference'])
            edges = [0.] + boundaries + [example['duration']]
            model = fa_model(example, list(zip(phones, edges, edges[1:])),
                '给定原生参考音素序列的 K2 FA，保留dx（共同参考及OTTC将dx映为d）；无参考时间输入。保存的内部起点原样保留；首尾显示边界为语音端点。',
                'native duration/frame_count (~10ms); saved internal K2 boundaries, endpoint display completion')
            model['source_predictions'] = str(path)
            fa['models'][zap] = model
            track(ROOT / 'reproduction/zapformer_timit_20261013/evaluate.py')
        for label, model in example['models'].items():
            if model['text_free'] is False:
                item = copy.deepcopy(model)
                item['alignment_input'] = 'word_transcript'
                if label.startswith('Archived'):
                    item['identity_unverified'] = True
                fa['models'][label] = item
        auxiliary_covered = set()
        for result in [r for r in cif_results if r['recording_id'] == rid]:
            if result.get('alignment_failed'):
                continue
            native = result['method'] == 'cif_length_conditioned'
            base_name = {'CIF CE+Quantity': 'CIF CE+Q', 'CIF + CTC': 'CIF+CTC', 'CIF + OTTC': 'CIF+OTTC'}[result['model_label']]
            name = base_name + (' (length)' if native else ' (K2 aux)')
            COLORS[name] = COLORS[result['model_label']]
            model = dict(phones=result['phones'], starts=result['pred_starts'], ends=result['pred_ends'],
                boundaries=result['pred_boundaries'], metrics=result['metrics'], text_free=False,
                clock=result['clock'], boundary_method=result['method'],
                alignment_input=result['alignment_input'], source_predictions=str(cif_probe / 'predictions.jsonl'),
                checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
                original_model_label=result['model_label'],
                input_protocol=('只给定参考音素数量的CIF条件分段，不用音素内容搜索；不是K2 FA。' if native else
                    'CIF模型encoder辅助分支的逐帧后验＋给定音素序列，经原有K2 CTC图对齐；不是CIF fire边界。'))
            fa['models'][name] = model
            if not native:
                auxiliary_covered.add(result['model_label'])
        fa['unavailable_models'] = [n for n in recognition['models'] if n not in fa['models'] and n not in auxiliary_covered and n != 'OTTC upstream-JSD (α)']
        fa['auxiliary_note'] = ('新增CIF：length行只用目标数量，K2 aux行用辅助头＋音素序列，两者分开。' if cif_results else '') + 'OTTC/Zapformer给定参考音素序列；MFA给定词转写。两种输入条件不同，请分别解释；未输入参考时间。Zapformer保留dx，OTTC/参考用d映射。归档CTM身份未核实，默认隐藏。'
        pages['recognition'][rid], pages['fa'][rid] = recognition, fa
        coverage.append(dict(recording_id=rid, recognition_models=list(recognition['models']),
                             fa_models=list(fa['models']), unavailable_fa=fa['unavailable_models']))

    footer = ('STFT：25ms Hann窗、5ms hop、512 FFT，−80至0dB。点击音素区间持续高亮并仅播放该区间，到终点暂停。'
              '分数为所选整句、统一边界协议，不随局部放大改变。仅四条示例，不能代表完整测试集。'
              'Libri参考是模型身份未核实的既有自动对齐。')
    metadata = {
        'recognition': dict(mode='recognition', filename='recognition.html', title='1 · Recognition / 无文本识别与分段',
            description='同一条音频：波形 → spectrogram → 参考音素 → 各模型音频独立预测的音素与分段。点击任一区间可高亮并只播放该区间。',
            protocol='推理只输入音频，不提供参考词、参考音素序列、参考音素数量或参考时间。原CTC/OTTC/WavLM/Zapformer使用自由预测路径；OTTC α行使用预测音素数与α累计质量的帧内插值，识别序列不变；CIF使用模型自行触发的音素区间。', footer=footer),
        'fa': dict(mode='fa', filename='forced_alignment.html', title='2 · FA / 强制对齐与 CIF 条件分段',
            description='同一条音频：波形 → spectrogram → 参考音素 → 已保存的强制对齐结果。点击任一区间可高亮并只播放该区间。',
            protocol='输入条件：CIF length = 仅给定音素数量（非K2 FA）；CIF K2 aux = 辅助头＋给定音素序列；OTTC/Zapformer = 音频＋给定参考音素序列；MFA = 音频＋原始词转写（词典生成发音）。均不输入参考时间。两种文本条件不同。归档CTM身份未核实，默认隐藏。',
            footer=footer+' FA不报告识别PER。图中FA F1/R-value使用共同可视化参考；历史raw-PHN指标另存，不混用。MFA工具2.2.17，声学模型/词典v3.0.0；不是论文MFA3.0复现。')}
    for mode, examples in pages.items():
        info = metadata[mode]
        payload = json.dumps(dict(page=info, examples=examples), ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
        (OUT / 'interactive' / info['filename']).write_text(template.replace('__PAYLOAD__', payload))
        (OUT / f'{mode}_examples.json').write_text(json.dumps(dict(page=info, examples=examples), ensure_ascii=False)+'\n')
        for rid, ex in examples.items():
            converted = {}
            for label, model in ex['models'].items():
                row = dict(recording_id=rid, duration=ex['duration'], phones=model['phones'],
                    pred_starts=model['starts'], pred_ends=model['ends'], pred_boundaries=model['boundaries'],
                    gt_phones=ex['reference']['phones'], gt_starts=ex['reference']['starts'],
                    gt_ends=ex['reference']['ends'], gt_boundaries=ex['reference']['boundaries'],
                    text_free=model['text_free'], input_protocol=model.get('input_protocol','audio-only recognition'),
                    metrics=model['metrics'])
                converted[label] = row
                directory = OUT / 'subpages' / mode / 'alignments' / ex['split'] / label.replace(' ', '_').replace('+','_')
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'alignments.jsonl').write_text(json.dumps(row)+'\n')
                (directory / 'summary.json').write_text(json.dumps(dict(complete=True, utterances=1, mode=mode,
                    scope='selected visualization example only', text_free=model['text_free'],
                    input_protocol=row['input_protocol'], clock=model['clock'],
                    boundary_protocol='unique interval edges, 20ms utterance-edge exclusion, one-to-one matching 10/20/25ms'),indent=2)+'\n')
            textgrid_path = OUT / 'subpages' / mode / 'textgrids' / ex['split'] / (rid+'.TextGrid')
            textgrid_path.parent.mkdir(parents=True, exist_ok=True)
            exporter.write_textgrid(textgrid_path, converted)
            audio, sr = sf.read(manifest[rid]['wav'])
            # Recompute the identical STFT for scientific full/zoom exports.
            frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio,(200,200)),400)[::80]
            power = np.abs(np.fft.rfft(frames*np.hanning(400),n=512,axis=-1)).T**2
            db = 10*np.log10(np.maximum(power,1e-12)/max(float(power.max()),1e-12))
            figdir = OUT / 'subpages' / mode / 'figures'
            figdir.mkdir(parents=True,exist_ok=True)
            for view,(lo,hi) in [('full',(0,ex['duration'])),('zoom',ex['zoom'])]:
                models = list(ex['models'].items())
                fig, axes = plt.subplots(3+len(models),1,figsize=(20,5+.62*len(models)),sharex=True,
                    gridspec_kw={'height_ratios':[.6,3,.8]+[.8]*len(models)})
                axes[0].plot(np.arange(len(audio))/sr,audio,linewidth=.4,color='#34445a')
                axes[0].set_ylabel('Waveform',rotation=0,ha='right',fontsize=9);axes[0].set_yticks([])
                axes[0].set_title(f'{ex["split"]} / {ex["original_id"]} — {"Recognition / audio-only" if mode=="recognition" else "Forced alignment / supplied text"} — {view}',fontsize=14)
                axes[1].imshow(db,origin='lower',aspect='auto',extent=[*ex['spectrogram_time_extent'],0,sr/2000],cmap='magma',vmin=-80,vmax=0)
                axes[1].set_ylabel('Frequency\n(kHz)',fontsize=10)
                tiers=[('Reference',ex['reference'])]+models
                for ax,(label,tier) in zip(axes[2:],tiers):
                    color=COLORS.get(label,'#111827')
                    for t in ex['reference']['boundaries']:ax.axvline(t,color='#707070',alpha=.35,linewidth=.7,linestyle='--')
                    for phone,a,b in zip(tier['phones'],tier['starts'],tier['ends']):
                        if b<=lo or a>=hi:continue
                        ax.axvspan(a,b,color=color,alpha=.12)
                        if (min(b,hi)-max(a,lo))/(hi-lo)>.009:ax.text((max(a,lo)+min(b,hi))/2,.5,phone,ha='center',va='center',fontsize=8)
                    for t in tier['boundaries']:ax.axvline(t,color=color,linewidth=1.1)
                    title=label
                    if tier.get('metrics'):
                        metric=tier['metrics']['20'];prefix=('Length F1' if tier.get('boundary_method')=='cif_length_conditioned' else 'FA F1') if mode=='fa' else 'F1'
                        title+=f'\n{prefix} {metric["f1_pct"]:.1f} / R-value {metric["r_value_pct"]:.1f}'
                    if mode=='fa' and label!='Reference':title+='\n'+('phone count only' if tier.get('alignment_input')=='reference_phone_count_only' else 'supplied phones' if tier.get('alignment_input')=='supplied_phone_sequence' else 'word transcript')
                    ax.set_ylim(0,1);ax.set_yticks([]);ax.set_ylabel(title,rotation=0,ha='right',va='center',fontsize=8)
                for ax in axes:ax.set_xlim(lo,hi);ax.spines[['top','right']].set_visible(False)
                axes[-1].set_xlabel('Audio time (seconds). Shared figure-reference scores; full selected utterance. '+('Supplied phones and word transcripts are different input conditions.' if mode=='fa' else 'No reference text or phone counts enter recognition.'))
                fig.subplots_adjust(left=.18,right=.99,hspace=.18,bottom=.065,top=.94)
                for ext in ['png','pdf']:fig.savefig(figdir/f'{rid}_{view}.{ext}',dpi=160)
                plt.close(fig)
            print('Exported',mode,rid,len(ex['models']),'tracks',flush=True)

    index = OUT / 'interactive/index.html'
    combined = OUT / 'interactive/combined.html'
    if not combined.exists():shutil.copy2(index,combined)
    index.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语音分段对照 · 结果总览</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#1b283c;font:16px/1.65 system-ui,sans-serif}main{max-width:1050px;margin:60px auto;padding:24px}h1{font-size:30px}p{color:#536078}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin:30px 0}a.card{display:block;padding:26px;background:white;border:1px solid #ccd6e3;border-radius:12px;text-decoration:none;color:inherit}a.card:hover,a.card:focus{border-color:#247594;box-shadow:0 4px 20px #1e3a5f15}h2{margin:0;font-size:22px}.tag{color:#247594;font-weight:600}.note{padding:16px;background:#eaf0f7;border-radius:8px}a{color:#247594}</style>
<main><h1>语音分段对照</h1><p>先选择结果类型。两个子页使用相同的四条语音和频谱，都支持点击区间高亮、只播放选中区间、缩放和平移；切换子页会保留语音选择。</p>
<div class="grid"><a class="card" href="recognition.html"><span class="tag">01 · Audio only</span><h2>Recognition / 无文本识别与分段</h2><p>CTC、OTTC、WavLM、三种CIF和Zapformer的自由预测结果；推理只输入音频。</p><span>打开识别结果 →</span></a>
<a class="card" href="forced_alignment.html"><span class="tag">02 · Supplied text</span><h2>FA / 强制对齐</h2><p>三种OTTC、TIMIT Zapformer、CIF辅助头的K2 FA与MFA词转写FA；另列三种CIF的数量条件分段。逐轨注明输入条件。</p><span>打开FA结果 →</span></a></div>
<p class="note">四条示例：TIMIT、Buckeye、Libri clean、Libri other各一条。示例不能代表完整测试集。FA页中的给定音素与词转写条件不同；未把无文本预测当作FA。暂无匹配FA输出的模型会在该页注明。</p><p>每页可独立离线打开。一起使用请下载完整包。<a href="combined.html">此前的混合对照页</a>仍保留。</p></main></html>''')
    track(Path(__file__))
    summary=dict(status='complete',new_inference=False,model_training=False,scope='same four preselected visualization examples',
        pages={m:{**metadata[m], 'examples':4,'models_per_example':{rid:len(ex['models']) for rid,ex in examples.items()}} for m,examples in pages.items()},
        coverage=coverage,source_sha256=sources,
        boundary_protocol='shared figure references, unique interval endpoints; exclude edges20ms; time-only one-to-one 10/20/25ms; frozen IF-MDD R formula',
        fa_inputs='OTTC and Zapformer supplied phone sequence; MFA word transcript; no reference timestamps to aligner',
        raw_PHN_separate='MFA direct RESULTS_COMPLETE retains historical raw-PHN score; current plots use figure references. No TSE/common-count comparison claimed.')
    (OUT/'SUBPAGES_COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name in ['segmentation_subpages.py','segmentation_explorer.html']:
        shutil.copy2(Path(__file__).with_name(name),OUT/'provenance'/name)
    print('COMPLETE',index,flush=True)


if __name__=='__main__':main()
