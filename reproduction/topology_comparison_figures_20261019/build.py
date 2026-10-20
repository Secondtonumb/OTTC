"""Re-score saved TIMIT paths and extend the earlier segmentation explorer.

CPU only; immutable model/prediction inputs; no training or inference.
Run from repository root: python -m reproduction.topology_comparison_figures_20261019.build
"""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / 'reproduction/native_stride_decode_audit_20261018'
OLD = ROOT / 'reproduction/segmentation_figures_20261017'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from reproduction.segmentation_metrics import _scorer, SOURCE

COSTS = ['jsd_upstream', 'ce', 'jsd_standard']
NAMES = {'jsd_upstream': 'upstream-JSD', 'ce': 'CE', 'jsd_standard': 'standard-JSD'}
COLORS = {'mono': '#187d8f', 's3t2ss': '#db723a'}
HASHES = {}
CACHE = {}


def track(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load(path):
    return json.loads(track(path).read_text())


def records(path):
    path = Path(path)
    if path not in CACHE:
        values = [json.loads(s) for s in track(path).read_text().splitlines()]
        CACHE[path] = {Path(r.get('utt', r.get('id'))).stem: r for r in values}
        assert len(values) == len(CACHE[path]) == 192, path
    return CACHE[path]


def function(path, name):
    tree = ast.parse(track(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


MATCH = function(ROOT / 'reproduction/libri100_singlehead_20261014/train.py', 'boundary_matches')
track(SOURCE)


def boundaries(reference, predicted, duration):
    rb = sorted({float(t) for _, s, e in reference for t in (s, e) if .02 < t < duration - .02})
    pb = sorted({float(t) for t in predicted if .02 < t < duration - .02})
    return {str(ms): _scorer.finalize_counts(MATCH(rb, pb, ms / 1000), MATCH(rb, pb, ms / 1000), len(pb), len(rb))
            for ms in [10, 20, 25, 30]}


def aggregate(values):
    result = {}
    for ms in ['10', '20', '25', '30']:
        c = {k: sum(v[ms][k] for v in values) for k in ['matches', 'predicted_boundaries', 'reference_boundaries']}
        result[ms] = _scorer.finalize_counts(c['matches'], c['matches'], c['predicted_boundaries'], c['reference_boundaries'])
    return result


def points(row, head, mode):
    if mode == 'fa':
        return [t for _, a, b in row['forced_alignment']['predicted_spans'] for t in (a, b)]
    if head == 's3t2ss':
        path = row['fsa_text_free']
        return [round(a * len(path['frame_token_ids']) / row['duration']) * .01 for _, a, _ in path['predicted_spans'][1:]]
    ids = np.array(row['text_free']['frame_token_ids'])
    ids[ids == 41] = 0
    return ((np.flatnonzero(ids[1:] != ids[:-1]) + 1) * .01).tolist()


def plot_save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def overview(selection, recognition):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [('Audio-only boundary F1 @20ms', 'nonFA', 'f1_pct'),
              ('Supplied-phone FA F1 @20ms', 'FA', 'f1_pct'),
              ('FA timestamp MAE (ms; lower is better)', 'MAE', None),
              ('Audio-only R-value @20ms', 'nonFA', 'r_value_pct'),
              ('Supplied-phone FA R-value @20ms', 'FA', 'r_value_pct'),
              ('Audio-only PER (%; lower is better)', 'PER', None)]
    for ax, (title, mode, key) in zip(axes.flat, panels):
        for offset, head in [(-.19, 'mono'), (.19, 's3t2ss')]:
            vals = []
            for cost in COSTS:
                recipe = f'wav2vec2__timit__10ms__{head}__{cost}'
                exp = selection[recipe]
                if mode == 'PER':
                    value = next(r['test_PER'] for r in recognition if r['recipe'] == recipe)
                elif mode == 'MAE':
                    value = exp['dev_MAE_selected_FA']['test']['boundary_MAE_ms']
                else:
                    test = exp['dev_R_selected_' + ('nonFA' if mode == 'nonFA' else 'FA')]['test']
                    metric = test if mode == 'nonFA' else test['boundaries']['20']
                    value = metric[{'f1_pct': 'f1_percent', 'r_value_pct': 'r_value_percent'}[key]]
                vals.append(value)
            bars = ax.bar(np.arange(3) + offset, vals, width=.36, color=COLORS[head], label='Mono' if head == 'mono' else 'S3T2** / K2')
            ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(3), [NAMES[c] for c in COSTS], fontsize=9)
        ax.set_ylim(0, max(b.get_height() for b in ax.patches) * 1.19)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, .98), frameon=False)
    fig.suptitle('TIMIT core-test 192 | Native 10 ms Wav2Vec2 OTTC', y=1.015, fontsize=15)
    fig.text(.5, -.015, 'Dev-only selection: boundary panels by dev R-value; MAE by dev MAE; PER by dev PER.\nAudio-only: fixed 10 ms clock. FA: saved duration/frame-count clock. Each panel uses its own stated checkpoint selection.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .94))
    plot_save(fig, 'current_six_overview')


def bootstrap_plot(bootstrap):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, mode, key, title in zip(axes, ['nonFA_dev_R', 'FA_dev_R', 'FA_dev_MAE'], ['F1', 'F1', 'MAE_ms'],
                                  ['Audio-only F1 change (pp)', 'Supplied-phone FA F1 change (pp)', 'FA MAE change (ms)']):
        for i, cost in enumerate(COSTS):
            v = bootstrap['experiments'][cost][mode][key]
            ax.errorbar(v['delta'], i, xerr=[[v['delta']-v['CI95_low']], [v['CI95_high']-v['delta']]], fmt='o', capsize=5, color=COLORS['s3t2ss'], markersize=7)
            ax.annotate(f"{v['delta']:+.2f} [{v['CI95_low']:+.2f}, {v['CI95_high']:+.2f}]", (v['delta'], i), xytext=(0, 13), textcoords='offset points', ha='center', fontsize=9)
        ax.axvline(0, color='#475569', linewidth=1, linestyle='--')
        ax.set_yticks(range(3), [NAMES[c] for c in COSTS]);ax.set_ylim(2.55, -.6)
        ax.set_title(title, fontsize=11);ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='x', alpha=.2)
        ax.set_xlim((-13, 2) if key == 'F1' else (-2, 17))
    fig.suptitle('S3T2** minus Mono | Paired test-utterance bootstrap, 95% CI', fontsize=14)
    fig.text(.5, -.02, '4,000 paired resamples; 192 utterances; fixed dev-selected checkpoints. Sampling uncertainty only; no multi-seed claim.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94));plot_save(fig, 'paired_changes')


def main():
    (OUT / 'figures').mkdir(exist_ok=True)
    (OUT / 'interactive/audio').mkdir(parents=True, exist_ok=True)
    selection = load(AUDIT / 'BOUNDARY_SELECTION_CHECK.json')['experiments']
    recognition = load(AUDIT / 'RESULTS.json')['rows']
    bootstrap = load(AUDIT / 'PAIRED_BOOTSTRAP.json')
    gt = records(Path(recognition[0]['predictions']))
    vocab = load(ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json')
    inverse = {v: k for k, v in vocab.items()}
    assert inverse[0] == '<pad>' and inverse[41] == '<unk>'
    tree = ast.parse(track(ROOT / 'ottc/tools/dataset_loader/timit.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'timit_phoneme_fusion_map' for t in n.targets))
    mapping = dict(ast.literal_eval(assignment.value), dx='d')
    silence = {'sil', '<sil>', 'sp', 'spn', 'pau', 'h#', 'epi', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q'}
    def normalized(phones):
        return [mapping.get(p.lower().rstrip('012'), p.lower().rstrip('012')) for p in phones if p.lower().rstrip('012') not in silence]
    table, per_utterance = [], []
    current_sources = {}
    for recipe, choices in selection.items():
        head, cost = recipe.split('__')[-2:]
        for mode in ['recognition', 'fa']:
            choice = choices['dev_R_selected_' + ('nonFA' if mode == 'recognition' else 'FA')]['selected']
            path = AUDIT / 'results' / recipe / f'benchmark_{choice}_test.predictions.jsonl'
            rows = records(path);assert rows.keys() == gt.keys()
            current_sources[recipe, mode] = (path, rows, choice)
            scored = []
            for uid, row in rows.items():
                assert row['reference'] == gt[uid]['reference']
                assert row['reference_times_used_by_aligner'] is False
                score = boundaries(gt[uid]['reference'], points(row, head, mode), row['duration'])
                scored.append(score)
                per_utterance.append(dict(group='current', recipe=recipe, mode=mode, candidate=choice, utt=uid, boundaries=score))
            totals = aggregate(scored)
            expected = choices['dev_R_selected_' + ('nonFA' if mode == 'recognition' else 'FA')]['test']
            if mode == 'fa':expected = expected['boundaries']['20']
            for ours, theirs in [('f1_pct','f1_percent'),('r_value_pct','r_value_percent')]:assert abs(totals['20'][ours]-expected[theirs]) < 1e-5
            table.append(dict(group='current_native10', model=f"10ms {'Mono' if head=='mono' else 'S3/K2'} / {NAMES[cost]}", recipe=recipe, mode=mode,
                              selected=choice, selection='dev boundary R-value@20ms', utterances=192, boundaries=totals, source=str(path)))
    print('Verified all 12 current full-set boundary evaluations', flush=True)

    # Extend the historical all-version table using exactly the current reference.
    cif_active = load(ROOT / 'reproduction/cif_timit_20261016/active_runs.json')
    history = [('CTC', 'old', ROOT / 'reproduction/results/runexp_ctc_fuse_phoneme/timit_core_test_192.jsonl'),
               ('OTTC upstream-JSD', 'old', ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_jsd/timit_core_test_192.jsonl'),
               ('OTTC CE', 'old', ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme/timit_core_test_192.jsonl'),
               ('OTTC standard-JSD', 'old', ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_jsd_standard/timit_core_test_192.jsonl'),
               ('WavLM OTTC upstream-JSD', 'old', ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013/timit_core_test_192.jsonl'),
               ('WavLM OTTC CE', 'old', ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_ce_wavlm_b128_20261013/timit_core_test_192.jsonl')]
    history += [(label, 'cif', Path(cif_active[k]) / 'final_test.predictions.jsonl') for label, k in [('CIF CE+Quantity','no_ctc'), ('CIF + CTC','ctc_0p5'), ('CIF + OTTC','ottc_jsd_0p5')]]
    history += [('OTTC upstream-JSD (alpha)', 'alpha', ROOT / 'reproduction/ottc_alpha_full_20261017/timit_predictions.jsonl'),
                ('Zapformer upstream-JSD', 'zap', ROOT / 'reproduction/zapformer_timit_20261013/runs/jsd_seed42/final_test.predictions.jsonl')]
    for label, kind, path in history:
        rows = records(path);assert rows.keys() == gt.keys()
        scores, errors, nphones = [], 0, 0
        for uid, row in rows.items():
            reference = gt[uid]['reference'];hyp = row['hypothesis']
            if kind == 'old':
                assert row['reference'] == reference
                pb = row['predicted_boundaries']
            elif kind == 'cif':pb = [t*.02*row['frame_count']/row['duration'] for t in row['boundaries']['text_free']]
            elif kind == 'alpha':pb = row['native_alpha_boundaries']
            else:pb = row['boundaries']['text_free'];hyp = [inverse[i] for i in hyp]
            score = boundaries(reference, pb, gt[uid]['duration']);scores.append(score)
            refphones = normalized([p for p,_,_ in reference]);errors += Levenshtein.distance(refphones, normalized(hyp));nphones += len(refphones)
            per_utterance.append(dict(group='historical', model=label, mode='recognition', utt=uid, boundaries=score))
        table.append(dict(group='historical', model=label, mode='recognition', utterances=192, PER=100*errors/nphones,
                          clock='duration/frame_count' if kind == 'zap' else 'fixed20ms; fractional crossings retained',
                          selection='original historical checkpoint; unchanged', source=str(path), boundaries=aggregate(scores)))
    print('Re-scored 11 historical versions against the same 192 references', flush=True)
    overview(selection, recognition);bootstrap_plot(bootstrap)
    free = [r for r in table if r['mode'] == 'recognition']
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    y = np.arange(len(free))
    for ax, key, title in zip(axes, ['f1_pct','r_value_pct'], ['Boundary F1 @20ms (%)', 'R-value @20ms (%)']):
        vals = [r['boundaries']['20'][key] for r in free]
        cs = [('#b4bbc6' if r['group']=='historical' else COLORS['s3t2ss' if '__s3t2ss__' in r['recipe'] else 'mono']) for r in free]
        bars = ax.barh(y, vals, color=cs, height=.67);ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
        ax.set_yticks(y, [r['model'] + (' [historical]' if r['group']=='historical' else '') for r in free], fontsize=10)
        ax.set_xlim(0,100);ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.axhline(5.5,color='#64748b',linewidth=1,linestyle='--')
    axes[0].invert_yaxis()
    fig.suptitle('TIMIT core-test 192 | All saved versions, common reference and scorer', fontsize=15)
    fig.text(.5, -.005, 'Current 10 ms: Libri960-parent transfer, 10 epochs, dev-R selection. Historical: different training/selection; 20 ms except native Zapformer.\nThis is a benchmark comparison across recipes, not a controlled hop-size/backbone/loss ablation. Audio-only results only.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.96));plot_save(fig,'historical_and_current')

    # Same fixed example as the earlier figure; historical lanes initially hidden.
    example_id = 'timit__FDHC0_SI1559';uid = 'FDHC0_SI1559'
    exporter_path = track(OLD / 'export_segmentation_bundle.py')
    spec = importlib.util.spec_from_file_location('existing_exporter', exporter_path)
    exporter = importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    template = track(ROOT / 'reproduction/cif_comparison_20261016/segmentation_explorer.html').read_text()
    template = template.replace("c.checked=!current.models[name].identity_unverified", "c.checked=!current.models[name].identity_unverified&&!current.models[name].default_hidden")
    template = template.replace("colors[n]||", "current.models[n].color||colors[n]||")
    examples = {}
    import soundfile as sf
    wav = track(OLD / 'selected_audio' / (example_id + '.wav'))
    audio, sr = sf.read(wav)
    shutil.copy2(wav, OUT / 'interactive/audio' / wav.name)
    converted = {'recognition': {}, 'fa': {}}
    for mode in ['recognition', 'fa']:
        ex = copy.deepcopy(load(OLD / (('recognition' if mode=='recognition' else 'fa')+'_examples.json'))['examples'][example_id])
        historical = ex['models'];ex['models'] = {};ex.pop('unavailable_models', None)
        for oldname, model in historical.items():
            model['default_hidden'] = True
        for cost in COSTS:
            for head in ['mono','s3t2ss']:
                recipe = f'wav2vec2__timit__10ms__{head}__{cost}'
                path, rows, choice = current_sources[recipe, mode];row = rows[uid]
                assert np.allclose(np.array([[s,e] for _,s,e in row['reference']]), np.array(list(zip(ex['reference']['starts'],ex['reference']['ends']))),rtol=0,atol=1e-8)
                if mode == 'fa':spans = copy.deepcopy(row['forced_alignment']['predicted_spans'])
                elif head == 's3t2ss':
                    p = row['fsa_text_free'];frames=len(p['frame_token_ids'])
                    spans = [[ph,round(a*frames/row['duration'])*.01,round(b*frames/row['duration'])*.01] for ph,a,b in p['predicted_spans']]
                else:
                    ids=np.array(row['text_free']['frame_token_ids']);ids[ids==41]=0
                    edges=[0]+(np.flatnonzero(ids[1:]!=ids[:-1])+1).tolist()+[len(ids)]
                    spans=[['<blk>' if ids[a]==0 else inverse[ids[a]],a*.01,b*.01] for a,b in zip(edges,edges[1:])]
                if spans[-1][2] < ex['duration']-1e-7:spans.append(['<tail>',spans[-1][2],ex['duration']])
                spans=[[p,max(0.,a),min(ex['duration'],b)] for p,a,b in spans if a<ex['duration'] and b>a]
                pb=sorted(set(t for t in points(row,head,mode) if .02<t<row['duration']-.02))
                label=f"10ms {'Mono' if head=='mono' else 'S3/K2'} / {NAMES[cost]}"
                model=dict(phones=[p for p,_,_ in spans], starts=[a for _,a,_ in spans], ends=[b for _,_,b in spans], boundaries=pb,
                           metrics=boundaries(row['reference'],pb,row['duration']), text_free=mode=='recognition',color=COLORS[head],
                           clock='fixed native10ms' if mode=='recognition' else 'duration/frame_count; blank previous',
                           input_protocol='只输入音频；S3使用原IF-MDD K2 phone-loop。' if mode=='recognition' else '给定参考音素序列，不输入参考时间；K2 FA。',
                           candidate=choice,source_predictions=str(path),selection='dev boundary R-value@20ms')
                ex['models'][label]=model
        ex['models'].update({'Historical / '+n:m for n,m in historical.items()})
        ex['auxiliary_note']='默认显示本次六个10ms版本；勾选Historical轨可与之前版本对照。当前权重按dev边界R选择。整句例子不是全测试集结论。'
        page=dict(mode=mode, title='TIMIT 10ms topology / '+('Recognition 无文本分段' if mode=='recognition' else 'FA 给定音素对齐'),
                  protocol='当前六组完整core-test192统计见总览；图示沿用此前FDHC0_SI1559。当前与历史训练、时钟和选择策略不同。FA、数量条件、词转写输入分别注明。',
                  footer='灰虚线＝参考边界。无文本当前固定10ms、历史20ms（Zapformer原生）；FA保留各自duration/frames时钟、blank归前音素。10/20/25/30ms一对一边界评分，20ms端点排除。未拟合时间偏移。无新训练或推理。归档CTM身份未核实，默认隐藏。')
        data=dict(page=page,examples={example_id:ex})
        examples[mode]=data
        (OUT/f'{mode}_examples.json').write_text(json.dumps(data,ensure_ascii=False)+'\n')
        filename='recognition.html' if mode=='recognition' else 'forced_alignment.html'
        (OUT/'interactive'/filename).write_text(template.replace('__PAYLOAD__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
        for label,m in ex['models'].items():
            converted[mode][label]=dict(recording_id=example_id,duration=ex['duration'], phones=m['phones'],pred_starts=m['starts'],pred_ends=m['ends'],pred_boundaries=m['boundaries'],
                                       gt_phones=ex['reference']['phones'],gt_starts=ex['reference']['starts'],gt_ends=ex['reference']['ends'],gt_boundaries=ex['reference']['boundaries'],metrics=m['metrics'])
        exporter.write_textgrid(OUT/'textgrids'/mode/(example_id+'.TextGrid'),converted[mode])
        # Export all current alignments, with exact scored points distinct from display tails.
        for label,m in list(ex['models'].items())[:6]:
            recipe=next(r['recipe'] for r in table if r['model']==label and r['mode']==mode)
            _, rows, choice=current_sources[recipe,mode]
            path=OUT/'alignments'/mode/recipe/'alignments.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(converted[mode][label])+'\n')
            path.with_name('summary.json').write_text(json.dumps(dict(complete=True,utterances=1,scope='fixed visualization example only; complete192 scoring separately',candidate=choice,clock=m['clock'],mode=mode),indent=2)+'\n')
        dbframes=np.lib.stride_tricks.sliding_window_view(np.pad(audio,(200,200)),400)[::80]
        power=np.abs(np.fft.rfft(dbframes*np.hanning(400),n=512,axis=-1)).T**2
        db=10*np.log10(np.maximum(power,1e-12)/max(float(power.max()),1e-12))
        for view,(lo,hi) in [('full',(0,ex['duration'])),('zoom',ex['zoom'])]:
            tiers=[('Reference',ex['reference'])]+list(ex['models'].items())[:6]
            fig,axes=plt.subplots(2+len(tiers),1,figsize=(17,9),sharex=True,gridspec_kw={'height_ratios':[.65,2.5]+[.7]*len(tiers)})
            axes[0].plot(np.arange(len(audio))/sr,audio,lw=.4,color='#475569');axes[0].set_yticks([]);axes[0].set_ylabel('Waveform',rotation=0,ha='right')
            axes[1].imshow(db,origin='lower',aspect='auto',extent=[*ex['spectrogram_time_extent'],0,sr/2000],vmin=-80,vmax=0,cmap='magma');axes[1].set_ylabel('kHz')
            for ax,(label,m) in zip(axes[2:],tiers):
                c=m.get('color','#111827')
                for t in ex['reference']['boundaries']:ax.axvline(t,ls='--',lw=.6,color='#64748b',alpha=.4)
                for ph,a,b in zip(m['phones'],m['starts'],m['ends']):
                    if b<=lo or a>=hi:continue
                    ax.axvspan(a,b,alpha=.1,color=c)
                    if (min(b,hi)-max(a,lo))/(hi-lo)>.015:ax.text((max(a,lo)+min(b,hi))/2,.5,ph,ha='center',va='center',fontsize=8)
                for t in m['boundaries']:ax.axvline(t,color=c,lw=1.1)
                ylabel=label
                if m.get('metrics'):ylabel+=f"\nF1 {m['metrics']['20']['f1_pct']:.1f} / R {m['metrics']['20']['r_value_pct']:.1f}"
                ax.set_ylabel(ylabel,rotation=0,ha='right',va='center',fontsize=9);ax.set_yticks([]);ax.set_ylim(0,1)
            for ax in axes:ax.set_xlim(lo,hi);ax.spines[['top','right']].set_visible(False)
            axes[0].set_title(f'FDHC0_SI1559 | {mode} | fixed earlier example | {view}',fontsize=14)
            axes[-1].set_xlabel('Audio time (seconds). Dashed gray: reference; colored: scored model boundaries. F1/R describe this full utterance only.')
            fig.subplots_adjust(left=.215,right=.99,hspace=.16,top=.95,bottom=.07)
            plot_save(fig,f'example_{mode}_{view}')

    report=dict(status='complete',evaluation_only=True,new_inference=False,new_training=False,utterances=192,reference_boundaries=7149,
                boundary_protocol='shared references; unique endpoints; exclude edges20ms; sorted one-to-one10/20/25/30ms; frozen IF-MDD R-value',
                current_selection='dev boundary R for F1/R; dev MAE for MAE; dev PER for PER; four predetermined candidates, no test selection',
                historical_warning='Same scoring/reference, distinct recipes and clocks; not controlled ablation. Historical CIF uses historical dev-selected checkpoints.',
                rows=table,recognition_selected=recognition,bootstrap=bootstrap)
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'per_utterance_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_utterance))
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['group','model','mode','selected','PER_recognition_selected','P20','Recall20','F1_20','R_value20','source']);writer.writeheader()
        for r in table:
            per=next((v['test_PER'] for v in recognition if v['recipe']==r.get('recipe')),r.get('PER'))
            b=r['boundaries']['20'];writer.writerow(dict(group=r['group'],model=r['model'],mode=r['mode'],selected=r.get('selected','historical'),PER_recognition_selected=per if r['mode']=='recognition' else '',
                P20=b['precision_pct'],Recall20=b['recall_pct'],F1_20=b['f1_pct'],R_value20=b['r_value_pct'],source=r['source']))
    rows_html=''.join(f"<tr><td>{r['model']}</td><td>{r['boundaries']['20']['precision_pct']:.2f}</td><td>{r['boundaries']['20']['recall_pct']:.2f}</td><td>{r['boundaries']['20']['f1_pct']:.2f}</td><td>{r['boundaries']['20']['r_value_pct']:.2f}</td></tr>" for r in free)
    (OUT/'interactive/index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TIMIT topology 多版本比较</title><style>body{max-width:1200px;margin:36px auto;padding:20px;font:16px/1.7 system-ui;color:#243449;background:#f8fafc}img{width:100%;background:white}a{color:#087c90}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #dde3e9;text-align:right}td:first-child{text-align:left}.note{background:#e5eff5;padding:16px}h1{font-size:28px}</style><h1>TIMIT：10ms 单音素 / S3T2** 与历史版本</h1><p class="note">全量统计：core-test 192句。无文本边界与FA边界分开；每项权重只用dev选择。历史模型采用各自原训练配方和原时钟，只作跨配方比较。</p><p><a href="recognition.html">播放无文本分段对照 →</a>　<a href="forced_alignment.html">播放FA对照 →</a>　<a href="../metrics.csv">下载全量比较CSV</a>　<a href="../REPORT_ZH.md">评分说明</a></p><p>逐句页默认显示本次六组；勾选Historical可看此前CTC、CIF、WavLM、OTTC、Zapformer等轨。沿用之前FDHC0_SI1559，示例不代替全量统计。</p><img src="../figures/current_six_overview.png" alt="当前六组指标"><img src="../figures/paired_changes.png" alt="差值与置信区间"><img src="../figures/historical_and_current.png" alt="历史与当前版本"><h2>无文本边界 @20ms 完整测试集</h2><table><tr><th>版本</th><th>P%</th><th>Recall%</th><th>F1%</th><th>R-value%</th></tr>'''+rows_html+'</table><p>置信区间是固定dev所选模型下的逐句bootstrap，不包含训练seed和模型选择的不确定性。</p></html>')
    lines=['# TIMIT topology 多版本图与统一复算','', '复用保存预测，未训练、未推理；完整core-test192，7149个参考边界。当前六组12项（无文本/FA）结果精确复算通过。历史11版本重新对同一参考评分。', '',
           '打开 interactive/index.html 看汇总；recognition.html / forced_alignment.html 可听、缩放、选择区间。默认显示本次六组，历史轨手动勾选。示例沿用先前按ID/时长挑选的 FDHC0_SI1559，不按模型好坏挑选。', '',
           '协议：唯一边界点、排除音频首尾20ms、排序一对一10/20/25/30ms容差、冻结IF-MDD R-value。非FA本次统一固定10ms；FA保留duration/frame_count与blank归前音素；旧20ms与CIF分数帧保留，Zapformer保留原生时钟。没有拟合偏移。', '',
           '选择：边界F1/R按dev边界R选；FA MAE按dev MAE选；PER按dev PER选。候选仅best-dev/final/final5/final10，测试集不参与选点。CSV中的PER来自识别所选权重，与边界所选权重可能不同，勿视为同一checkpoint。', '',
           '历史模型：TIMIT40轮旧配方、不同冻结/学习率/batch/backbone；当前10ms是同一Libri960 parent再迁移TIMIT10轮。相同参考与评分只保证benchmark口径，不能由此把差值归因于hop/topology/loss单一因素。CIF使用此前dev所选权重，未重选、未重训。', '',
           'FA图的历史length行只给定数量，aux行是辅助头K2；MFA是词转写输入；其余OTTC是给定音素。参考时间只用于评分。归档CTM身份未核实，不称为论文MFA3.0；本次没有重算MFA raw-PHN/TSE或声称超越论文MFA。', '',
           '图与Praat仅导出这一条固定示例：recognition17模型轨，FA17模型轨。alignments/每文件1句；全量复算的metrics.json、CSV、per_utterance_metrics.jsonl分别标明192句范围。', '',
           '可复现：`python -m reproduction.topology_comparison_figures_20261019.build`。source_manifest.json记录输入及源码SHA256。paired_changes图复用既有4000次配对bootstrap。', '', '|当前目标|无文本Mono/S3 F1|FA Mono/S3 F1|FA Mono/S3 MAE ms|','|---|---:|---:|---:|']
    for c in COSTS:
        a=selection[f'wav2vec2__timit__10ms__mono__{c}'];b=selection[f'wav2vec2__timit__10ms__s3t2ss__{c}']
        lines.append(f"|{NAMES[c]}|{a['dev_R_selected_nonFA']['test']['f1_percent']:.2f}/{b['dev_R_selected_nonFA']['test']['f1_percent']:.2f}|{a['dev_R_selected_FA']['test']['boundaries']['20']['f1_percent']:.2f}/{b['dev_R_selected_FA']['test']['boundaries']['20']['f1_percent']:.2f}|{a['dev_MAE_selected_FA']['test']['boundary_MAE_ms']:.2f}/{b['dev_MAE_selected_FA']['test']['boundary_MAE_ms']:.2f}|")
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    track(Path(__file__))
    (OUT/'source_manifest.json').write_text(json.dumps(HASHES,indent=2)+'\n')
    (OUT/'bundle_summary.json').write_text(json.dumps(dict(status='complete',new_inference=False,new_training=False,full_set_utterances=192,full_set_model_modes=len(table),
        selected_utterances=[example_id],selected_by='reuse exact previous example, independent of current predictions',
        interactive_tracks={mode:len(d['examples'][example_id]['models']) for mode,d in examples.items()},textgrids=2,copied_audio=1,
        historical_MFA='display-only reused saved example; word-input and phone-input differ; archive identity unverified'),indent=2)+'\n')
    print('COMPLETE',OUT/'interactive/index.html',flush=True)


if __name__ == '__main__':
    main()
