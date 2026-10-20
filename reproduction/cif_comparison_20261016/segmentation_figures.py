"""Spectrograms and text-free segmentations from frozen saved predictions.

No model inference. Examples are selected by duration and ID only.
"""
import ast
import base64
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproduction/segmentation_figures_20261017'
SKILL = Path('/work/gm64/m64000/.codex_local/skills/export-speech-segmentation')


def main():
    OUT.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
    import numpy as np
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reproduction.segmentation_metrics import _scorer, SOURCE

    hashes, source_sizes = {}, {}

    def track(p):
        p = Path(p)
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def load(p):
        return json.loads(track(p).read_text())

    def records(p):
        rows = [json.loads(s) for s in track(p).read_text().splitlines()]
        keyed = {Path(r.get('utt', r.get('id'))).stem: r for r in rows}
        assert len(keyed) == len(rows)
        source_sizes[str(p)] = len(rows)
        return keyed

    tree = ast.parse(track(ROOT / 'reproduction/buckeye_singlehead_20261014/train.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boundary_matches')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'frozen_boundary_matcher', 'exec'), ns)
    match = ns['boundary_matches']
    track(SOURCE)
    timit_cif = load(ROOT / 'reproduction/cif_timit_20261016/active_runs.json')
    english_cif = load(ROOT / 'reproduction/cif_quantity_20261016/active_runs.json')
    manifest, examples, specs = {}, {}, []
    primary = [('CTC', 'runexp_ctc_fuse_phoneme'), ('OTTC CE', 'runexp_ottc_fuse_phoneme'),
               ('OTTC upstream-JSD', 'runexp_ottc_fuse_phoneme_jsd'),
               ('OTTC standard-JSD', 'runexp_ottc_fuse_phoneme_jsd_standard'),
               ('WavLM OTTC CE', 'runexp_ottc_fuse_phoneme_ce_wavlm_b128_20261013'),
               ('WavLM OTTC upstream-JSD', 'runexp_ottc_fuse_phoneme_jsd_wavlm_20261013')]
    colors = {'CTC': '#475569', 'OTTC CE': '#277da8', 'OTTC upstream-JSD': '#248543',
              'OTTC standard-JSD': '#8e6bbd', 'WavLM OTTC CE': '#14647d',
              'WavLM OTTC upstream-JSD': '#1b7c54', 'CIF CE+Quantity': '#d79015',
              'CIF + CTC': '#c84b31', 'CIF + OTTC': '#c15d9a',
              'Zapformer JSD (native 10ms)': '#646464',
              'Archived CTM (ID unverified)': '#0f766e', 'MFA 2.2.17 (word FA)': '#0369a1'}
    for split in ['timit', 'buckeye', 'libri_clean', 'libri_other']:
        predictions = {}
        if split == 'timit':
            for label, suffix in primary:
                predictions[label] = ('spans', records(ROOT / f'reproduction/results/{suffix}/timit_core_test_192.jsonl'))
            baseline = predictions['OTTC upstream-JSD'][1]
            for label, key in [('CIF CE+Quantity', 'no_ctc'), ('CIF + CTC', 'ctc_0p5'), ('CIF + OTTC', 'ottc_jsd_0p5')]:
                predictions[label] = ('cif', records(Path(timit_cif[key]) / 'final_test.predictions.jsonl'))
            # This supplementary lane retains its native duration/T clock.
            predictions['Zapformer JSD (native 10ms)'] = ('zap', records(ROOT / 'reproduction/zapformer_timit_20261013/runs/jsd_seed42/final_test.predictions.jsonl'))
            verification = load(ROOT / 'reproduction/english_suite/mfa_protocol_comparison/MFA_ROW_VERIFIED.json')
            raw_ctm = track(Path('/work/gm64/m64000/IF-MDD/TIMIT_MFA/ctm.phone.timit.test'))
            assert hashes[str(raw_ctm)] == verification['raw_ctm_sha256']
            archive = records(Path('/work/gm64/m64000/IF-MDD/fa_research/results/timit_mfa_ctm_baseline/alignments.jsonl'))
            predictions['Archived CTM (ID unverified)'] = ('archive', {k: r for k, r in archive.items() if k in baseline})
            track(ROOT / 'reproduction/english_suite/mfa_protocol_comparison/scorer_source/recompute_timit_raw_phn_tse_rvalue.py')
        else:
            dataset = 'buckeye' if split == 'buckeye' else 'libri100'
            test = 'test' if split == 'buckeye' else 'test_' + split.split('_')[1]
            for label, objective in [('OTTC CE', 'ce'), ('OTTC upstream-JSD', 'jsd_upstream'), ('OTTC standard-JSD', 'jsd_standard')]:
                d = ROOT / f'reproduction/{dataset}_singlehead_20261014/evaluation/{objective}'
                m = load(d / 'metrics.json')
                vocab = load(Path(m['checkpoint']).parent / 'vocab_used.json')
                predictions[label] = ('dense', records(d / f'{test}_predictions.jsonl'), {v: k for k, v in vocab.items()})
            baseline = predictions['OTTC upstream-JSD'][1]
            name = 'final_test' if split == 'buckeye' else 'benchmark_libri_' + test
            for label, variant in [('CIF CE+Quantity', 'ce'), ('CIF + CTC', 'ctc_0p5'), ('CIF + OTTC', 'ottc_jsd_0p5')]:
                predictions[label] = ('cif', records(Path(english_cif[f'{dataset}_{variant}']) / f'{name}.predictions.jsonl'))
        common = set(baseline)
        for item in predictions.values():
            assert set(item[1]) == common
        candidates = []
        for key in sorted(common):
            r = baseline[key]
            duration = r.get('duration', r['reference'][-1][2])
            if 2.5 <= duration <= 4.0:
                candidates.append(key)
        assert candidates, split
        key = candidates[0]  # ID only, no model metrics or predictions used.
        base = baseline[key]
        audio_path = track(Path(base['wav']))
        audio, sr = sf.read(audio_path, dtype='int16')
        assert sr == 16000 and audio.ndim == 1
        duration = len(audio) / sr
        rid = split + '__' + key
        fresh_mfa = ROOT / 'reproduction/mfa_direct_20261017/alignments.jsonl'
        if fresh_mfa.exists():
            complete = load(fresh_mfa.with_name('RESULTS_COMPLETE.json'))
            assert complete['status'] == 'complete' and complete['utterances'] == 4
            fresh = records(fresh_mfa)
            assert fresh[key]['recording_id'] == rid
            predictions['MFA 2.2.17 (word FA)'] = ('mfa', {key: fresh[key]})
        wav = OUT / 'selected_audio' / (rid + '.wav')
        wav.parent.mkdir(exist_ok=True)
        sf.write(wav, audio, sr, subtype='PCM_16')
        check, check_sr = sf.read(wav, dtype='int16')
        assert check_sr == sr and np.array_equal(audio, check)
        manifest[rid] = dict(wav=str(wav), original_audio=str(audio_path), sample_rate=sr, samples=len(audio))
        gt = [[p, float(s), min(float(e), duration)] for p, s, e in base['reference']]
        rb = sorted({float(t) for _, s, e in gt for t in (s, e) if .02 < t < duration - .02})
        inv_zap = None
        if split == 'timit':
            zrows = records(ROOT / 'reproduction/zapformer_timit_20261013/data/test.jsonl')
            z = zrows[key]
            inv_zap = {i: p for i, (p, _, _) in zip(z['tokens'], z['references'])}
            # Include the full shared vocabulary so insertion labels are covered.
            m = load(ROOT / 'reproduction/results/runexp_ottc_fuse_phoneme_jsd/metrics.json')
            inv_zap.update({v: k for k, v in load(Path(m['checkpoint']).parent / 'vocab_used.json').items()})
        ex = dict(recording_id=rid, split=split, original_id=key, duration=duration,
                  reference=dict(phones=[x[0] for x in gt], starts=[x[1] for x in gt], ends=[x[2] for x in gt], boundaries=rb),
                  models={}, reference_kind='automatic alignment (identity unverified)' if split.startswith('libri') else 'human phone annotations')
        if split == 'timit':
            ex['auxiliary_note'] = '归档 CTM 使用词转写的元数据记录，模型/构图身份待核实；与无文本预测分开，不显示 text-free 分数。'
        if fresh_mfa.exists():
            ex['auxiliary_note'] = ex.get('auxiliary_note', '') + '本次真实 MFA：工具2.2.17，官方 english_us_arpa 声学模型/词典v3.0.0，输入原始词转写；FA与无文本分段分开。'
        for label, item in predictions.items():
            kind, data = item[:2]
            row = data[key]
            if kind == 'spans':
                nonblank = row['predicted_spans']
                spans, cursor = [], 0.
                frame_end = predictions['CIF CE+Quantity'][1][key]['frame_count'] * .02
                for phone, start, end in nonblank:
                    if start > cursor + 1e-7:
                        spans.append(['<blk>', cursor, start])
                    spans.append([phone, start, end])
                    cursor = end
                if frame_end > cursor + 1e-7:
                    spans.append(['<blk>', cursor, frame_end])
                bounds = row['predicted_boundaries']
                # Dense blank runs recovered from gaps must reproduce saved
                # scored boundaries, including the extra CTC transitions.
                recovered = sorted({s for _, s, _ in spans[1:] if .02 < s < duration - .02})
                assert len(recovered) == len(bounds) and np.allclose(recovered, bounds, atol=1e-7), (label, key)
            elif kind == 'dense':
                ids = row['text_free']['frame_token_ids']
                spans, start = [], 0
                inverse = item[2]
                for i in range(1, len(ids) + 1):
                    if i == len(ids) or ids[i] != ids[start]:
                        spans.append(['<blk>' if ids[start] == 0 else inverse[ids[start]], start * .02, i * .02])
                        start = i
                bounds = [i * .02 for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
            elif kind in ('archive', 'mfa'):
                spans = [list(v) for v in zip(row['pred_phones'], row['pred_starts'], row['pred_ends'])]
                bounds = sorted({float(t) for _, a, b in spans for t in (a, b) if .02 < t < duration - .02})
            else:
                if kind == 'cif':
                    bounds = [t * row['frame_count'] * .02 / row['duration'] for t in row['boundaries']['text_free']]
                    phones = row['hypothesis']
                else:
                    bounds = row['boundaries']['text_free']
                    phones = [inv_zap[i] for i in row['hypothesis']]
                assert len(bounds) + 1 == len(phones), (label, key)
                edges = [0.] + list(bounds) + [duration]
                spans = [[p, a, b] for p, a, b in zip(phones, edges, edges[1:])]
            spans = [[p, max(0., float(a)), min(duration, float(b))] for p, a, b in spans if b > a and a < duration]
            if spans[-1][2] < duration - 1e-7:
                spans.append(['<tail>', spans[-1][2], duration])
            scored = sorted(set(float(t) for t in bounds if .02 < t < duration - .02))
            metrics = {}
            for ms in ([] if kind == 'archive' else [10, 20, 25]):
                hits = match(rb, scored, ms / 1000)
                metrics[str(ms)] = _scorer.finalize_counts(hits, hits, len(scored), len(rb))
            model = dict(phones=[s[0] for s in spans], starts=[s[1] for s in spans], ends=[s[2] for s in spans],
                         boundaries=scored, metrics=metrics,
                         clock='native MFA 10ms timestamps, no fitted shift' if kind == 'mfa' else ('archived CTM timestamps' if kind == 'archive' else ('native duration/T (supplementary)' if kind == 'zap' else 'fixed 20ms; CIF fractional crossings retained')),
                         text_free=kind not in ('archive', 'mfa'))
            if kind == 'archive':
                model['input_protocol'] = 'Archived metadata says word_transcript; model and graph provenance unverified. Not verified MFA 3.0.'
            elif kind == 'mfa':
                model['input_protocol'] = 'Fresh MFA 2.2.17, official english_us_arpa acoustic/dictionary v3.0.0. Original word transcript; no reference phones or times.'
            ex['models'][label] = model
            directory = OUT / 'alignments' / split / label.replace(' ', '_').replace('+', '_')
            directory.mkdir(parents=True, exist_ok=True)
            converted = dict(recording_id=rid, duration=duration, effective_frame_shift_s=None if kind == 'archive' else (.01 if kind in ('zap', 'mfa') else .02),
                             phones=model['phones'], pred_starts=model['starts'], pred_ends=model['ends'], pred_boundaries=scored,
                             gt_phones=ex['reference']['phones'], gt_starts=ex['reference']['starts'], gt_ends=ex['reference']['ends'], gt_boundaries=rb,
                             metrics={f'boundary_only_{ms}ms_{k}_pct': m[k + '_pct'] for ms, m in metrics.items() for k in ['precision', 'recall', 'f1', 'r_value']})
            path = directory / 'alignments.jsonl'
            path.write_text(json.dumps(converted) + '\n')
            (directory / 'summary.json').write_text(json.dumps(dict(split=split, utterances=1, complete=True,
                scope='selected visualization example only; not full evaluation split', source_prediction_count=len(data),
                frame_shift_s=converted['effective_frame_shift_s'], clock=model['clock'], text_free=model['text_free'],
                boundary_protocol='shared reference, one-to-one, edge exclusion20ms; no fitted shifts'), indent=2) + '\n')
            if kind != 'archive':
                specs += ['--alignment', label + '=' + str(path)]

        # Standard STFT: 25ms Hann window, 5ms hop, 512-point FFT, relative dB.
        samples = audio.astype(np.float32) / 32768
        nwin, hop, nfft = 400, 80, 512
        frames = np.lib.stride_tricks.sliding_window_view(np.pad(samples, (nwin // 2, nwin // 2)), nwin)[::hop]
        power = np.abs(np.fft.rfft(frames * np.hanning(nwin), n=nfft, axis=-1)).T ** 2
        db = 10 * np.log10(np.maximum(power, 1e-12) / max(float(power.max()), 1e-12))
        figdir = OUT / 'figures'
        figdir.mkdir(exist_ok=True)
        spec_path = figdir / (rid + '_spectrogram.png')
        plt.imsave(spec_path, db, origin='lower', vmin=-80, vmax=0, cmap='magma')
        spec_extent = [-hop / (2 * sr), (len(frames) - .5) * hop / sr]
        ex['spectrogram_time_extent'] = spec_extent
        ex['spectrogram_png'] = 'data:image/png;base64,' + base64.b64encode(spec_path.read_bytes()).decode()
        ex['audio'] = 'data:audio/wav;base64,' + base64.b64encode(wav.read_bytes()).decode()
        ex['peaks'] = [float(np.max(np.abs(s))) for s in np.array_split(samples, 1000) if len(s)]
        ex['zoom'] = [0.5, min(duration, 1.8)]
        for mode, (lo, hi) in [('full', (0, duration)), ('zoom', ex['zoom'])]:
            models = list(ex['models'].items())
            fig, axes = plt.subplots(3 + len(models), 1, figsize=(20, 5 + .62 * len(models)), sharex=True,
                gridspec_kw={'height_ratios': [.6, 3, .8] + [.8] * len(models)})
            axes[0].plot(np.arange(len(samples)) / sr, samples, linewidth=.4, color='#34445a')
            axes[0].set_ylabel('Waveform', rotation=0, ha='right', fontsize=9)
            axes[0].set_yticks([])
            axes[0].set_title(f'{split} / {key} — segmentation comparison / {mode}', fontsize=14)
            axes[1].imshow(db, origin='lower', aspect='auto', extent=[*spec_extent, 0, sr / 2000], cmap='magma', vmin=-80, vmax=0)
            axes[1].set_ylabel('Frequency\n(kHz)', fontsize=10)
            tiers = [('Reference', ex['reference'], '#111827')] + [(n, m, colors[n]) for n, m in models]
            for ax, (name, tier, color) in zip(axes[2:], tiers):
                for t in rb:
                    ax.axvline(t, color='#707070', alpha=.35, linewidth=.7, linestyle='--')
                for phone, a, b in zip(tier['phones'], tier['starts'], tier['ends']):
                    if b <= lo or a >= hi:
                        continue
                    ax.axvspan(a, b, color=color, alpha=.12)
                    if (min(b, hi) - max(a, lo)) / (hi - lo) > .009:
                        ax.text((max(a, lo) + min(b, hi)) / 2, .5, phone, ha='center', va='center', fontsize=8)
                for t in tier['boundaries']:
                    ax.axvline(t, color=color, linewidth=1.1)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                label = name
                if tier.get('metrics'):
                    prefix = 'FA F1' if tier.get('text_free') is False else 'F1'
                    label += f"\n{prefix} {tier['metrics']['20']['f1_pct']:.1f} / R-value {tier['metrics']['20']['r_value_pct']:.1f}"
                elif name == 'Archived CTM (ID unverified)':
                    label += '\nword-input FA; identity unverified'
                ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
            for ax in axes:
                ax.set_xlim(lo, hi)
                ax.spines[['top', 'right']].set_visible(False)
            axes[-1].set_xlabel('Audio time (seconds); dashed gray = shared reference boundaries. Scores describe the full utterance, including in zoom views.')
            fig.subplots_adjust(left=.17, right=.99, hspace=.18, bottom=.06, top=.95)
            fig.savefig(figdir / (rid + '_' + mode + '.png'), dpi=160)
            fig.savefig(figdir / (rid + '_' + mode + '.pdf'))
            plt.close(fig)
        examples[rid] = ex
        print('Rendered', split, key, len(ex['models']), 'models.', flush=True)

    manifest_path = OUT / 'timestamp_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    # Adapt the skill exporter locally: its original template assumes IPA127
    # and interval-end boundaries. Here inventories and exact scored points
    # are explicit. Replace its HTML afterward with a spectrogram explorer.
    original_exporter = track(SKILL / 'scripts/export_segmentation_bundle.py')
    text = original_exporter.read_text().replace('IPA127', 'ARPAbet')
    text = text.replace('first["gt_ends"][:-1], xmax', 'first.get("gt_boundaries", first["gt_ends"][:-1]), xmax')
    text = text.replace('row["pred_ends"][:-1], xmax', 'row.get("pred_boundaries", row["pred_ends"][:-1]), xmax')
    local_exporter = OUT / 'export_segmentation_bundle.py'
    local_exporter.write_text(text)
    subprocess.run([sys.executable, str(local_exporter), *specs, '--manifest', str(manifest_path),
                    '--output-dir', str(OUT), '--examples-per-split', '1', '--title', 'Spectrogram and text-free model segmentation'], check=True)
    # The archival FA lane has no text-free score. Export its exact intervals
    # and boundary points without fabricating the helper's required metrics.
    spec = importlib.util.spec_from_file_location('local_segmentation_exporter', local_exporter)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    for rid, ex in examples.items():
        if ex['split'] == 'timit':
            converted = {}
            for label in ex['models']:
                p = OUT / 'alignments/timit' / label.replace(' ', '_').replace('+', '_') / 'alignments.jsonl'
                converted[label] = json.loads(p.read_text())
            exporter.write_textgrid(OUT / 'textgrids/timit' / (rid + '.TextGrid'), converted)
    template = track(Path(__file__).with_name('segmentation_explorer.html')).read_text()
    payload = json.dumps(dict(examples=examples), ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (OUT / 'interactive/index.html').write_text(template.replace('__PAYLOAD__', payload))
    summary_path = OUT / 'bundle_summary.json'
    summary = json.loads(summary_path.read_text())
    track(Path(__file__))
    summary.update(source_sha256=hashes, source_prediction_counts=source_sizes,
                   selected_by='first lexicographic common utterance ID with duration 2.5–4 seconds; no prediction quality used',
                   selected_utterances=4, source_inference_reused=True,
                   new_inference=fresh_mfa.exists(), fresh_mfa_alignment=fresh_mfa.exists(),
                   new_inference_by_plotting_script=False,
                   auxiliary_alignment='TIMIT archived CTM, word-transcript metadata, identity unverified; not MFA 3.0; no text-free score',
                   stft=dict(window_samples=400, hop_samples=80, fft_size=512, window='Hann', dB_range=[-80, 0]),
                   note='All reference/model boundaries are independent of phone-label agreement. Blank kept. CIF tail capped at waveform end for interval display only; scored points unchanged. Zapformer lane retains native clock.')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'examples.json').write_text(json.dumps(dict(examples=examples), ensure_ascii=False) + '\n')
    from reproduction.cif_comparison_20261016.segmentation_subpages import main as build_subpages
    build_subpages()
    print('COMPLETE', OUT / 'interactive/index.html', flush=True)


if __name__ == '__main__':
    main()
