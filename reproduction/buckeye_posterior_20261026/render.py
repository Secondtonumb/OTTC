"""Matched-scale scientific figures and a local Posterior Lab compatible viewer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from reproduction.buckeye_posterior_20261026.export import METHODS, sha, write


def draw(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    reference = json.loads((output / 'reference.json').read_text())
    data = {cost: np.load(output / f'{cost}.npz') for cost, _ in METHODS}
    labels = manifest['labels']
    # Fixed gold-defined first-word window, independent of posterior differences.
    window = (0., reference['words'][0]['end'])
    phones = list(dict.fromkeys(p for p, s, e in reference['references']['ottc_arpabet'] if s < window[1])) + ['sil']
    colors = plt.get_cmap('tab10')(np.arange(len(phones)))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    for j, (cost, title) in enumerate(METHODS):
        ax = fig.add_subplot(2, 2, j + 1, projection='3d', computed_zorder=False)
        t = data[cost]['frame_time_seconds']
        keep = (t >= window[0]) & (t <= window[1])
        x = t[keep]
        for i, phone in enumerate(phones):
            z = data[cost]['posterior'][keep, labels.index(phone)]
            collection = PolyCollection([[(x[0], 0), *zip(x, z), (x[-1], 0)]],
                                        facecolors=[colors[i]], edgecolors=[colors[i]], alpha=.55)
            ax.add_collection3d(collection, zs=i, zdir='y')
            ax.plot(x, np.full(len(x), i), z, color=colors[i], linewidth=1.8)
        meta = manifest['methods'][j]
        ax.set_title(f'{title} | current best-dev epoch {meta["epoch"]}', loc='left', fontsize=13)
        ax.set(xlim=window, ylim=(-.3, len(phones)-.7), zlim=(0, 1), xlabel='Time (s)', zlabel='Posterior')
        ax.set_yticks(range(len(phones)), phones)
        ax.set_zticks([0, .5, 1])
        ax.view_init(elev=27, azim=-66)
        ax.set_box_aspect((2.3, 1.25, 1))
    fig.suptitle(f'Real Buckeye phone posteriors: "{reference["words"][0]["word"]}"', x=.06, ha='left', fontsize=21)
    fig.text(.06, .94, f'{manifest["utterance"]} | "{manifest["text"]}" | seed {manifest["seed"]} | FP32, audio-only, native 10 ms', fontsize=11)
    fig.text(.06, .035, 'Interim best-dev snapshots; training is unfinished. Same phone colors, camera and 0-1 probability scale.\n'
             'First dev utterance, first word. Displayed classes are not renormalized. One example does not rank objectives.', fontsize=10)
    fig.subplots_adjust(left=.025, right=.98, top=.89, bottom=.11, hspace=.13, wspace=.04)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'posterior_3d_comparison.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, layout='constrained')
    for ax, (cost, title) in zip(axes, METHODS):
        t = data[cost]['frame_time_seconds']
        image = ax.imshow(data[cost]['posterior'].T, aspect='auto', origin='lower', vmin=0, vmax=1,
                          extent=[-.005, float(t[-1])+.005, -.5, 41.5], cmap='magma', interpolation='nearest')
        ax.set_yticks(range(42), labels, fontsize=6)
        ax.set_ylabel(title)
        for phone, start, end in reference['references']['ottc_arpabet']:
            ax.axvline(start, color='cyan', alpha=.25, linewidth=.55)
            ax.text((start+end)/2, 42, phone, fontsize=7, ha='center', clip_on=False)
    axes[-1].set_xlabel('Time (s); exact mapped gold intervals above each panel; annotation gaps retained')
    fig.suptitle(f'Buckeye full40 | {manifest["utterance"]} | interim best-dev checkpoints')
    fig.colorbar(image, ax=axes, label='Posterior', shrink=.6)
    fig.savefig(output / 'posterior_full42_heatmap.png', dpi=180)
    plt.close(fig)
    site_json(output, manifest, reference)


def site_json(output, manifest, reference):
    from reproduction.timit_posterior_20261026 import site_adapter as adapter
    filename = f'buckeye_{manifest["utterance"]}_seed{manifest["seed"]}.posterior-v1.json'
    adapter.FILENAME = filename
    n = manifest['methods'][0]['shape'][0]
    offset = manifest['inference']['receptive_center_offset_seconds'] * 1000
    times = [offset + i * 10 for i in range(n)]
    methods = []
    for cost, key, label in adapter.METHODS:
        raw = json.loads((output / f'{cost}.json').read_text())
        assert len(raw['posterior']) == n and raw['labels'] == manifest['labels']
        assert all(abs(t - c * 1000) < .001 for t, c in zip(times, raw['receptive_center_seconds']))
        methods.append(dict(id=key, label=label, posterior=raw['posterior']))
    payload = dict(schema_version=1, source_type='model', utterance_id=manifest['utterance'],
                   text=manifest['text'], phones=manifest['labels'], time_ms=times, methods=methods,
                   reference_segments=[dict(start_ms=s*1000, end_ms=e*1000, phone=p)
                                       for p, s, e in reference['references']['ottc_arpabet']],
                   metadata=dict(dataset='Buckeye full40 20261022', split='dev', seed=manifest['seed'],
                                 experiment_family='buckeye_full40_seed_sweep_20261025',
                                 snapshot_utc=manifest['snapshot_utc'], training_complete=False,
                                 checkpoint_selection='interim single best dev PER per model',
                                 frame_hop_ms=10, time_convention='CNN receptive field centers', time_offset_ms=offset,
                                 sample_rate_hz=16000, receptive_field_samples=manifest['inference']['receptive_field_samples'],
                                 inference_dtype='float32', reference_conditioned_inference=False,
                                 reference_times_quantized=False, complete_vocabulary=True, subset_renormalized=False,
                                 reference_tier='full40 exact mapped human intervals; real gaps retained',
                                 methods=[{k: m[k] for k in ('cost', 'title', 'seed', 'epoch', 'step', 'checkpoint_sha256',
                                                                          'config_sha256', 'selection', 'training_complete')}
                                          for m in manifest['methods']], audio_sha256=manifest['audio_sha256']))
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    assert '/work/' not in text and '/home/' not in text
    (output / filename).write_text(text)
    for name in ('posterior-format.md', 'posterior.schema.json', 'convert_posterior.py'):
        (output / name).write_bytes((adapter.CONTRACT / name).read_bytes())
    viewer = adapter.standalone_viewer(output, payload)
    html_path = output / 'posterior_3d_interactive.html'
    html = html_path.read_text().replace('TIMIT', 'Buckeye')
    html = html.replace('<audio controls', '<p style="color:#934b10">训练中 best-dev 快照；单句实例，不代表全数据集统计或最终模型排名。</p><audio controls')
    html_path.write_text(html)
    recovered = json.loads((output / filename).read_text())
    for (cost, _), method in zip(METHODS, recovered['methods']):
        assert method['posterior'] == json.loads((output / f'{cost}.json').read_text())['posterior']
    write(output / 'SITE_INTEGRATION.json', dict(file=filename, shape=[4, n, 42],
          posterior_values_unchanged=True, offline=True, time_offset_ms=offset, viewer=viewer,
          json_sha256=sha(output / filename), adapter_sha256=sha(__file__)))
