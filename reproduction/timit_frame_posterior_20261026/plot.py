"""Scientific figures and a standalone offline Plotly viewer for raw posteriors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

METHODS = [('ce', 'Frame-wise CE'), ('jsd_standard', 'Frame-wise JSD'), ('ag', 'Frame-wise AG'), ('jdiv', 'Frame-wise J-DIV')]
PHONES = ['b', 'r', 'ih', 'k', 's', 'sil']
COLORS = ['#2563eb', '#e65d28', '#10a681', '#a64ac9', '#d6a111', '#6b778b']
WINDOW = (.10, .47)


def draw(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    reference = json.loads((output / 'reference.json').read_text())
    data = {cost: np.load(output / f'{cost}.npz') for cost, _ in METHODS}
    vocab = {label: i for i, label in enumerate(manifest['labels'])}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    for j, (cost, title) in enumerate(METHODS):
        ax = fig.add_subplot(2, 2, j + 1, projection='3d', computed_zorder=False)
        t = data[cost]['receptive_center_seconds']
        keep = (t >= WINDOW[0]) & (t <= WINDOW[1])
        x = t[keep]
        for p, phone in enumerate(PHONES):
            z = data[cost]['posterior'][keep, vocab[phone]]
            verts = [(x[0], 0), *zip(x, z), (x[-1], 0)]
            collection = PolyCollection([verts], facecolors=COLORS[p], edgecolors=COLORS[p], alpha=.55)
            ax.add_collection3d(collection, zs=p, zdir='y')
            ax.plot(x, np.full(len(x), p), z, color=COLORS[p], linewidth=1.8)
        meta = manifest['methods'][j]
        ax.set_title(f'{title}  |  best-dev epoch {meta["epoch"]:g}', loc='left', fontsize=13, pad=4)
        ax.set(xlim=WINDOW, ylim=(-.3, 5.3), zlim=(0, 1), xlabel='Time (s)', zlabel='Posterior')
        ax.set_yticks(range(len(PHONES)), PHONES)
        ax.set_xticks([.1, .2, .3, .4])
        ax.set_zticks([0, .5, 1])
        ax.view_init(elev=27, azim=-66)
        ax.set_box_aspect((2.3, 1.25, 1))
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(alpha=.2)
    fig.suptitle('Frame-wise TIMIT phone posteriors: "bricks"', x=.06, ha='left', fontsize=21, fontweight='bold')
    fig.text(.06, .94, f'{manifest["utterance"]}  |  "{manifest["text"]}"  |  seed 43  |  FP32, audio-only, native 10 ms', fontsize=11, color='#465367')
    fig.text(.06, .035, 'Same phone colors, camera and probability scale. Raw softmax over all 42 IDs; six classes displayed without renormalization.\n'
             'Fixed first dev utterance. CNN frame-center times; no smoothing or subset renormalization. One example only.', fontsize=10, color='#465367')
    fig.subplots_adjust(left=.025, right=.98, top=.89, bottom=.11, hspace=.13, wspace=.04)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'posterior_3d_comparison.{ext}', dpi=210, facecolor='white')
    plt.close(fig)

    # Full-vocabulary heatmaps provide a compact check for omitted competing classes.
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, layout='constrained')
    for ax, (cost, title) in zip(axes, METHODS):
        array = data[cost]['posterior']
        t = data[cost]['receptive_center_seconds']
        im = ax.imshow(array.T, aspect='auto', origin='lower', vmin=0, vmax=1,
                       extent=[float(t[0])-.005, float(t[-1])+.005, -.5, 41.5], cmap='magma', interpolation='nearest')
        ax.set_yticks(range(42), manifest['labels'], fontsize=6)
        ax.set_ylabel(title, fontsize=12)
        # Gold times remain exact; they never enter model inference.
        for phone, start, end in reference['references']['ottc_arpabet']:
            ax.axvline(start, color='cyan', alpha=.22, linewidth=.55)
            ax.text((start+end)/2, 42, phone, fontsize=6, ha='center', clip_on=False)
    axes[-1].set_xlabel('Time (s); exact mapped PHN intervals labeled above each panel')
    fig.suptitle(f'{manifest["utterance"]}: full 42-ID posterior, {manifest["text"]}', fontsize=15)
    fig.colorbar(im, ax=axes, label='Posterior', shrink=.6)
    fig.savefig(output / 'posterior_full42_heatmap.png', dpi=180)
    plt.close(fig)

    from reproduction.timit_frame_posterior_20261026.site_adapter import build
    build(output)


if __name__ == '__main__':
    import sys
    draw(Path(sys.argv[1]))
