"""Export benchmark bars and identical-utterance segment comparisons."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reproduction/vendor'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = [('CTC', 'runexp_ctc_fuse_phoneme'), ('OTTC CE', 'runexp_ottc_fuse_phoneme'),
            ('OTTC upstream jsd', 'runexp_ottc_fuse_phoneme_jsd'),
            ('OTTC standard JS', 'runexp_ottc_fuse_phoneme_jsd_standard')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-softdtw', action='store_true')
    args = parser.parse_args()
    if args.include_softdtw:
        VARIANTS.append(('Soft-DTW CE (gamma=0.1)', 'runexp_softdtw_fuse_phoneme'))
    target = ROOT/'reproduction/results'/('softdtw_comparison' if args.include_softdtw else 'js_comparison')
    target.mkdir(parents=True, exist_ok=True)
    metrics = [json.loads((ROOT/'reproduction/results'/s/'metrics.json').read_text()) for _, s in VARIANTS]
    splits = list(metrics[0]['splits'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [('normalized_nosil_PER', 'No-sil phone error rate (lower is better)'),
             ('f1', 'Boundary F1 at 20 ms'), ('matched_interval_overlap_ratio', 'Matched interval overlap')]
    x = np.arange(len(splits))
    for ax, (metric, title) in zip(axes, specs):
        for i, ((label, _), results) in enumerate(zip(VARIANTS, metrics)):
            values = [100*(results['splits'][s]['boundaries']['20'][metric] if metric == 'f1'
                           else results['splits'][s][metric]) for s in splits]
            width = .8/len(VARIANTS)
            ax.bar(x+(i-(len(VARIANTS)-1)/2)*width, values, width=width*.95, label=label)
        ax.set_xticks(x, ['TIMIT README 240', 'TIMIT core 192', 'Buckeye 1840', 'L2-ARCTIC 900'], rotation=25, ha='right')
        ax.set_title(title)
        ax.set_ylabel('Percent')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(target/f'benchmark_comparison.{ext}', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, suffix in VARIANTS:
        history = json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/
                              suffix/'trainer_state.json').read_text())['log_history']
        epochs = [h for h in history if 'eval_wer' in h]
        ax.plot([h['epoch'] for h in epochs], [100*h['eval_wer'] for h in epochs], label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Trainer phoneme edit rate (%)')
    ax.set_title('Training convergence — identical final-epoch selection')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(target/f'convergence.{ext}', dpi=160)
    plt.close(fig)
    records = [[json.loads(line) for line in (ROOT/'reproduction/results'/s/'readme_kaldi_test_240.jsonl').read_text().splitlines()[:3]]
               for _, s in VARIANTS]
    vocabulary = json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/
                             VARIANTS[0][1]/'vocab_used.json').read_text())
    for n in range(3):
        assert len({r[n]['utt'] for r in records}) == 1
        assert all(r[n]['reference'] == records[0][n]['reference'] for r in records)
        fig, axes = plt.subplots(len(VARIANTS)+1, 1, figsize=(16, len(VARIANTS)+3), sharex=True)
        rows = [('Reference', records[0][n]['reference'])]+[(label, r[n]['predicted_spans']) for (label, _), r in zip(VARIANTS, records)]
        for ax, (label, spans) in zip(axes, rows):
            for phone, start, end in spans:
                ax.axvspan(start, end, color=plt.get_cmap('tab20')(vocabulary.get(phone, 0)%20), alpha=.75)
                ax.text((start+end)/2, .5, phone, rotation=90, fontsize=7, ha='center', va='center')
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9)
        axes[0].set_title(records[0][n]['utt']+' — final epoch 40, audio-only segmentation')
        axes[-1].set_xlabel('Original waveform time (seconds)')
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(target/f'segmentation_{n+1}.{ext}', dpi=160)
        plt.close(fig)
    print('Saved benchmark and 3 shared-utterance comparisons:', target)


if __name__ == '__main__':
    main()
