"""Plot raw frame posteriors and predicted/reference phone intervals."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffixes', nargs='+', default=['runexp_ottc_fuse_phoneme', 'runexp_ctc_fuse_phoneme'])
    args = parser.parse_args()
    for suffix in args.suffixes:
        title = suffix.removeprefix('runexp_').replace('_fuse_phoneme', '').upper()
        target = ROOT/'reproduction/results'/suffix
        model_dir = ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/suffix
        vocabulary = json.loads((model_dir/'vocab_used.json').read_text())
        labels = [None]*len(vocabulary)
        for label, i in vocabulary.items():
            labels[i] = label
        examples = np.load(target/'examples.npz')
        records = [json.loads(line) for line in (target/'readme_kaldi_test_240.jsonl').read_text().splitlines()]
        for i in range(3):
            record = records[i]
            logits = examples[f'{i}_logits']
            exp = np.exp(logits-logits.max(axis=1, keepdims=True))
            probabilities = exp/exp.sum(axis=1, keepdims=True)
            figure, axes = plt.subplots(3, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [5, 1, 1]}, sharex=True)
            axes[0].imshow(probabilities.T, origin='lower', aspect='auto', extent=[0, len(logits)*.02, -.5, len(labels)-.5], vmin=0, vmax=1, cmap='magma')
            axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
            axes[0].set_title(f'{title} epoch 40 — {record["utt"]}')
            colors = plt.get_cmap('tab20')
            for ax, spans, title in zip(axes[1:], [record['reference'], record['predicted_spans']], ['Reference (official fused phones)', 'Audio-only argmax phone intervals']):
                for label, start, end in spans:
                    index = vocabulary.get(label, 0)
                    ax.axvspan(start, end, color=colors(index%20), alpha=.75)
                    ax.text((start+end)/2, .5, label, ha='center', va='center', fontsize=7, rotation=90)
                ax.set_yticks([])
                ax.set_ylim(0, 1)
                ax.set_ylabel(title, fontsize=8)
            axes[-1].set_xlabel('Original waveform time (seconds)')
            figure.tight_layout()
            figure.savefig(target/f'example_{i+1}.png', dpi=160)
            figure.savefig(target/f'example_{i+1}.pdf')
            plt.close(figure)


if __name__ == '__main__':
    main()
