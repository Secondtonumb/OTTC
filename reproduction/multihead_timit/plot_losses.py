"""Render only recorded Trainer loss points for the local Markdown report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    counts = {}
    for axis, objective, title in zip(axes, ['ctc', 'ce', 'jsd_upstream'],
                                      ['CTC', 'OTTC + CE', 'OTTC + upstream jsd']):
        history = json.loads((ROOT/'models'/objective/'trainer_state.json').read_text())['log_history']
        points = [e for e in history if 'loss' in e]
        axis.plot([e['step'] for e in points], [e['loss'] for e in points], marker='.', markersize=4)
        axis.set(title=title, xlabel='Optimizer step', ylabel='Recorded train loss')
        axis.grid(alpha=.25)
        counts[objective] = len(points)
    output = ROOT/'notion_update'/'figures'
    output.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        figure.savefig(output/f'train_losses.{extension}', dpi=160)
    plt.close(figure)
    print(json.dumps({'recorded_points': counts, 'output': str(output)}))


if __name__ == '__main__':
    main()
