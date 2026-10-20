"""Fixed utterance plots of references and complete topology occurrences."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'reproduction/ipa_arpabet_comparison_20261012'))
from rescore import TOKENS, mapped


def label(phone):
    labels = mapped([phone], 'english_collapsed')
    return labels[0].replace('UNMAPPED_IPA:', '') if labels else TOKENS[phone]


def segments(path):
    result = []
    active = None
    for i, state_id in enumerate(path):
        phone = 0 if state_id == 0 else (state_id - 1) // 3 + 1
        state = -1 if state_id == 0 else (state_id - 1) % 3
        if active is None or phone != active[0] or (state == 0 and active[3] == 2):
            if active is not None:
                result.append(active[:3])
            active = [phone, i * .01, (i + 1) * .01, state]
        else:
            active[2] = (i + 1) * .01
            active[3] = state
    if active is not None:
        result.append(active[:3])
    return result


def main():
    refs = {r['id']: r for r in map(json.loads, (ROOT / 'reproduction/spottc_jsd_s3t2_framece_libri100_20261012/eval_data/development.jsonl').open())}
    fixed_ids = list(refs)[:2]
    colors = plt.get_cmap('tab20')
    for key in fixed_ids:
        ref = refs[key]
        rows = [('Reference', [(p['mfa_id'], p['start'], p['end']) for p in ref['phone_intervals']])]
        for arm in ('jsd_fs', 'jsd_only', 'ce_fs', 'ce_only'):
            file = HERE / arm / 'final_development_predictions.jsonl'
            if not file.exists():
                continue
            prediction = None
            for line in file.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r['id'] == key:
                    prediction = r
                    break
            if prediction is not None:
                rows.append((arm + ' legal path', segments(prediction['legal_state_path'])))
        figure, axes = plt.subplots(len(rows), 1, figsize=(20, 1.25 * len(rows) + 1.3), sharex=True)
        for ax, (name, spans) in zip(axes, rows):
            for phone, start, end in spans:
                ax.add_patch(Rectangle((start, 0), end - start, 1,
                             facecolor='#eeeeee' if phone == 0 else colors(phone % 20),
                             edgecolor='white', linewidth=.6))
                if phone != 0 and end - start >= .018:
                    ax.text((start + end) / 2, .5, label(phone), ha='center', va='center',
                            fontsize=8, rotation=90 if end - start < .055 else 0)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10)
            ax.set_xlim(0, ref['duration'])
        transcript_file = Path(ref['wav']).with_suffix('.txt')
        transcript = ' '.join(transcript_file.read_text().split()[2:])
        figure.suptitle(key + ' — ' + transcript + '\nEach decoded box is one complete S0→[S1]→S2 occurrence; gray = blank.', fontsize=13)
        axes[-1].set_xlabel('Time (seconds)')
        figure.tight_layout(rect=(.02, 0, 1, .90))
        figure.savefig(HERE / (key.split('_')[-1] + '_reference_decode.png'), dpi=170)
        figure.savefig(HERE / (key.split('_')[-1] + '_reference_decode.pdf'))
        plt.close(figure)


if __name__ == '__main__':
    main()
