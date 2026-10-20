"""Standalone scientific figure: full test-set F1 versus tolerance."""
import json
import os
from reproduction.ottc_alpha_full_20261017.prepare import HERE


def main():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ottc_alpha_full_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads((HERE / 'RESULTS.json').read_text())
    assert data['status'] == 'complete'
    names = {'timit_core_test_192': 'TIMIT core test (192)', 'buckeye_test': 'Buckeye test (1,840)',
             'libri_test_clean': 'Libri test-clean (2,620)', 'libri_test_other': 'Libri test-other (2,938)'}
    styles = [('argmax', 'Argmax', '#52677f', 's'), ('alpha_grid', 'Alpha rounded', '#dc8b22', '^'),
              ('alpha_fractional', 'Alpha fractional', '#007c83', 'o')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (split, result) in zip(axes, data['splits'].items()):
        for method, label, color, marker in styles:
            ax.plot([10, 20, 25, 30], [result['methods'][method][str(ms)]['f1_pct'] for ms in [10, 20, 25, 30]],
                    label=label, color=color, marker=marker, linewidth=1.8, markersize=4)
        ax.set_title(names[split], fontsize=10)
        ax.set_xticks([10, 20, 25, 30])
        ax.set_xlabel('Boundary tolerance (ms)')
        ax.grid(alpha=.2)
        ax.axvline(20, linewidth=.8, color='black', alpha=.2)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('Boundary F1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Upstream-JSD: full test sets, identical audio-only recognition; fixed 20 ms frame cells', fontsize=11)
    fig.tight_layout(rect=(0, .12, 1, .93))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / ('full_test_boundary_f1.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
