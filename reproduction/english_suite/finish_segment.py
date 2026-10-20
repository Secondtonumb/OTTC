"""Record allocation outcomes and continue the authorized long training when needed."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT/'reproduction/english_suite'


def main():
    runs = {}
    for dataset in ['libri100', 'libri960']:
        for objective in ['ctc', 'ce', 'jsd_upstream']:
            directory = ROOT/'reproduction'/f'multihead_{dataset}'/'models'/objective
            marker = directory/'TRAIN_COMPLETE.json'
            complete = json.loads(marker.read_text()) if marker.exists() else None
            runs[f'{dataset}/{objective}'] = {'complete': bool(complete and complete['epochs'] >= 40 and not complete['smoke']),
                                             'completion': complete}
    done = all(r['complete'] for r in runs.values())
    status = {'updated_at': datetime.now().astimezone().isoformat(), 'all_complete': done, 'runs': runs}
    if not done:
        command = ['qsub', '-v', 'CHAIN_CONTINUE=1', '-o', str(SUITE/'train_continuation.log'), str(SUITE/'train.pbs')]
        status['continuation_job'] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    status['evaluation_job'] = subprocess.check_output(['qsub', '-o', str(SUITE/'evaluation_continuation.log'),
                                                       str(SUITE/'evaluate.pbs')], cwd=ROOT, text=True).strip()
    (SUITE/'continuation_status.json').write_text(json.dumps(status, indent=2)+'\n')
    from summarize_results import main as summarize
    summarize()
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
