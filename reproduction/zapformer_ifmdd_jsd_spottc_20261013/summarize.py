"""Create a readable reference/decode comparison from completed pilot outputs."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,required=True);args=ap.parse_args()
    out=args.output_root;pilot=json.loads((out/'fixed4/results.json').read_text())
    rows=pilot['snapshots'][-1]['utterances'];n=sum(r['reference_phones'] for r in rows)
    per=100*sum(r[k] for r in rows for k in ('insertions','deletions','substitutions'))/n
    fsa_per=100*sum(r['fsa_edits'][k] for r in rows for k in ('insertions','deletions','substitutions'))/n
    inv=json.loads((Path(__file__).parent/'inventory_mapping.json').read_text())
    labels={r['mfa_id']:r['mfa_label'] for r in inv['mfa_mapping']}
    summary=dict(objective='ifmdd_standard_jsd_spottc',steps=pilot['config']['steps'],reference_tokens=n,
                 raw_loss_first=pilot['first']['raw_loss'],raw_loss_last=pilot['last']['raw_loss'],
                 greedy_PER_percent=per,FSA_PER_percent=fsa_per,
                 scope='4 fixed training utterances; not held-out benchmark',
                 checkpoint=str((out/'fixed4'/f"step-{pilot['config']['steps']:09d}.pt").resolve()))
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# IF-MDD standard-JSD SPOTTC: decode vs MFA reference','',
           'Fixed training utterance diagnostic; not a held-out benchmark.','']
    for r in rows:
        lines.extend([f"## {r['id']}",'', 'Reference: `'+ ' '.join(r['reference_labels'])+'`','',
                      'IF-MDD greedy: `'+ ' '.join(r['hypothesis_labels'])+'`','',
                      'S3T2** FSA: `'+ ' '.join(labels[p] for p in r['fsa_hypothesis'])+'`','',
                      f"Greedy PER {r['PER_percent']:.2f}%; FSA PER {r['fsa_edits']['PER_percent']:.2f}%",''])
    (out/'DECODE_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
