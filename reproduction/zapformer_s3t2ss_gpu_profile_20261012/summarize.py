"""Aggregate real native-loop throughput without averaging rates across ranks."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--require-single',action='store_true')
    args=ap.parse_args();summaries=[]
    for output in sorted(args.root.glob('*_*')):
        if not output.is_dir():continue
        files=sorted(output.glob('profile-rank-*.json'))
        if not files:continue
        rows=[json.loads(f.read_text()) for f in files]
        successful=len(rows)==rows[0]['world_size'] and all(r['error'] is None and r['actual_updates']==r['requested_updates'] for r in rows)
        result=dict(output=str(output),world_size=rows[0]['world_size'],max_duration=rows[0]['max_duration'],success=successful)
        if successful:
            n=rows[0]['requested_updates'];warmup=rows[0]['warmup_updates']
            seconds=sum(max(r['records'][i]['seconds'] for r in rows) for i in range(warmup,n))
            audio=sum(r['audio_seconds'] for r in rows)
            result.update(audio_seconds_per_second=audio/seconds,audio_seconds_per_node_second=audio/seconds/len(rows),
                mean_update_seconds=seconds/(n-warmup),peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),padding_fraction=sum(r['padding_fraction'] for r in rows)/len(rows))
        summaries.append(result)
    if args.require_single:assert any(s['success'] and s['world_size']==1 for s in summaries),'No successful single-GPU profile'
    result=dict(profiles=summaries,ranking='audio seconds per node-second; constrained by memory and stability')
    (args.root/'profile_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
