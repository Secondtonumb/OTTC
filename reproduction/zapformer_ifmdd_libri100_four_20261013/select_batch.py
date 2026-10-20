"""Require all four real-GPU probes and choose one shared duration with headroom."""
import hashlib,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    candidates={}
    for duration in (320,640):
        reports=[json.loads((HERE/'throughput'/arm/str(duration)/'SUMMARY.json').read_text()) for arm in ('jsd_fs','jsd_only','ce_fs','ce_only')]
        for r in reports:assert r['source_sha256']==identity
        if any(r['status']!='passed' for r in reports):
            candidates[duration]=dict(safe=False,reason='A GPU candidate failed',arms=reports);continue
        peak=max(v['peak_reserved_GiB']/r['total_memory_GiB'] for r in reports for v in r['results'])
        worst=min(statistics.median(v['audio_seconds_per_second'] for v in r['results']) for r in reports)
        candidates[duration]=dict(peak_reserved_fraction=peak,worst_arm_median_audio_seconds_per_second=worst,arms=reports,safe=peak<.85)
    safe=[d for d,v in candidates.items() if v['safe']]
    assert safe,'No candidate has 15% GPU memory headroom; formal dispatch blocked'
    # Retain the morning duration if safe and within 10% of best throughput.
    best=max(safe,key=lambda d:candidates[d]['worst_arm_median_audio_seconds_per_second'])
    chosen=640 if 640 in safe and candidates[640]['worst_arm_median_audio_seconds_per_second']>=.9*candidates[best]['worst_arm_median_audio_seconds_per_second'] else best
    result=dict(status='passed',max_duration=chosen,policy='same duration all four; require <85% reserved GPU memory on short/median/longest real batches; retain morning640 when within10% of best worst-arm throughput',candidates=candidates,source_sha256=identity)
    (HERE/'BATCH_SELECTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(max_duration=chosen,candidates={d:{k:v for k,v in r.items() if k!='arms'} for d,r in candidates.items()})))

if __name__=='__main__':main()
