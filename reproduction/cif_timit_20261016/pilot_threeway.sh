#!/usr/bin/env bash
# Five-epoch real-data trends, with the full 40-epoch schedule left resumable.
set -euo pipefail
cd /work/gm64/m64000/OTTC
export PYTHONPATH="$PWD/reproduction/vendor:$PWD"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
CIF_PY=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
CIF_PROBE="${CIF_PROBE_ROOT:-$PWD/reproduction/cif_timit_20261016/probes/${PBS_JOBID}}"
"$CIF_PY" - "$CIF_PROBE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('no_ctc','ctc_0p5','ottc_jsd_0p5'):
    marker=json.loads((root/name/'PROBE_COMPLETE.json').read_text())
    assert marker['progress']['step']==4, name
    config=json.loads((root/name/'run_config.json').read_text())
    assert config['batch_size']==64 and config['split_counts']['train']==4620
    for path,digest in config['sources_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
print('All three longest-batch full-encoder profile gates passed',flush=True)
PY
CIF_BASE="${CIF_PILOT_ROOT:-$PWD/reproduction/cif_timit_20261016/pilots/${PBS_JOBID}}"
mkdir -p "$CIF_BASE"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > "$CIF_BASE/telemetry.csv" &
CIF_TELEMETRY=$!
trap 'kill "$CIF_TELEMETRY" 2>/dev/null || true' EXIT
CIF_FAILURES=0
# Prioritize upstream-JSD OTTC; keep every other authorized experiment.
for CIF_VARIANT in ottc_jsd_0p5 no_ctc ctc_0p5; do
  CIF_CTC=0
  CIF_OTTC=0
  if [[ "$CIF_VARIANT" == ctc_0p5 ]]; then CIF_CTC=0.5; fi
  if [[ "$CIF_VARIANT" == ottc_jsd_0p5 ]]; then CIF_OTTC=0.5; fi
  if CIF_OUTPUT="$CIF_BASE/$CIF_VARIANT" CIF_CTC_WEIGHT="$CIF_CTC" CIF_OTTC_WEIGHT="$CIF_OTTC" \
      CIF_MAX_WALL_SECONDS=900 bash reproduction/cif_timit_20261016/interactive.sh \
      --max-steps 365 --evaluate-on-stop; then
    printf 'PILOT_INVOCATION_OK %s\n' "$CIF_VARIANT"
  else
    CIF_FAILURES=$((CIF_FAILURES+1))
    printf 'PILOT_INVOCATION_FAILED %s\n' "$CIF_VARIANT"
  fi
done
exit "$CIF_FAILURES"
