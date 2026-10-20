#!/bin/bash
# Run inside an eight-node Miyabi interactive allocation.
# Continue the existing JSD-only experiment with its saved checkpoint/W&B ID.
set -euo pipefail
task_stage=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo=$(cd -- "$task_stage/../.." && pwd)
cd "$task_repo"
source "$task_repo/reproduction/zapformer_s3t2ss_native_20261012/environment.sh"
: "${PBS_NODEFILE:?First request an eight-node interactive allocation; see INTERACTIVE_8GPU.md}"
: "${PBS_JOBID:?Run inside a PBS allocation}"
task_node_count=$(awk '!seen[$0]++ {n++} END {print n+0}' "$PBS_NODEFILE")
if [[ "$task_node_count" != 8 ]]; then
  printf 'Expected eight distinct allocated nodes; got %s\n' "$task_node_count" >&2
  exit 2
fi
export SPOTTC_ARM=jsd_only
export WANDB_MODE="${WANDB_MODE:-online}" PYTHONUNBUFFERED=1
# Two completed epochs per session; deadline leaves90seconds for final save/exit.
export KEEP_BATCH_HELD="${KEEP_BATCH_HELD:-1}"
[[ "$KEEP_BATCH_HELD" == 0 || "$KEEP_BATCH_HELD" == 1 ]] || { echo 'KEEP_BATCH_HELD must be0or1'; exit 2; }
if [[ -n "${EPOCH_TARGET:-}" ]]; then
 [[ "$EPOCH_TARGET" =~ ^[1-9][0-9]*$ ]] && (( EPOCH_TARGET <= 50 )) || { echo 'EPOCH_TARGET must be1..50'; exit 2; }
 export EPOCH_TARGET
fi
export EPOCHS_PER_SESSION="${EPOCHS_PER_SESSION:-2}"
export INTERACTIVE_MAX_SECONDS="${INTERACTIVE_MAX_SECONDS:-600}"
export INTERACTIVE_SAVE_RESERVE="${INTERACTIVE_SAVE_RESERVE:-90}"
for task_value in "$EPOCHS_PER_SESSION" "$INTERACTIVE_MAX_SECONDS" "$INTERACTIVE_SAVE_RESERVE"; do
  [[ "$task_value" =~ ^[1-9][0-9]*$ ]] || { echo 'Session settings must be positive integers'; exit 2; }
done
if [[ -n "${SP4_STOP_AFTER:-}" ]]; then
  echo 'Unset SP4_STOP_AFTER; the two-epoch target is managed automatically.' >&2; exit 2
fi
SP4_DEADLINE_UNIX=$("$MFA_PYTHON" - "$PBS_JOBID" "$INTERACTIVE_MAX_SECONDS" "$INTERACTIVE_SAVE_RESERVE" <<'PYDEADLINE'
import re,subprocess,sys,time
job,cap,reserve=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
status=subprocess.run(['qstat','-f',job],capture_output=True,text=True,check=True).stdout
start=re.search(r'^\s*stime = (.+)$',status,re.M)
wall=re.search(r'^\s*Resource_List.walltime = (\d+):(\d+):(\d+)',status,re.M)
if not start or not wall:raise SystemExit('Cannot determine allocation start/walltime from qstat')
started=time.mktime(time.strptime(start.group(1).strip(),'%a %b %d %H:%M:%S %Y'))
allowed=int(wall.group(1))*3600+int(wall.group(2))*60+int(wall.group(3))
deadline=min(started+allowed,time.time()+cap)-reserve
if deadline<=time.time()+30:raise SystemExit('Allocation has too little time remaining; request a fresh allocation')
print(int(deadline))
PYDEADLINE
)
export SP4_DEADLINE_UNIX
"$MFA_PYTHON" "$task_stage/verify.py" --arm jsd_only
"$MFA_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():raise SystemExit('CUDA unavailable on this interactive node')
print('GPU:',torch.cuda.get_device_name(0),'; visible devices:',torch.cuda.device_count(),flush=True)
PY
task_output="$task_stage/models/jsd_only"
mkdir -p "$task_output"
task_log="$task_output/interactive_${PBS_JOBID%%.*}.log"
exec > >(tee -a "$task_log") 2>&1
# Hold only the matching queued batch job for the duration of this interactive run.
# Existing run_ddp.sh also acquires the shared training lock before touching state.
task_batch_job=$("$MFA_PYTHON" - "$task_stage/jobs.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['training']['jsd_only']['job_id'])
PY
)
[[ "$task_batch_job" =~ ^[0-9]+\.[A-Za-z0-9_-]+$ ]] || { echo 'Invalid saved batch job ID'; exit 2; }
task_status_file=$(mktemp /tmp/ifmdd_jsdonly_status.XXXXXX)
task_held_job=''
cleanup() {
  local task_rc=$?
  trap - EXIT
  rm -f "$task_status_file"
  if [[ -n "$task_held_job" && "$KEEP_BATCH_HELD" == 1 ]]; then
    printf 'Batch job %s remains held for repeated interactive sessions; release: /opt/pbs/bin/qrls -h u %s\n' "$task_held_job" "$task_held_job"
  elif [[ -n "$task_held_job" ]]; then
    if /opt/pbs/bin/qrls -h u "$task_held_job"; then
      printf 'Released queued batch job %s; it will resume the saved checkpoint.\n' "$task_held_job"
    else
      printf 'Could not release %s; release it with: /opt/pbs/bin/qrls -h u %s\n' "$task_held_job" "$task_held_job" >&2
    fi
  fi
  exit "$task_rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if qstat -f "$task_batch_job" > "$task_status_file" 2>/dev/null; then
  task_state=$(awk '$1=="job_state" && $2=="=" {print $3;exit}' "$task_status_file")
  task_holds=$(awk '$1=="Hold_Types" && $2=="=" {print $3;exit}' "$task_status_file")
  case "$task_state" in
    R|E|S|T|B) echo "Batch job $task_batch_job is already running; use its logs for this experiment." >&2; exit 1 ;;
    Q)
      /opt/pbs/bin/qhold -h u "$task_batch_job"
      task_held_job="$task_batch_job"
      qstat -f "$task_batch_job" > "$task_status_file"
      task_state=$(awk '$1=="job_state" && $2=="=" {print $3;exit}' "$task_status_file")
      task_holds=$(awk '$1=="Hold_Types" && $2=="=" {print $3;exit}' "$task_status_file")
      [[ "$task_state" == H && "$task_holds" == *u* ]] || { echo 'Could not hold queued batch job before launch'; exit 1; }
      printf 'Temporarily held matching queued batch job %s.\n' "$task_batch_job"
      ;;
    H)
      echo "Batch job $task_batch_job already has hold=$task_holds; leaving that hold in place."
      ;;
  esac
fi
printf 'JSD-only: IF-MDD standard JSD, S3T2**, FS=0; eightGPU DDP; batch640s/rank;50epoch recipe.\n'
printf 'Session default: +%s completed epochs; max segment %ss; save reserve %ss; output: %s\n' "$EPOCHS_PER_SESSION" "$INTERACTIVE_MAX_SECONDS" "$INTERACTIVE_SAVE_RESERVE" "$task_output"
if [[ -n "${EPOCH_TARGET:-}" ]]; then printf 'Absolute target: epoch%s\n' "$EPOCH_TARGET"; fi
printf 'Local log: %s\n' "$task_log"
bash "$task_stage/interactive_epochs/dispatch.sh"
printf 'Interactive segment finished; latest progress and checkpoint are in %s\n' "$task_output"
