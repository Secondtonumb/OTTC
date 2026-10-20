#!/bin/bash
set -euo pipefail
usage() {
    printf 'Usage: bash %s {jsd|ce|both} [total_seconds=2400]\n' "$0"
    printf 'Run inside a PBS GPU allocation. Existing queued/running matching jobs must finish or be cancelled first.\n'
}
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then usage; exit 0; fi
selection=${1:-both}
requested_seconds=${2:-2400}
if (( $# > 2 )); then usage >&2; exit 2; fi
case "$selection" in jsd|ce) costs=("$selection") ;; both) costs=(jsd ce) ;; *) usage >&2; exit 2 ;; esac
if [[ ! "$requested_seconds" =~ ^[1-9][0-9]{0,4}$ ]] || (( requested_seconds < 180 )); then
    printf 'Budget must be an integer between 180 and 99999 seconds\n' >&2
    exit 2
fi
: "${PBS_JOBID:?Run inside an allocated PBS GPU interactive session}"
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OTTC_EVAL_UPLOAD=0 WANDB_MODE=disabled
recipe_root="$PWD/reproduction/timit_bpe500_20261020"
job_info=$(qstat -f "$PBS_JOBID")
walltime=$(printf '%s\n' "$job_info" | awk '$1 == "Resource_List.walltime" {print $3}')
elapsed=$(printf '%s\n' "$job_info" | awk '$1 == "resources_used.walltime" {print $3}')
to_seconds() {
    local hours minutes seconds
    [[ "$1" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || return 1
    IFS=: read -r hours minutes seconds <<< "$1"
    printf '%s\n' "$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}
if ! limit_seconds=$(to_seconds "$walltime") || ! elapsed_seconds=$(to_seconds "$elapsed"); then
    printf 'Cannot determine PBS walltime; no training started\n' >&2
    exit 2
fi
SECONDS=0
budget=$((limit_seconds - elapsed_seconds - 900))
if (( requested_seconds < budget )); then budget=$requested_seconds; fi
if (( budget < 180 * ${#costs[@]} )); then
    printf 'Insufficient walltime after 900-second checkpoint/decode reserve\n' >&2
    exit 2
fi
if [[ -f "$recipe_root/DISPATCH.json" ]]; then
    queue=$(qstat)
    submitted=$("$LIBRI_PYTHON" -c 'import json,sys; records=json.load(open(sys.argv[1]))["jobs"]; print("\n".join(record["cost"]+" "+record["job"].split(".")[0] for record in records))' "$recipe_root/DISPATCH.json")
    while read -r cost job; do
        [[ "$selection" == both || "$selection" == "$cost" ]] || continue
        [[ "$job" != "${PBS_JOBID%%.*}" ]] || continue
        job_state=$(printf '%s\n' "$queue" | awk -v job="$job" '$1 == job || index($1, job ".") == 1 {print $3}')
        case "$job_state" in
            ''|FINISHED|COMPLETED|F|C) ;;
            *) printf '%s already has job %s (%s). Finish or cancel that job before interactive execution.\n' "$cost" "$job" "$job_state" >&2; exit 2 ;;
        esac
    done <<< "$submitted"
fi
mkdir -p "$recipe_root/jobs"
exec 9>"$recipe_root/jobs/.interactive.lock"
if ! flock -n 9; then
    printf 'Another BPE500 interactive invocation is active\n' >&2
    exit 2
fi
log="$recipe_root/jobs/interactive_${PBS_JOBID}_${selection}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
printf 'Job=%s costs=%s total_budget=%ss log=%s\n' "$PBS_JOBID" "$selection" "$budget" "$log"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$LIBRI_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No allocated CUDA GPU"'
"$LIBRI_PYTHON" -u "$recipe_root/train.py" --check
remaining_costs=${#costs[@]}
for cost in "${costs[@]}"; do
    model_dir="$recipe_root/models/ottc__$cost"
    if [[ -f "$model_dir/PROBE_RESULTS.json" ]] && "$LIBRI_PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["step"] >= 200 else 1)' "$model_dir/PROBE_RESULTS.json"; then
        printf '%s already completed its 200-step probe; skipping\n' "$cost"
        remaining_costs=$((remaining_costs - 1))
        continue
    fi
    cost_deadline=$((SECONDS + (budget - SECONDS) / remaining_costs))
    check_budget() {
        remaining=$((cost_deadline - SECONDS))
        if (( remaining < 180 )); then
            printf 'Budget reached. Rerun in a fresh allocation to resume.\n'
            exit 0
        fi
    }
    smoke_dir="$recipe_root/smoke/ottc__$cost"
    if [[ ! -f "$smoke_dir/PASS.json" ]]; then
        check_budget
        if [[ ! -f "$smoke_dir/LATEST_STATE.json" ]]; then
            "$LIBRI_PYTHON" -u "$recipe_root/train.py" --cost "$cost" --smoke --stop-after-steps 1 --max-seconds "$remaining"
        fi
        check_budget
        "$LIBRI_PYTHON" -u "$recipe_root/train.py" --cost "$cost" --smoke --max-seconds "$remaining"
        if [[ ! -f "$smoke_dir/PASS.json" ]]; then
            printf '%s smoke paused; probe not started. Rerun to resume.\n' "$cost"
            exit 0
        fi
    fi
    check_budget
    "$LIBRI_PYTHON" -u "$recipe_root/train.py" --cost "$cost" --stop-after-steps 200 --max-seconds "$remaining"
    remaining_costs=$((remaining_costs - 1))
done
printf 'Invocation finished. Results: %s/models/\n' "$recipe_root"