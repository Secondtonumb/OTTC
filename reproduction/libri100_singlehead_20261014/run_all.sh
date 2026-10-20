#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri100_singlehead_20261014/environment.sh
export LIBRI_DEADLINE=$(($(date +%s)+${LIBRI_BUDGET_SECONDS:-82800}))
exec 7> reproduction/libri100_singlehead_20261014/.suite.lock
flock -n 7
"$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/prepare.py
"$LIBRI_PYTHON" -u reproduction/check_js_cost.py
"$LIBRI_PYTHON" reproduction/libri100_singlehead_20261014/update_status.py 'three-node parallel continuation starting'
objectives=(ce jsd_upstream jsd_standard)
export OMPI_MCA_mca_base_env_list='PYTHONPATH;LD_LIBRARY_PATH;HF_HUB_OFFLINE;TRANSFORMERS_OFFLINE;OMP_NUM_THREADS;MKL_NUM_THREADS;OPENBLAS_NUM_THREADS;WANDB_MODE;WANDB_LOG_MODEL;WANDB_WATCH;WANDB_RUN_GROUP;LIBRI_DEADLINE'
for phase in validate train; do
    if [[ "$phase" == train && "${LIBRI_VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
    mpirun -np 3 --map-by ppr:1:node --bind-to none \
        bash reproduction/libri100_singlehead_20261014/rank_worker.sh "$phase"
    phase_status=0
    for i in 0 1 2; do
        status=$(cat "reproduction/libri100_singlehead_20261014/${phase}_${objectives[$i]}_exit_code.txt")
        if ((status != 0)); then phase_status=$status; fi
    done
    if ((phase_status != 0)); then
        "$LIBRI_PYTHON" reproduction/libri100_singlehead_20261014/update_status.py 'parallel phase paused or failed; inspect per-cost exit codes'
        exit "$phase_status"
    fi
    if [[ "$phase" == validate ]]; then
        "$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/gate.py
    fi
done
exec 8> reproduction/libri100_singlehead_20261014/.report.lock
flock 8
"$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/summarize.py
"$LIBRI_PYTHON" reproduction/libri100_singlehead_20261014/update_status.py 'all three training/evaluation runs complete'
