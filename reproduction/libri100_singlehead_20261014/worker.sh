#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri100_singlehead_20261014/environment.sh
objective=${1:?objective}
phase=${2:?validate or train}
export CUDA_VISIBLE_DEVICES=${3:?GPU index}
exec 9> "reproduction/libri100_singlehead_20261014/.${objective}.lock"
flock -n 9
if [[ "$phase" == validate ]]; then
    "$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/train.py --objective "$objective" --smoke
    "$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/evaluate.py --objective "$objective" --smoke
else
    remaining=$((LIBRI_DEADLINE-$(date +%s)))
    if ((remaining < 600)); then exit 75; fi
    "$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/budget_train.py --objective "$objective" --max-wall-seconds "$remaining"
    "$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/evaluate.py --objective "$objective"
fi
