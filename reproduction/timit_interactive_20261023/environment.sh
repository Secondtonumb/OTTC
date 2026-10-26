# Source this inside an already allocated interactive GPU node.
source /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/environment.sh
export PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin:$PATH"
export WANDB_RUN_GROUP=timit_interactive_20261023
export OTTC_EVAL_UPLOAD=0
