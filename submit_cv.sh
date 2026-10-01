#!/bin/bash
# Cross-validation: one GPU task per (config, modality, held-out mouse), then a CPU evaluation job
# that starts automatically when they have all finished.
#   bash submit_cv.sh             # all configs in configs.CV_CONFIGS
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
DEP=$(bash ./submit_setup.sh --dep)   # empty once the environment is built
case "${CM_CV_TASKS:-}" in *cpsam*) DEP=$(bash ./submit_setup.sh --sam --dep);; esac   # Cellpose-SAM: own overlay
N=$(python3 tasks.py count)
JID=$(sbatch --parsable $GPU_OPTS $DEP --kill-on-invalid-dep=yes --export=ALL,CM_HOME="$CM_HOME" --array=0-$((N - 1)) jobs/train_cv.sbatch); JID=${JID%%;*}
echo "training array job $JID with $N tasks"
[ -n "${CM_NO_EVAL:-}" ] || bash ./submit_eval.sh --after "$JID"   # scoring starts when all training tasks have finished
echo "monitor: squeue -u \$USER"
