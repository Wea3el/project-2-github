#!/bin/bash
# Cross-validation: one GPU task per (config, modality, held-out mouse), then a CPU evaluation job
# that starts automatically when they have all finished.
#   bash submit_cv.sh             # all configs in configs.CV_CONFIGS
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
N=$(python3 tasks.py count)
JID=$(sbatch --parsable --export=ALL,CM_HOME="$CM_HOME" --array=0-$((N - 1)) jobs/train_cv.sbatch); JID=${JID%%;*}
echo "training array job $JID with $N tasks"
EID=$(sbatch --parsable --export=ALL,CM_HOME="$CM_HOME" --dependency=afterany:"$JID" jobs/evaluate.sbatch); EID=${EID%%;*}
echo "evaluation job $EID (waits for $JID)  ->  results in logs/cm_eval_${EID}.out and runs/best_config.json"
echo "monitor: squeue -u \$USER"
