#!/bin/bash
# Final models + submission. Trains the chosen configurations on all three mice (GPU), then builds
# submission.csv on CPU.
#   bash submit_full.sh                      # use runs/best_config.json from the cross-validation
#   bash submit_full.sh cyto3_x3 cyto3_x3    # or name the in-vivo and ex-vivo configs yourself
# Thresholds (cp_iv, cp_ex, pair thr) come from runs/best_config.json unless you pass
#   PRED_ARGS="--cp-iv -1 --cp-ex 0 --thr 0.1" bash submit_full.sh ...
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
best() { python3 -c "import json; print(json.load(open('runs/best_config.json'))['$1'])"; }
CFG_IV=${1:-$(best iv_config)}; CFG_EX=${2:-$(best ex_config)}
if [ ! -f runs/best_config.json ] && [ -z "$PRED_ARGS" ]; then
  export PRED_ARGS="--cp-iv -1 --cp-ex 0 --thr 0.1"   # defaults of submission v4
fi
echo "in-vivo config: $CFG_IV | ex-vivo config: $CFG_EX | extra predict args: ${PRED_ARGS:-none}"
JID=$(sbatch --parsable $GPU_OPTS --export=ALL,CM_HOME="$CM_HOME",CFG_IV="$CFG_IV",CFG_EX="$CFG_EX" --array=0-1 jobs/train_full.sbatch); JID=${JID%%;*}
echo "final training job $JID"
PID=$(sbatch --parsable $CPU_OPTS --export=ALL,CM_HOME="$CM_HOME",CFG_IV="$CFG_IV",CFG_EX="$CFG_EX",PRED_ARGS="${PRED_ARGS:-}" \
      --dependency=afterok:"$JID" jobs/predict.sbatch); PID=${PID%%;*}
echo "prediction job $PID (waits for $JID)  ->  $CM_HOME/submission.csv"
