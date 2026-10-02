#!/bin/bash
# Final models + submission. Trains the chosen configurations on all three mice (GPU), then builds the
# submission on CPU.
#   bash submit_full.sh                      # settings from runs/best_config.json (the cross-validation)
#   bash submit_full.sh cyto3_x3 cyto3_x3    # or name the in-vivo and ex-vivo configs yourself
# Environment options:
#   BEST=runs/best_config_round2.json        another evaluation's result (evaluate_cv.py --tag round2)
#   PRED_ARGS="--cp-iv -1 --cp-ex 0 --thr 0.1 --pairs oof"   override thresholds / pair classifier
#   OUT=submission_v9.csv                    output file (default submission.csv) - use different names
#                                            when several runs are in flight
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
DEP=$(bash ./submit_setup.sh --dep)   # empty once the environment is built
export BEST=${BEST:-runs/best_config.json} OUT=${OUT:-submission.csv}
[ -e "$OUT" ] && { echo "$OUT already exists: submission files are never overwritten, set OUT to a new name"; exit 1; }
best() { python3 -c "import json; print(json.load(open('$BEST'))['$1'])"; }
export CFG_IV=${1:-$(best iv_config)} CFG_EX=${2:-$(best ex_config)}
case "$CFG_IV $CFG_EX" in *cpsam*) DEP=$(bash ./submit_setup.sh --sam --dep);; esac   # Cellpose-SAM: own overlay
if [ ! -f "$BEST" ] && [ -z "$PRED_ARGS" ]; then
  PRED_ARGS="--cp-iv -1 --cp-ex 0 --thr 0.1"   # defaults of submission v4
fi
export PRED_ARGS="--best $BEST ${PRED_ARGS:-}"
N=$(python3 tasks.py full "$CFG_IV" "$CFG_EX" count)   # ensembles "a+b" train every member
TODO=$(python3 tasks.py full "$CFG_IV" "$CFG_EX" todo)  # the others are trained and already ran on the test set
echo "in-vivo: $CFG_IV | ex-vivo: $CFG_EX | predict args: $PRED_ARGS | output: $OUT"
PDEP=""; JID=""
if [ -n "$TODO" ]; then
  JID=$(sbatch --parsable $GPU_OPTS $DEP --kill-on-invalid-dep=yes --export=ALL --array=$TODO jobs/train_full.sbatch); JID=${JID%%;*}
  echo "final training job $JID (tasks $TODO of 0-$((N - 1)))"
  PDEP="--dependency=afterok:$JID --kill-on-invalid-dep=yes"
else
  echo "models and test-set outputs already exist: no GPU job"
fi
PID=$(sbatch --parsable $CPU_OPTS --export=ALL $PDEP jobs/predict.sbatch); PID=${PID%%;*}
echo "prediction job $PID ${JID:+(waits for $JID)} ->  $CM_HOME/$OUT"
