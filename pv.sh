#!/bin/bash
# Prediction-only submission from cached test-set outputs (CPU job, no GPU):
#   bash pv.sh NAME CFG_IV CFG_EX "PRED_ARGS"      e.g.  bash pv.sh s1 cyto3_x3_auto cpsam2_x3 "--cp-iv -1 --cp-ex -1 --thr 0.05 --pairs oof"
# -> submission_NAME.csv, its settings appended to submissions_log.tsv, pictures in viz/submission_NAME/
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs; source ./cluster.sh
[ -e "submission_$1.csv" ] && { echo "submission_$1.csv already exists: submission files are never overwritten, pick a new name"; exit 1; }
CFG_IV=$2 CFG_EX=$3 OUT=submission_$1.csv PRED_ARGS="$4" sbatch $CPU_OPTS --export=ALL --job-name=cm_pv_$1 jobs/predict.sbatch
