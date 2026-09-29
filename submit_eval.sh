#!/bin/bash
# Scores the trained cross-validation models (CPU jobs): stage A (one job), then stage B split over
# EVAL_SHARDS array tasks, then the aggregation that writes runs/best_config.json.
#   bash submit_eval.sh                 # after the training (submit_cv.sh) has finished
#   bash submit_eval.sh --after <jobid> # used by submit_cv.sh: start when that job has finished
# Rerunning is safe: stage A and finished stage-B combinations are reused.
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
N=${EVAL_SHARDS:-12}
AFTER=""
[ "${1:-}" = "--after" ] && AFTER="--dependency=afterany:$2"
AID=$(sbatch --parsable $CPU_OPTS $AFTER --time=01:00:00 --export=ALL,CM_HOME="$CM_HOME",EVAL_STAGE=a jobs/evaluate.sbatch); AID=${AID%%;*}
BID=$(sbatch --parsable $CPU_OPTS --dependency=afterok:"$AID" --kill-on-invalid-dep=yes --array=0-$((N - 1)) \
      --export=ALL,CM_HOME="$CM_HOME",EVAL_STAGE=b,EVAL_NSHARDS="$N" jobs/evaluate.sbatch); BID=${BID%%;*}
GID=$(sbatch --parsable $CPU_OPTS --dependency=afterany:"$BID" --time=00:30:00 --cpus-per-task=2 --mem=8G \
      --export=ALL,CM_HOME="$CM_HOME",EVAL_STAGE=agg jobs/evaluate.sbatch); GID=${GID%%;*}
echo "evaluation: stage A job $AID -> stage B array $BID ($N parts) -> aggregation $GID"
echo "  progress:  ls runs/cv_B/*.csv | wc -l      result: logs/cm_eval_${GID}.out and runs/best_config.json"
