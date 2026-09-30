#!/bin/bash
# Pair classifier trained on out-of-fold predicted masks (CPU only, no GPU):
# builds weights/pairs_oof.pkl from the cross-validation outputs, then scores the shipped classifier (gt)
# against the new one (oof) and both combined, with the given segmentation settings.
#   bash submit_pairs.sh                                     # current best settings
#   IV=cyto3_x2:-0.5 EX=cyto3_x3:0 bash submit_pairs.sh
# Result: the last cm_eval log -> "best: ..." and runs/best_config_pairs.json
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
IV=${IV:-cyto3_x2:-0.5}; EX=${EX:-cyto3_x3:0}
BID=$(sbatch --parsable $CPU_OPTS --job-name=cm_pairs --cpus-per-task=4 --mem=16G --time=03:00:00 \
      --output=logs/cm_pairs_%j.out --error=logs/cm_pairs_%j.out --export=ALL,CM_HOME="$CM_HOME" \
      --wrap="source '$CM_HOME/hpc_env.sh'; cm_run 'python build_pairs_oof.py --iv $IV --ex $EX'"); BID=${BID%%;*}
echo "out-of-fold pair data: job $BID  (log logs/cm_pairs_$BID.out)"
export EVAL_ARGS="--tag pairs --configs ${IV%%:*},${EX%%:*} --pairs gt,oof,both --iv-top 0 --iv-extra $IV --ex-top 0 --ex-extra ${EX%%:*} --ex-cps ${EX##*:}"
EVAL_SHARDS=3 bash ./submit_eval.sh --after "$BID"
