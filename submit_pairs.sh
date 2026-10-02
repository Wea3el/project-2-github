#!/bin/bash
# Pair classifier trained on out-of-fold predicted masks (CPU only, no GPU): builds weights/pairs_NAME.pkl from the
# cross-validation outputs segmented with the given settings (those of the submission it is meant for), then compares it
# with the current classifier (oof) on the held-out mice. Existing classifier files are never overwritten.
#   NAME=oofs1 IV=cyto3_x3_auto:-1 EX=cpsam2_x3:-1 bash submit_pairs.sh      # then predict with --pairs oofs1
# Result: the last cm_eval log -> "Stage B [pairs_NAME]" table and runs/best_config_pairs_NAME.json
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
NAME=${NAME:?set NAME, the new classifier name (e.g. oofs1)}; IV=${IV:?set IV=config:cellprob}; EX=${EX:?set EX=config:cellprob}
[ -e "weights/pairs_$NAME.pkl" ] && { echo "weights/pairs_$NAME.pkl already exists: pick a new NAME"; exit 1; }
BID=$(sbatch --parsable $CPU_OPTS --job-name=cm_pairs --cpus-per-task=4 --mem=16G --time=03:00:00 \
      --output=logs/cm_pairs_%j.out --error=logs/cm_pairs_%j.out --export=ALL,CM_HOME="$CM_HOME" \
      --wrap="source '$CM_HOME/hpc_env.sh'; cm_run 'python build_pairs_oof.py --iv $IV --ex $EX --out $CM_HOME/weights/pairs_$NAME.pkl'"); BID=${BID%%;*}
echo "out-of-fold pair data: job $BID  (log logs/cm_pairs_$BID.out) -> weights/pairs_$NAME.pkl"
export EVAL_ARGS="--tag pairs_$NAME --configs ${IV%%:*},${EX%%:*} --pairs oof,$NAME --iv-top 0 --iv-extra $IV --ex-top 0 --ex-extra ${EX%%:*} --ex-cps ${EX##*:}"
EVAL_SHARDS=3 bash ./submit_eval.sh --after "$BID"
