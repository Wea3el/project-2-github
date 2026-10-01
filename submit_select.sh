#!/bin/bash
# Per-region choice between versions of the in-vivo masks (CPU only, see select_versions.py).
#   bash submit_select.sh cv      score each version and each selection rule on the held-out mice
#   bash submit_select.sh test    build the submission (default OUT=submission_sel.csv, RULE=pen LAM=0.2: set LAM
#                                 to the best penalty of the cv run)
# Options (environment): VERSIONS, EX, PAIRS, THR, RULE, LAM, OUT
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
MODE=${1:?usage: bash submit_select.sh cv|test}
DEP=""; [ "${2:-}" = "--after" ] && DEP="--dependency=afterok:$3"   # e.g. wait for the cyto3_x3_auto CV outputs
# the same versions for the held-out check and the test set (cyto3_x3_auto needs its CV outputs, README 6E)
V=${VERSIONS:-cyto3_x3_auto:-1,cyto3_x3_auto:-1.5,cyto3_x3_auto:-0.5,cyto3_x3:-1,cyto3_x2:-1,cyto3_x2:-0.5,cyto3_x2:0}
ARGS="$MODE --versions $V --ex ${EX:-cyto3_x3:0} --pairs ${PAIRS:-oof} --thr ${THR:-0.05} --rule ${RULE:-pen} --lam ${LAM:-0.2} --out ${OUT:-submission_sel.csv} --workers 8"
JID=$(sbatch --parsable $CPU_OPTS $DEP --job-name=cm_select --cpus-per-task=8 --mem=48G --time=03:00:00 \
      --output=logs/cm_select_%j.out --error=logs/cm_select_%j.out --export=ALL,CM_HOME="$CM_HOME" \
      --wrap="source '$CM_HOME/hpc_env.sh'; cm_run 'export OMP_NUM_THREADS=1; python select_versions.py $ARGS'"); JID=${JID%%;*}
echo "selection ($MODE) job $JID  ->  logs/cm_select_$JID.out"
