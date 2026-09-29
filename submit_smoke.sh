#!/bin/bash
# Quick end-to-end environment test (~10-15 min on one L4). Check logs/cm_smoke_<jobid>.out for "SMOKE TEST OK".
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
DEP=$(bash ./submit_setup.sh --dep)   # empty once the environment is built
JID=$(sbatch --parsable $GPU_OPTS $DEP --kill-on-invalid-dep=yes --export=ALL,CM_HOME="$CM_HOME" jobs/smoke.sbatch); JID=${JID%%;*}
echo "smoke test job $JID  ->  tail -f logs/cm_smoke_${JID}.out"
