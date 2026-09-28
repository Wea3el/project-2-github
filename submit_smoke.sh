#!/bin/bash
# Quick end-to-end environment test (~10-15 min on one L4). Check logs/cm_smoke_<jobid>.out for "SMOKE TEST OK".
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
JID=$(sbatch --parsable $GPU_OPTS --export=ALL,CM_HOME="$CM_HOME" jobs/smoke.sbatch); JID=${JID%%;*}
echo "smoke test job $JID  ->  tail -f logs/cm_smoke_${JID}.out"
