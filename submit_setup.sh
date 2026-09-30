#!/bin/bash
# Builds the environment (setup_overlay.sh) in a CPU job, once (~15-20 min).
#   bash submit_setup.sh          submits the build, or says it is already built / already queued
#   bash submit_setup.sh --dep    used by the other submit_*.sh scripts: queues the build if needed and
#                                 prints "--dependency=afterok:<jobid>" so their jobs wait for it
#   bash submit_setup.sh --sam [--dep]   the same for the Cellpose-SAM overlay (cellpose 4)
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
source ./cluster.sh
source ./hpc_env.sh
NAME=cm_setup
if [ "${1:-}" = "--sam" ]; then
  shift; export OVERLAY="$SAM_OVERLAY" CM_CELLPOSE="cellpose==4.2.1.1"; NAME=cm_setup_sam
fi
if overlay_ready; then
  [ "${1:-}" = "--dep" ] || echo "environment already built: $OVERLAY"
  exit 0
fi
JID=$(squeue -h -u "$USER" -n $NAME -o %i | head -1)
if [ -z "$JID" ]; then
  JID=$(sbatch --parsable $CPU_OPTS --job-name=$NAME --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
        --time=01:30:00 --output=logs/${NAME}_%j.out --error=logs/${NAME}_%j.out \
        --export=ALL,CM_HOME="$CM_HOME" --wrap="bash '$CM_HOME/setup_overlay.sh'")
  JID=${JID%%;*}
  MSG="environment build submitted: job $JID"
else
  MSG="environment build already queued/running: job $JID"
fi
if [ "${1:-}" = "--dep" ]; then
  echo "$MSG (the next job waits for it; if the build fails, that job is cancelled automatically)" >&2
  echo "--dependency=afterok:$JID"
else
  echo "$MSG  ->  tail -f logs/${NAME}_$JID.out   (finishes with 'environment ready')"
fi
