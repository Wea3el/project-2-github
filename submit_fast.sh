#!/bin/bash
# One GPU job: cyto3 fine-tuned for ex-vivo + current in-vivo model -> submission.csv (~1 h on an L4)
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs
JID=$(sbatch --parsable --export=ALL,CM_HOME="$CM_HOME" jobs/fast.sbatch); JID=${JID%%;*}
echo "fast job $JID  ->  tail -f logs/cm_fast_${JID}.out   (result: $CM_HOME/submission.csv)"
