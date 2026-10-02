#!/bin/bash
# One-off CPU job:  bash cpu.sh NAME "python script.py args"   -> logs/cm_NAME_<jobid>.out   (no single quotes in the command)
#   e.g. bash cpu.sh diag_f3 "python diagnose_cv.py --iv cyto3_x3_auto:-1 --ex cpsam2_x3:-1 --thr 0.05 --pairs oof+flow_ex=0.3 --out runs/diag_f3.csv"
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"; mkdir -p logs; source ./cluster.sh
sbatch $CPU_OPTS --job-name=cm_$1 --cpus-per-task=${CPUS:-8} --mem=${MEM:-32G} --time=${TIME:-03:00:00} \
  --output=logs/cm_$1_%j.out --error=logs/cm_$1_%j.out --export=ALL,CM_HOME="$CM_HOME" \
  --wrap="source '$CM_HOME/hpc_env.sh'; cm_run '$2'"
