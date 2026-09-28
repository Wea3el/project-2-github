#!/bin/bash
#SBATCH --job-name=cm_eval
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
# CPU only: scores every trained configuration on its held-out mouse with the competition metric
# and writes runs/cv_stageA.csv, runs/cv_stageB.csv, runs/best_config.json.
source "$CM_HOME/hpc_env.sh"
cm_run "python evaluate_cv.py --workers ${EVAL_WORKERS:-8}"
