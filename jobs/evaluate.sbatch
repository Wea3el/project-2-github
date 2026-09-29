#!/bin/bash
#SBATCH --job-name=cm_eval
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU only: scores the trained cross-validation models with the competition metric (see evaluate_cv.py).
# EVAL_STAGE = a | b | agg | all. Stage B runs as an array job: each array task takes one share of the
# combinations. Finished combinations are saved in runs/cv_B/ and skipped when rerun.
source "$CM_HOME/hpc_env.sh"
SHARD=""
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && SHARD="--shard $SLURM_ARRAY_TASK_ID --nshards ${EVAL_NSHARDS:?}"
# one thread per worker process: the parallelism comes from the worker pool
cm_run "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        python evaluate_cv.py --stage ${EVAL_STAGE:-all} --workers ${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}} $SHARD ${EVAL_ARGS:-}"
