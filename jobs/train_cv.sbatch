#!/bin/bash
#SBATCH --job-name=cm_cv
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
# One cross-validation task: train one (config, modality) with one mouse held out, then cache the
# model's outputs on the held-out mouse. Resumes from its checkpoint if preempted and requeued.
source "$CM_HOME/hpc_env.sh"
read CFG MOD FOLD <<< "$(python3 "$CM_HOME/tasks.py" get "$SLURM_ARRAY_TASK_ID")"
echo "task $SLURM_ARRAY_TASK_ID: config=$CFG mod=$MOD held-out=$FOLD  (restart count ${SLURM_RESTART_COUNT:-0})"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,compute_mode --format=csv
cm_run "python train_seg_hpc.py --mod $MOD --config $CFG --fold $FOLD && \
        python infer_flows_hpc.py --mod $MOD --config $CFG --fold $FOLD"
