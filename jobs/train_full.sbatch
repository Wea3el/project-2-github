#!/bin/bash
#SBATCH --job-name=cm_full
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
# Final models: train the chosen in-vivo (task 0) / ex-vivo (task 1) configuration on all three
# mice, then cache its outputs on the hidden test set. Needs CFG_IV and CFG_EX (see submit_full.sh).
source "$CM_HOME/hpc_env.sh"
read CFG MOD FOLD <<< "$(python3 "$CM_HOME/tasks.py" full "$CFG_IV" "$CFG_EX" get "$SLURM_ARRAY_TASK_ID")"
echo "final model: config=$CFG mod=$MOD (restart count ${SLURM_RESTART_COUNT:-0})"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,compute_mode --format=csv
if [ "$CFG" = "shipped" ]; then
  cm_run "python infer_flows_hpc.py --mod $MOD --config shipped --fold full --split hidden_test --model weights/full_$MOD"
else
  cm_run "python train_seg_hpc.py --mod $MOD --config $CFG --fold full && \
          python infer_flows_hpc.py --mod $MOD --config $CFG --fold full --split hidden_test"
fi
