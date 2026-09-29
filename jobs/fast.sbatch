#!/bin/bash
#SBATCH --job-name=cm_fast
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
# "Fast" run in one job: current (shipped) in-vivo model + cyto3 fine-tuned for ex-vivo on all three
# mice -> submission.csv. Resumes from its checkpoint if preempted and requeued.
source "$CM_HOME/hpc_env.sh"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,compute_mode --format=csv
cm_run "set -e
        python infer_flows_hpc.py --mod iv --config shipped --fold full --split hidden_test --model weights/full_iv
        python train_seg_hpc.py   --mod ex --config cyto3_x3 --fold full
        python infer_flows_hpc.py --mod ex --config cyto3_x3 --fold full --split hidden_test
        python predict_test.py --iv-config shipped --ex-config cyto3_x3 --cp-iv -1 --cp-ex 0 --thr 0.1 --out submission.csv"
