#!/bin/bash
#SBATCH --job-name=cm_smoke
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment="gpu_mps=yes"
# ~10 min end-to-end check of the environment (GPU, cellpose, cyto3 download, data) on a tiny
# subset. Writes to runs_smoke/, never to runs/.
export CM_RUNS="$CM_HOME/runs_smoke" CM_MAX_REGIONS=2
source "$CM_HOME/hpc_env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
cm_run "python -c 'import torch; assert torch.cuda.is_available(), \"no GPU visible\"; print(\"GPU:\", torch.cuda.get_device_name(0))' && \
        python train_seg_hpc.py --mod ex --config base --fold subject_db6b8b --epochs 3 && \
        python train_seg_hpc.py --mod iv --config base --fold subject_db6b8b --epochs 3 && \
        python train_seg_hpc.py --mod ex --config cyto3_x3 --fold subject_db6b8b --epochs 3 && \
        python infer_flows_hpc.py --mod ex --config base --fold subject_db6b8b && \
        python infer_flows_hpc.py --mod iv --config base --fold subject_db6b8b && \
        python infer_flows_hpc.py --mod ex --config cyto3_x3 --fold subject_db6b8b && \
        python evaluate_cv.py --workers 2 --configs base,cyto3_x3 && echo 'SMOKE TEST OK'"
