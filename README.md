# CS-GY 6643 Project 2, Q5: Cell Matching × Segmentation

The pipeline:
1. Cellpose instance segmentation for in-vivo and ex-vivo images.
2. In-vivo → ex-vivo registration: Hough voting, then ICP, then a mouse-level consensus
   (`consensus2.py`).
3. Hungarian assignment of candidate pairs.
4. A gradient-boosted pair classifier that picks which pairs to report.

This repo retrains the segmentation networks on NYU HPC (cloud bursting, Lecture 1 setup) and
writes `submission.csv`. The competition data is **not** in the repo; you upload it to the
cluster separately (step 2).

## 1. Clone on HPC
VPN → https://ood.burst.hpc.nyu.edu → Clusters → shell, then:
```bash
cd /scratch/$USER
git clone <your-repo-url> project-2-github
cd project-2-github
```

## 2. Add the data
Upload `Project_2_Dataset.zip` to `/scratch/$USER/` with the OOD **Files** app, then:
```bash
cd /scratch/$USER/project-2-github
unzip -q ../Project_2_Dataset.zip && mv Project_2_Dataset data
ls data        # hidden_test  sample_submission.csv  training
```

## 3. Environment (once)
If you don't have a Singularity overlay with the conda env `/ext3/envs/torch` yet, create one
(Lecture 1, "Singularity Setup"):
```bash
mkdir -p /scratch/$USER/overlays && cd /scratch/$USER/overlays
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz . && gunzip overlay-15GB-500K.ext3.gz
mv overlay-15GB-500K.ext3 overlay.ext3
singularity exec --overlay /scratch/$USER/overlays/overlay.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
# --- now inside the container (prompt "Singularity>") ---
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
EOF
source /ext3/env.sh
conda create -p /ext3/envs/torch python=3.11 -y
conda activate /ext3/envs/torch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
exit
```
Then install this project's packages and download the pretrained Cellpose model:
```bash
cd /scratch/$USER/project-2-github
bash setup_env.sh
```
It should end with `47/47 training regions, 29/29 test regions -> OK` and `setup done`. If your
overlay lives somewhere other than `/scratch/$USER/overlays/overlay.ext3`, edit the first lines of
`hpc_env.sh`. No job may be running on the overlay while you run `setup_env.sh`.

## 4. Smoke test (~15 min)
```bash
bash submit_smoke.sh
tail -f logs/cm_smoke_*.out          # wait for: SMOKE TEST OK   (Ctrl-C to stop watching)
```

## 5. Run
**Fast (~1 h, 1 GPU job):** fine-tunes the pretrained Cellpose `cyto3` model for ex-vivo on all
three mice, keeps the current in-vivo model, and writes `submission.csv`.
```bash
bash submit_fast.sh
```
**Full comparison (several hours):** trains 5 configurations with each mouse held out in turn (30
GPU jobs). A CPU job then scores every setting with the competition metric and writes
`runs/best_config.json`. After that, train the winner on all mice:
```bash
bash submit_cv.sh
# when logs/cm_eval_*.out shows "best: ...":
bash submit_full.sh
```

## Monitoring and results
```bash
squeue -u $USER                      # your jobs
tail -f logs/<job log>.out           # progress
scancel <jobid>                      # cancel
```
The result is `/scratch/$USER/project-2-github/submission.csv`; download it with the OOD Files
app. If a job is preempted, it is requeued and resumes from its last checkpoint. Rerunning a
submit command skips finished steps.

GPU use: about 1 L4-hour for the fast run and 10–20 for the full comparison (the class budget is
300).

## Files
| file | role |
|---|---|
| `configs.py` | training configurations and evaluation grids |
| `train_seg_hpc.py` | resumable Cellpose training / cyto3 fine-tuning |
| `infer_flows_hpc.py` | runs a model and caches its outputs per region |
| `evaluate_cv.py` | leave-one-mouse-out scoring with the competition metric |
| `predict_test.py` | test-set masks → registration → pairs → `submission.csv` |
| `cm_pipeline.py`, `consensus2.py`, `match.py`, `pipeline.py` | matching pipeline |
| `cmutil.py`, `segdata.py`, `seg_infer.py`, `shape_ops.py` | utilities, metric, Cellpose inference |
| `weights/` | current segmentation weights (`full_iv`, `full_ex`) and pair classifier |
| `hpc_env.sh`, `setup_env.sh`, `submit_*.sh`, `jobs/*.sbatch` | HPC job scripts |
| `cellmatch_colab_train.ipynb` | the same runs on Google Colab (expects the code and data in `MyDrive/cellmatch/`) |
