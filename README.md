# CS-GY 6643 Project 2, Q5: Cell Matching × Segmentation

The pipeline:
1. Cellpose instance segmentation for in-vivo and ex-vivo images.
2. In-vivo → ex-vivo registration: Hough voting, then ICP, then a mouse-level consensus
   (`consensus2.py`).
3. Hungarian assignment of candidate pairs.
4. A gradient-boosted pair classifier that picks which pairs to report.

This repo retrains the segmentation networks on NYU HPC (Torch, or the Lecture 1 cloud-bursting
setup) and writes `submission.csv`. The competition data is **not** in the repo; you upload it to the
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

## 3. Environment (once, ~15-20 min)
```bash
bash submit_setup.sh
tail -f logs/cm_setup_*.out          # finishes with: environment ready
```
This runs `setup_overlay.sh` as a CPU job. It builds `/scratch/$USER/overlay/cellmatch.ext3` with a
conda env `/ext3/envs/cellmatch`: PyTorch 2.5.1 + CUDA 12.1, Cellpose 3.1.1.3 and the other packages
at the versions the pipeline was tested with. It also downloads the Cellpose `cyto3` weights and
checks the data. You can also skip this step: every `submit_*.sh` queues the build automatically when
the environment isn't there yet, and its GPU jobs wait for it (if the build fails, they are cancelled).

The overlay is only usable once `/scratch/$USER/overlay/cellmatch.ext3.ready` exists. The build writes
that file last, so a half-finished overlay is never used. Two things about Torch: it runs Apptainer,
where writing into an overlay needs `--fakeroot`; and login nodes cap memory at 2 GB, too little to
install torch. So the build always runs on a compute node.

**By hand** (same result, in an interactive session):
```bash
srun --account=torch_pr_355_general --partition=cpu_short --cpus-per-task=4 --mem=16G --time=01:30:00 --pty /bin/bash
cd /scratch/$USER/overlay && rm -f cellmatch.ext3*
gunzip -c /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz > cellmatch.ext3
singularity exec --fakeroot --overlay cellmatch.ext3:rw /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
#   inside: install Miniforge to /ext3/miniforge3, write /ext3/env.sh, conda create -p /ext3/envs/cellmatch
#   python=3.11, then pip install the versions in requirements.txt (torch from the cu121 index); exit
touch /scratch/$USER/overlay/cellmatch.ext3.ready
```

To reuse an overlay you already have instead, set `OVERLAY` and `CONDA_ENV` at the top of
`hpc_env.sh` and run `bash setup_env.sh` in an interactive CPU session. It only installs the packages
into that env and then writes the `.ready` marker. No job may be using the overlay while either script
runs.

## 3b. Pick the account and partitions (`cluster.sh`)
All submit scripts read the SLURM account and partitions from `cluster.sh`. The defaults are for the
NYU **Torch** cluster (`torch_pr_355_general`, GPU partition `l40s_public`, CPU partition `cpu_short`).
For the course's cloud-bursting cluster, use the values in the comments at the top of that file. Check
that a combination is accepted, without actually submitting anything:
```bash
sbatch --test-only --account=torch_pr_355_general --partition=l40s_public --gres=gpu:1 -t 1:00:00 --wrap=hostname
```
A line like `Job ... to start at ...` means it's accepted. An error names what's wrong.

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
**Full comparison (a few hours):** trains 5 configurations with each mouse held out in turn (30 GPU
jobs, ~10 min each on an L40S). CPU jobs then score the settings with the competition metric
(`submit_eval.sh`: stage A ranks the segmentations, stage B runs the full pipeline for the best
combinations, split over 12 array tasks) and write `runs/best_config.json`. After that, train the
winner on all mice:
```bash
bash submit_cv.sh                    # training + scoring (scoring starts when training ends)
# when logs/cm_eval_*.out of the last job shows "best: ...":
bash submit_full.sh
```
Every scored combination is saved in `runs/cv_B/` as soon as it finishes. If a scoring job is
interrupted, `bash submit_eval.sh` resumes it and skips the finished ones.

## 6. More experiments (all can run at the same time)
Each run writes to its own folders, so they don't interfere. Use a different `OUT=` for each
submission you build.

**A. New segmentation variants (GPU, ~30 cross-validation tasks).** Trains the round-2 configs of
`configs.py` with each mouse held out (longer training, 4x upsampling, brightness/contrast
augmentation, the Cellpose `nuclei` model, extra random seeds, test-time augmentation), then scores
everything with the tag `round2`. The flow ensembles in `configs.ENSEMBLES` (averaged outputs of
several models) are scored automatically once their members exist:
```bash
CM_CV_TASKS="$(python3 -c 'from configs import ROUND2; print(ROUND2)')" EVAL_ARGS="--tag round2" bash submit_cv.sh
# result: runs/best_config_round2.json, then
BEST=runs/best_config_round2.json OUT=submission_round2.csv bash submit_full.sh
```

**B. Pair classifier trained on predicted masks (CPU only).** The shipped pair classifier was fitted on
candidate pairs from ground-truth masks; this fits it on out-of-fold predicted masks (what it sees on
the test set) and compares gt / oof / both:
```bash
bash submit_pairs.sh
# result: runs/best_config_pairs.json; to use the winner with the current models:
PRED_ARGS="--pairs oof" OUT=submission_pairs.csv bash submit_full.sh cyto3_x2 cyto3_x3
```

**C. Direct leaderboard tries** (no cross-validation; the ex-vivo model is already trained):
```bash
OUT=submission_tta.csv PRED_ARGS="--cp-iv -0.5 --cp-ex 0 --thr 0.2" bash submit_full.sh cyto3_x2 cyto3_x3_tta
```
Config names: `<config>_tta` = that model with test-time augmentation; `<config>_auto` = that model
rescaled per region so the median cell size matches training (for test mice with bigger or smaller
cells); `a+b` = ensemble of a and b (same upsampling), e.g. `cyto3_x3+cyto3_x3_lr1`.

**D. Registration variants (CPU only).** A pipeline variant is written `<pair data>+option=value...`
(options in `cm_pipeline.METHOD_DEFAULTS`: `weak=keep` keeps regions whose registration is not
confident instead of dropping them, `zwin`/`zalone` lower the confidence needed, `bright=0.5`
registers with the brightest half of the in-vivo cells, `grow_iv`/`grow_ex` grow the masks). Compare
them with the current models:
```bash
EVAL_ARGS="--tag reg --configs cyto3_x2,cyto3_x3 --iv-top 0 --iv-extra cyto3_x3:-1,cyto3_x2:-0.5 --ex-top 0 --ex-extra cyto3_x3 --ex-cps 0 --pairs oof,oof+weak=keep,oof+zwin=3+zalone=6,oof+bright=0.5,oof+weak=keep+zwin=3+zalone=6" EVAL_SHARDS=6 bash submit_eval.sh
# result: runs/best_config_reg.json; use a variant for the test set with PRED_ARGS="--pairs oof+weak=keep ..."
```

**E. Per-region choice between mask versions (CPU only).** Small changes to the in-vivo masks move
the registration of some regions. This runs the pipeline once per version (model and cellprob
threshold) and keeps, per region, the version the pair classifier is most confident about:
```bash
bash submit_select.sh cv     # held-out score of every version and selection rule -> runs/select_cv.csv
bash submit_select.sh test   # the submission -> submission_sel.csv (submit only if cv shows a clear gain)
```

**F. Cellpose-SAM (GPU, 12 cross-validation tasks).** Cellpose 4's `cpsam_v2` model (a ViT-L image
encoder from Segment Anything, trained on cells 7.5-120 px across) fine-tuned per modality with its
authors' recipe (AdamW, lr 1e-5, weight decay 0.1, batch 1, 100 epochs) at 3x and 2x upsampling
(`cpsam2_x3`, `cpsam2_x2` in `configs.py`). It needs cellpose 4, so it runs in its own overlay,
`/scratch/$USER/overlay/cellsam.ext3`. The first run queues that build (`submit_setup.sh --sam`, a
~20 min CPU job that also downloads the weights) and the GPU tasks wait for it:
```bash
CM_CV_TASKS="cpsam2_x3:iv cpsam2_x3:ex cpsam2_x2:iv cpsam2_x2:ex" \
EVAL_ARGS="--tag sam --configs cyto3_x3,cyto3_x2,cpsam2_x3,cpsam2_x2 --iv-top 2 --ex-top 2 --pairs oof" bash submit_cv.sh
# result: runs/best_config_sam.json; a test submission, e.g. with Cellpose-SAM for both images:
OUT=submission_sam.csv PRED_ARGS="--cp-iv -1 --cp-ex 0 --thr 0.05 --pairs oof" bash submit_full.sh cpsam2_x3_auto cpsam2_x3
```
Scoring and prediction (CPU) stay in the Cellpose 3 environment: they only read the cached network
outputs, and both Cellpose versions turn those into the same masks.

**G. Test-set variants from cached outputs, and a threshold check.** A submission from outputs that
already exist (other thresholds, an ensemble `a+b`, ...) needs no GPU, only the prediction job:
```bash
export CM_HOME=$PWD; source cluster.sh
pv() { CFG_IV=$2 CFG_EX=$3 OUT=submission_$1.csv PRED_ARGS="$4" sbatch $CPU_OPTS --export=ALL --job-name=cm_pv_$1 jobs/predict.sbatch; }
pv t1 cyto3_x3_auto cyto3_x3 "--cp-iv -1 --cp-ex -0.5 --thr 0.05 --pairs oof"      # -> submission_t1.csv
```
The thresholds tuned on the training mice did not carry over to the test mice (ex-vivo 0 -> 0.5:
+0.016 held-out, -0.075 on the leaderboard). `calib_check.py` shows how each threshold changes the
masks per mouse (cells per region, median cell area, cell probability inside the cells; on training
mice also the ground truth and PQ), training mice from the cross-validation models, test mice from
the final ones:
```bash
sbatch $CPU_OPTS --job-name=cm_calib --cpus-per-task=8 --mem=32G --time=01:00:00 --output=logs/cm_calib_%j.out \
  --export=ALL,CM_HOME=$PWD --wrap="source $PWD/hpc_env.sh; cm_run 'export OMP_NUM_THREADS=1; python calib_check.py --configs cyto3_x3,cpsam2_x3'"
```

**H. Training variants judged on the leaderboard.** Each trains one modality on all three mice and
keeps the other as in v13, so every submission tests one change (no cross-validation: the held-out
mice did not predict the test mice). New configs in `configs.py`: `*_pl` self-training (the test images
added with v13's masks as labels), `*_sr` wider size augmentation, `cpsam2_x3_aug` brightness/contrast
augmentation, `cpsam2_x3_long` 3x longer, `cpsam_x3` the original Cellpose-SAM weights. A config name
ending in `_auto` is the in-vivo run with per-region size rescaling.
```bash
P="--cp-iv -1 --cp-ex 0 --thr 0.05 --pairs oof"
ex() { OUT=submission_$1.csv PRED_ARGS="$P" bash submit_full.sh cyto3_x3_auto $2; }   # new ex-vivo model
iv() { OUT=submission_$1.csv PRED_ARGS="$P" bash submit_full.sh $2 cyto3_x3; }        # new in-vivo model
ex e1 cyto3_x3_pl; iv i1 cyto3_x3_pl_auto
```
`predict_test.py --pairs NAME` also reads a pair classifier from `weights/pairs_NAME.pkl`, e.g. one
built on Cellpose-SAM masks with `build_pairs_oof.py --out weights/pairs_oofsam.pkl`.

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
| `train_seg_hpc.py` | resumable Cellpose training / cyto3 and Cellpose-SAM fine-tuning |
| `infer_flows_hpc.py` | runs a model and caches its outputs per region |
| `select_versions.py`, `submit_select.sh` | per-region choice between versions of the in-vivo masks |
| `calib_check.py` | how the cell-probability threshold changes the masks, training vs test mice |
| `build_pairs_oof.py`, `submit_pairs.sh` | pair classifier from out-of-fold predicted masks |
| `evaluate_cv.py`, `submit_eval.sh` | leave-one-mouse-out scoring with the competition metric (resumable, split over CPU jobs) |
| `predict_test.py` | test-set masks → registration → pairs → `submission.csv` |
| `cm_pipeline.py`, `consensus2.py`, `match.py`, `pipeline.py` | matching pipeline |
| `cmutil.py`, `segdata.py`, `seg_infer.py`, `shape_ops.py` | utilities, metric, Cellpose inference |
| `weights/` | current segmentation weights (`full_iv`, `full_ex`) and pair classifier |
| `cluster.sh` | SLURM account + GPU/CPU partitions used by all submit scripts |
| `setup_overlay.sh`, `submit_setup.sh` | one-time build of the Singularity overlay + conda env (as a CPU job); `--sam`: the Cellpose-SAM overlay |
| `hpc_env.sh`, `setup_env.sh`, `submit_*.sh`, `jobs/*.sbatch` | HPC job scripts |
| `cellmatch_colab_train.ipynb` | the same runs on Google Colab (expects the code and data in `MyDrive/cellmatch/`) |
