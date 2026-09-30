# Shared settings for all jobs. Sourced by the sbatch scripts and helper scripts.
# ---- defaults = the overlay built by setup_overlay.sh; point these elsewhere to reuse your own ----
export OVERLAY="${OVERLAY:-/scratch/$USER/overlay/cellmatch.ext3}"   # built by setup_overlay.sh
export IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
export CONDA_ENV="${CONDA_ENV:-/ext3/envs/cellmatch}"
# Cellpose-SAM configs (cpsam*) need cellpose 4: same layout, own overlay (bash submit_setup.sh --sam)
export SAM_OVERLAY="${SAM_OVERLAY:-/scratch/$USER/overlay/cellsam.ext3}"
# ----------------------------------------------------------------------------------------
export CM_HOME="${CM_HOME:-/scratch/$USER/project-2-github}"
export CM_DATA="${CM_DATA:-$CM_HOME/data}"
export CM_RUNS="${CM_RUNS:-$CM_HOME/runs}"
export CM_CACHE="${CM_CACHE:-$CM_HOME/cache}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/scratch/$USER/cellpose_models}"
export SINGULARITY_TMPDIR="/scratch/$USER/singularity-tmp"
export SINGULARITY_CACHEDIR="/scratch/$USER/singularity-cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$CM_HOME/logs" "$CM_RUNS" "$CELLPOSE_LOCAL_MODELS_PATH"

# the environment counts as built only once setup_overlay.sh / setup_env.sh finished and left this marker
overlay_ready() { [ -f "$OVERLAY" ] && [ -f "$OVERLAY.ready" ]; }

# cm_run "python script.py ..."  -> runs inside the container with the conda env active.
# --cleanenv empties the environment, so CUDA_VISIBLE_DEVICES (which GPU SLURM gave this job) is passed
# in explicitly; without it PyTorch picks GPU 0 of the node, which may belong to another job
# ("CUDA-capable device(s) is/are busy or unavailable").
cm_run() {
  overlay_ready || { echo "ERROR: environment not built yet ($OVERLAY.ready missing). Run: bash submit_setup.sh" >&2; return 1; }
  singularity exec --nv --cleanenv ${CUDA_VISIBLE_DEVICES:+--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} --overlay "$OVERLAY":ro "$IMAGE" /bin/bash -c "
    source /ext3/env.sh
    conda activate $CONDA_ENV
    export CM_DATA='$CM_DATA' CM_RUNS='$CM_RUNS' CM_CACHE='$CM_CACHE' \
           CELLPOSE_LOCAL_MODELS_PATH='$CELLPOSE_LOCAL_MODELS_PATH' CM_MAX_REGIONS='${CM_MAX_REGIONS:-0}' \
           PYTHONUNBUFFERED=1
    cd '$CM_HOME'
    $*
  "
}
