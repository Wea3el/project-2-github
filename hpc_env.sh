# Shared settings for all jobs. Sourced by the sbatch scripts and helper scripts.
# ---- edit these to match your setup from Lecture 1 ("Singularity Setup") -------------
export OVERLAY="${OVERLAY:-/scratch/$USER/overlays/overlay.ext3}"
export IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
export CONDA_ENV="${CONDA_ENV:-/ext3/envs/torch}"
# ----------------------------------------------------------------------------------------
export CM_HOME="${CM_HOME:-/scratch/$USER/project-2-github}"
export CM_DATA="${CM_DATA:-$CM_HOME/data}"
export CM_RUNS="${CM_RUNS:-$CM_HOME/runs}"
export CM_CACHE="${CM_CACHE:-$CM_HOME/cache}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/scratch/$USER/cellpose_models}"
export SINGULARITY_TMPDIR="/scratch/$USER/singularity-tmp"
export SINGULARITY_CACHEDIR="/scratch/$USER/singularity-cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$CM_HOME/logs" "$CM_RUNS" "$CELLPOSE_LOCAL_MODELS_PATH"

# cm_run "python script.py ..."  -> runs inside the container with the conda env active
cm_run() {
  singularity exec --nv --cleanenv --overlay "$OVERLAY":ro "$IMAGE" /bin/bash -c "
    source /ext3/env.sh
    conda activate $CONDA_ENV
    export CM_DATA='$CM_DATA' CM_RUNS='$CM_RUNS' CM_CACHE='$CM_CACHE' \
           CELLPOSE_LOCAL_MODELS_PATH='$CELLPOSE_LOCAL_MODELS_PATH' CM_MAX_REGIONS='${CM_MAX_REGIONS:-0}' \
           PYTHONUNBUFFERED=1
    cd '$CM_HOME'
    $*
  "
}
