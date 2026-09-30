#!/bin/bash
# One-time environment build (~15-20 min). You normally don't run this yourself:
#     bash submit_setup.sh        (the submit_*.sh scripts also queue it automatically when needed)
# runs it as a CPU job. It can also be run by hand inside an interactive CPU session (srun ... --pty
# /bin/bash):  bash setup_overlay.sh
#
# It builds /scratch/$USER/overlay/cellmatch.ext3 (15 GB) containing Miniforge and the conda env
# /ext3/envs/cellmatch (PyTorch 2.5.1 + CUDA 12.1, Cellpose 3.1.1.3, ...). It then downloads the
# Cellpose cyto3 weights and checks the data.
# With OVERLAY=$SAM_OVERLAY CM_CELLPOSE=cellpose==4.2.1.1 (what `bash submit_setup.sh --sam` sets) it builds the
# Cellpose-SAM overlay instead: same layout and packages, cellpose 4, and the cpsam_v2 weights.
# The overlay is assembled under a temporary name. Only when everything has worked is it renamed to its
# final name and given a "<overlay>.ready" marker. An interrupted build therefore never leaves a
# half-written overlay that jobs could mount; rerunning simply starts over.
# Torch runs Apptainer, where writing into an overlay needs --fakeroot. Login nodes cap memory at 2 GB,
# too little for installing torch, so the build runs on a compute node.
set -euo pipefail
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"
source ./hpc_env.sh            # OVERLAY, IMAGE, CONDA_ENV, CELLPOSE_LOCAL_MODELS_PATH, overlay_ready
echo "host $(hostname) | overlay $OVERLAY | image $IMAGE"

if overlay_ready; then
  echo "environment already built: $OVERLAY"; exit 0
fi
if [ -f "$OVERLAY" ]; then
  if [ "$OVERLAY" = "/scratch/$USER/overlay/cellmatch.ext3" ] || [ "$OVERLAY" = "$SAM_OVERLAY" ]; then
    echo "removing an incomplete overlay left by an earlier attempt: $OVERLAY"
    rm -f "$OVERLAY" "$OVERLAY.gz"
  else
    echo "ERROR: $OVERLAY was not built by this script. To install the packages into it: bash setup_env.sh"; exit 1
  fi
fi

# ---------------------------------------------------------------- 0. internet access
# (needed for Miniforge, conda-forge, PyPI, the PyTorch wheels and the Cellpose weights)
if command -v curl >/dev/null; then
  for url in https://github.com https://conda.anaconda.org/conda-forge/ https://pypi.org/simple/ \
             https://download.pytorch.org/whl/cu121/ https://www.cellpose.org https://huggingface.co; do
    curl -sS -o /dev/null --max-time 30 "$url" || {
      echo "ERROR: $(hostname) cannot reach $url (no internet on this node?)."
      echo "Use the manual setup in README.md (section 3, 'By hand') from a node that has internet."; exit 2; }
  done
  echo "[0/4] internet access OK"
fi

BUILD="$OVERLAY.building"                         # renamed to $OVERLAY only at the very end
TMPB="/scratch/$USER/tmp-$(basename "$OVERLAY" .ext3)-build"   # installer, pip temp files (not /tmp: can be small)
rm -rf "$BUILD" "$TMPB"; mkdir -p "$(dirname "$OVERLAY")" "$TMPB"
done_ok=0
trap '[ "$done_ok" = 1 ] || rm -f "$BUILD"; rm -rf "$TMPB"' EXIT

# ---------------------------------------------------------------- 1. empty 15 GB overlay
SRC=""
for s in /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz; do
  [ -f "$s" ] && SRC="$s" && break
done
[ -n "$SRC" ] || { echo "ERROR: cannot find overlay-15GB-500K.ext3.gz"; exit 1; }
echo "[1/4] unpacking $SRC (15 GB, a few minutes) ..."
gunzip -c "$SRC" > "$BUILD"

# ---------------------------------------------------------------- 2. installer
echo "[2/4] downloading Miniforge ..."
curl -fsSL --retry 3 -o "$TMPB/miniforge.sh" \
     https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# ---------------------------------------------------------------- 3. install inside the overlay
# the commands run from a script file (not stdin), so nothing can accidentally read them as input
cat > "$TMPB/inner.sh" <<'INNER'
set -euo pipefail
export TMPDIR="$TMPB" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
bash "$TMPB/miniforge.sh" -b -p /ext3/miniforge3 > "$TMPB/miniforge.log"
cat > /ext3/env.sh <<'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
EOF
chmod +x /ext3/env.sh
source /ext3/env.sh
echo "  conda env $CONDA_ENV (python 3.11) ..."
conda create -y -q -p "$CONDA_ENV" python=3.11 > "$TMPB/conda.log"
conda activate "$CONDA_ENV"
echo "  torch 2.5.1 + CUDA 12.1 (large download) ..."
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
echo "  cellpose and the other packages ..."
# exact versions the pipeline was tested with
pip install -q "numpy==2.0.2" "$CM_CELLPOSE" "scikit-learn==1.8.0" "scikit-image==0.26.0" "tifffile==2026.3.3" \
               "opencv-python-headless==4.13.0.92" "pandas==3.0.2" "scipy==1.17.1"
case "$CM_CELLPOSE" in
  cellpose==4*) python -c "from cellpose.models import cache_model_path; cache_model_path('cpsam_v2'); print('  cpsam_v2 weights ready')" ;;
  *) python -c "from cellpose import models; models.CellposeModel(gpu=False, model_type='cyto3'); print('  cyto3 weights ready')" ;;
esac
python -c "
import torch, numpy, sklearn, skimage, pandas, scipy, cv2, tifffile
from importlib.metadata import version
print('  torch', torch.__version__, '(CUDA', str(torch.version.cuda) + ') | cellpose', version('cellpose'), '| numpy', numpy.__version__,
      '| sklearn', sklearn.__version__, '| skimage', skimage.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__)"
conda clean -ay > /dev/null
INNER
echo "[3/4] installing Python packages inside the overlay (the slow part, ~10-15 min) ..."
export CONDA_ENV CELLPOSE_LOCAL_MODELS_PATH TMPB CM_CELLPOSE="${CM_CELLPOSE:-cellpose==3.1.1.3}"
singularity exec --fakeroot --overlay "$BUILD":rw "$IMAGE" /bin/bash "$TMPB/inner.sh" < /dev/null

# ---------------------------------------------------------------- 4. finish
mv "$BUILD" "$OVERLAY"
touch "$OVERLAY.ready"
done_ok=1
echo "[4/4] overlay complete: $(ls -lh "$OVERLAY" | awk '{print $5}') $OVERLAY"

ntr=$(ls "$CM_DATA"/training/subject_*/region_*/exvivo.tif 2>/dev/null | wc -l)
nte=$(ls "$CM_DATA"/hidden_test/subject_*/region_*/exvivo.tif 2>/dev/null | wc -l)
if [ "$ntr" -eq 47 ] && [ "$nte" -eq 29 ] && [ -f "$CM_DATA/training/train_ground_truth.csv" ] && [ -f "$CM_DATA/sample_submission.csv" ]; then
  echo "data: $ntr/47 training regions, $nte/29 test regions -> OK"
else
  echo "data: $ntr/47 training regions, $nte/29 test regions -> PROBLEM: check $CM_DATA"
fi
echo "environment ready: $OVERLAY ($CONDA_ENV)"
