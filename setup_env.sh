#!/bin/bash
# One-time setup: installs the Python packages into your overlay's conda env and downloads the
# pretrained Cellpose "cyto3" weights. Run it in an interactive CPU session (login nodes cap memory at
# 2 GB), with no jobs currently using the overlay (it is mounted read-write here):
#     srun --account=<account> --partition=cpu_short --cpus-per-task=4 --mem=16G --time=01:30:00 --pty /bin/bash
#     bash setup_env.sh
set -e
cd "$(dirname "$0")"; export CM_HOME="$(pwd)"
source ./hpc_env.sh
echo "overlay: $OVERLAY"; echo "image:   $IMAGE"
[ -f "$OVERLAY" ] || { echo "overlay not found - do Lecture 1 'Singularity Setup' first, then edit hpc_env.sh"; exit 1; }
singularity exec --fakeroot --overlay "$OVERLAY":rw "$IMAGE" /bin/bash -c "
  set -e
  source /ext3/env.sh
  conda activate $CONDA_ENV
  python -c 'import torch; print(\"torch\", torch.__version__)' || \
     pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install --no-cache-dir 'numpy==2.0.2' 'cellpose==3.1.1.3' 'scikit-learn==1.8.0' 'scikit-image==0.26.0' 'tifffile==2026.3.3' \
              'opencv-python-headless==4.13.0.92' 'pandas==3.0.2' 'scipy==1.17.1'
  export CELLPOSE_LOCAL_MODELS_PATH='$CELLPOSE_LOCAL_MODELS_PATH'
  python -c 'from cellpose import models; models.CellposeModel(gpu=False, model_type=\"cyto3\"); print(\"cyto3 weights ready\")'
  python -c 'import torch, cellpose, sklearn, numpy; print(\"torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| sklearn\", sklearn.__version__)'
"
# data sanity check
python3 - <<'EOF'
import os, glob
d = os.environ["CM_DATA"]
tr = glob.glob(f"{d}/training/subject_*/region_*/exvivo.tif"); te = glob.glob(f"{d}/hidden_test/subject_*/region_*/exvivo.tif")
ok = len(tr) == 47 and len(te) == 29 and os.path.exists(f"{d}/training/train_ground_truth.csv") and os.path.exists(f"{d}/sample_submission.csv")
print(f"data: {len(tr)}/47 training regions, {len(te)}/29 test regions ->", "OK" if ok else f"PROBLEM: check {d}")
EOF
touch "$OVERLAY.ready"     # tells the submit scripts the environment is usable
echo "setup done"
