"""Segmentation training configurations compared by leave-one-mouse-out cross-validation.

pretrained : None -> train a Cellpose U-Net from scratch; "cyto3" -> fine-tune the pretrained
             Cellpose generalist model (downloaded once by setup_env.sh)
nbase      : channel widths of the from-scratch network
up         : cells are upsampled by this factor for the network (native cells are ~10 px across;
             cyto3 was trained on ~30 px cells, so up=3 matches its training scale)
bsize      : training crop size fed to the network (after upsampling)
nimg       : training tiles sampled per epoch
"""

CONFIGS = {
    # the configuration behind submissions v3/v4 (small net from scratch, native resolution)
    "base": dict(pretrained=None, nbase=(16, 32, 64, 128), up=1, bsize=128, epochs=300, nimg=128,
                 batch=8, lr=0.005),
    # wider from-scratch network, longer schedule
    "big": dict(pretrained=None, nbase=(32, 64, 128, 256), up=1, bsize=128, epochs=500, nimg=256,
                batch=16, lr=0.005),
    # fine-tune pretrained cyto3, cells upsampled 2x / 3x
    "cyto3_x2": dict(pretrained="cyto3", up=2, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005),
    "cyto3_x3": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005),
    "cyto3_x3_lr1": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.001),
    # round 2 (after the first cross-validation: cyto3_x3 best for ex-vivo, cyto3_x2 for in-vivo)
    "cyto3_x3_long": dict(pretrained="cyto3", up=3, bsize=224, epochs=600, nimg=512, batch=8, lr=0.005),
    "cyto3_x4": dict(pretrained="cyto3", up=4, bsize=256, epochs=300, nimg=256, batch=8, lr=0.005),
    # random brightness/contrast/noise on the training crops (test mice look different)
    "cyto3_x3_aug": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, aug=True),
    "nuclei_x2": dict(pretrained="nuclei", up=2, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005),
    # same recipe, other random seeds: members of flow ensembles (see ENSEMBLES)
    "cyto3_x3_s1": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, seed=1),
    "cyto3_x3_s2": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, seed=2),
    "cyto3_x2_s1": dict(pretrained="cyto3", up=2, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, seed=1),
    "cyto3_x2_s2": dict(pretrained="cyto3", up=2, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, seed=2),
    # Cellpose-SAM (cellpose 4, runs in its own overlay, see README 6F): the pretrained "cpsam_v2" ViT-L
    # fine-tuned with its authors' recipe (AdamW, lr 1e-5, weight decay 0.1, batch 1, 100 epochs, 256 px crops).
    # It was trained on cells 7.5-120 px across (mean 30), so native ~10 px cells are upsampled 3x / 2x.
    "cpsam2_x3": dict(pretrained="cpsam_v2", up=3, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1),
    "cpsam2_x2": dict(pretrained="cpsam_v2", up=2, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1),
    # with the brightness/contrast/noise augmentation of cyto3_x3_aug (robustness to the test mice)
    "cpsam2_x3_aug": dict(pretrained="cpsam_v2", up=3, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1, aug=True),
    "cpsam2_x3_long": dict(pretrained="cpsam_v2", up=3, bsize=256, epochs=300, nimg=128, batch=1, lr=1e-5, wd=0.1),
    "cpsam_x3": dict(pretrained="cpsam", up=3, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1),  # April 2025 weights
    # wider size augmentation (cells scaled 0.5-1.5x instead of 0.75-1.25x): ground-truth cell sizes differ a lot
    # between mice (median ex-vivo cell area 54-98 px^2 on the three training mice)
    "cyto3_x3_sr": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005, sr=1.0),
    "cpsam2_x3_sr": dict(pretrained="cpsam_v2", up=3, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1, sr=1.0),
    # self-training on the unlabelled test images: they are added to the training tiles with the masks of the
    # current best submission (v13) as labels
    "cyto3_x3_pl": dict(pretrained="cyto3", up=3, bsize=224, epochs=300, nimg=256, batch=8, lr=0.005,
                        pseudo=dict(iv="cyto3_x3_auto:-1", ex="cyto3_x3:0")),
    "cpsam2_x3_pl": dict(pretrained="cpsam_v2", up=3, bsize=256, epochs=100, nimg=128, batch=1, lr=1e-5, wd=0.1,
                         pseudo=dict(iv="cyto3_x3_auto:-1", ex="cyto3_x3:0")),
    # the shipped weights behind submissions v3/v4 (trained on all three mice): only for
    # predictions (runs/shipped/<mod>/full) or sanity checks, never for cross-validation
    "shipped": dict(pretrained=None, nbase=(16, 32, 64, 128), up=1),
}

# Two more kinds of "config" need no entry above:
#   "<config>_tta"  - the trained <config> model run with test-time augmentation (flipped tiles averaged)
#   "a+b+c"         - ensemble: the flows of configs a, b, c averaged (same "up" required)
ENSEMBLES = ["cyto3_x3+cyto3_x3_s1+cyto3_x3_s2", "cyto3_x3+cyto3_x3_lr1", "cyto3_x2+cyto3_x2_s1+cyto3_x2_s2"]

# round-2 cross-validation tasks (config:modality), run with  CM_CV_TASKS="$ROUND2" bash submit_cv.sh
ROUND2 = ("cyto3_x3_long:ex cyto3_x4:ex cyto3_x3_aug:ex nuclei_x2:ex cyto3_x3_s1:ex cyto3_x3_s2:ex cyto3_x3_tta:ex "
          "cyto3_x2_s1:iv cyto3_x2_s2:iv cyto3_x2_tta:iv")

# configurations run in cross-validation (edit to add/remove)
CV_CONFIGS = ["base", "big", "cyto3_x2", "cyto3_x3", "cyto3_x3_lr1"]
MODS = ["iv", "ex"]
SUBJECTS = ["subject_5d294c", "subject_b2ba5e", "subject_db6b8b"]

# Cellpose cell-probability thresholds tried at evaluation time (mask extent)
CP_GRID = {"iv": [-1.5, -1.0, -0.5, 0.0], "ex": [-1.0, -0.5, 0.0, 0.5]}
# pair-classifier thresholds tried at evaluation time
THR_GRID = [0.05, 0.1, 0.2]
