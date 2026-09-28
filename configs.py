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
    # the shipped weights behind submissions v3/v4 (trained on all three mice): only for
    # predictions (runs/shipped/<mod>/full) or sanity checks, never for cross-validation
    "shipped": dict(pretrained=None, nbase=(16, 32, 64, 128), up=1),
}

# configurations run in cross-validation (edit to add/remove)
CV_CONFIGS = ["base", "big", "cyto3_x2", "cyto3_x3", "cyto3_x3_lr1"]
MODS = ["iv", "ex"]
SUBJECTS = ["subject_5d294c", "subject_b2ba5e", "subject_db6b8b"]

# Cellpose cell-probability thresholds tried at evaluation time (mask extent)
CP_GRID = {"iv": [-1.5, -1.0, -0.5, 0.0], "ex": [-1.0, -0.5, 0.0, 0.5]}
# pair-classifier thresholds tried at evaluation time
THR_GRID = [0.05, 0.1, 0.2]
