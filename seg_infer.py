"""Cellpose inference helpers: Cellpose 3 U-Nets (from scratch or fine-tuned cyto3) and fine-tuned Cellpose-SAM."""
import os
import numpy as np
import torch
import cv2
from cellpose import models
try:
    from cellpose.resnet_torch import CPnet
except ImportError:  # cellpose 4 (the Cellpose-SAM overlay) has no U-Net; only load_sam is used there
    CPnet = torch.nn.Module
from segdata import norm_img
import torch.nn.functional as F


class CPnetPad(CPnet):
    """Cellpose's inference code assumes a 256-d style vector; pad smaller nets."""
    def forward(self, data):
        out = super().forward(data)
        y, style = out[0], out[1]
        if style.shape[1] < 256:
            style = F.pad(style, (0, 256 - style.shape[1]))
        return (y, style) + tuple(out[2:])


def infer_nbase(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    nb, i = [], 0
    while f"downsample.down.res_down_{i}.proj.1.weight" in sd:
        nb.append(int(sd[f"downsample.down.res_down_{i}.proj.1.weight"].shape[0])); i += 1
    return tuple(nb)


def load_cp(path, nbase=None, up=1.0):
    nbase = nbase or infer_nbase(path)
    use_gpu = torch.cuda.is_available() and os.environ.get("CM_CPU", "0") != "1"
    m = models.CellposeModel(gpu=use_gpu, pretrained_model=False, model_type=None,
                             diam_mean=float(up * 10.0), nchan=2)
    m.net = CPnetPad([2, *nbase], 3, sz=3, mkldnn=m.mkldnn, max_pool=True, diam_mean=float(up * 10.0)).to(m.device)
    m.net.load_model(path, device=m.device)
    m.net.eval()
    m.pretrained_model = path
    return m


def load_sam(path):
    """a fine-tuned Cellpose-SAM model (cellpose 4)"""
    return models.CellposeModel(gpu=torch.cuda.is_available() and os.environ.get("CM_CPU", "0") != "1",
                                pretrained_model=path)


def segment(model, img, up=1.0, cellprob=0.0, flow=0.4, min_size=15, tile_overlap=0.1, augment=False, return_flows=False,
            resample=True):
    x = norm_img(img)
    kw = dict(normalize=False, cellprob_threshold=cellprob, flow_threshold=flow, min_size=min_size,
              tile_overlap=tile_overlap, augment=augment, resample=resample)
    if getattr(model, "backbone", "") == "sam_vitl":
        # cellpose 4 ignores `rescale`: it upsamples by 30/diameter; flows come back at the native size, as below
        masks, flows, _ = model.eval(x, diameter=30.0 / up, bsize=256, batch_size=32, **kw)
    else:
        masks, flows, _ = model.eval(x, channels=[0, 0], diameter=None, rescale=up, bsize=224, **kw)
    masks = masks.astype(np.int32)
    if return_flows:
        return masks, flows
    return masks


TRAIN_DIAM = {"iv": 10.6, "ex": 10.3}  # median training diameters (px)


def median_diam_lab(lab):
    a = np.bincount(lab.ravel())[1:]
    a = a[a > 0]
    return float(np.median(2 * np.sqrt(a / np.pi))) if len(a) >= 10 else None


def segment_adaptive(model, img, mod, tol=0.12, **kw):
    """Segment, estimate the median cell diameter, and re-segment at a rescale factor that
    brings cells to the training size when the estimate is off by more than `tol`."""
    lab = segment(model, img, up=1.0, **kw)
    d = median_diam_lab(lab)
    if d is None:
        return lab, 1.0
    r = TRAIN_DIAM[mod] / d
    if abs(r - 1) <= tol:
        return lab, 1.0
    r = float(np.clip(r, 0.7, 1.5))
    return segment(model, img, up=r, **kw), r
