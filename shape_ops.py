import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image


def pq(pl, gl, thr=0.75):
    m = (pl > 0) & (gl > 0)
    pa = np.bincount(pl.ravel()); ga = np.bincount(gl.ravel())
    npred = int((pa[1:] > 0).sum()); ngt = int((ga[1:] > 0).sum())
    if m.sum() == 0:
        return (0.0 if ngt or npred else 1.0), 0
    pairs, cnt = np.unique(np.stack([pl[m], gl[m]]), axis=1, return_counts=True)
    iou = cnt / (pa[pairs[0]] + ga[pairs[1]] - cnt)
    ok = iou > thr
    tp = int(ok.sum())
    return iou[ok].sum() / (tp + 0.5 * (npred - tp) + 0.5 * (ngt - tp)), tp


def per_cell(lab, op, pad=4):
    H, W = lab.shape; out = np.zeros_like(lab)
    for k0, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        k = k0 + 1
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        L = lab[y0:y1, x0:x1]; m = L == k
        if m.sum() < 3:
            continue
        new = op(m) & ((L == 0) | (L == k))
        t = out[y0:y1, x0:x1]; t[new & (t == 0)] = k
    return out


cross = ndi.generate_binary_structure(2, 1)
OPS = {
    "orig": lambda m: m,
    "hull": lambda m: convex_hull_image(m),
    "dil_cross": lambda m: ndi.binary_dilation(m, cross),
    "hull_dil": lambda m: ndi.binary_dilation(convex_hull_image(m), cross),
    "open_dil": lambda m: ndi.binary_dilation(ndi.binary_opening(m, cross), cross) | m,
}
