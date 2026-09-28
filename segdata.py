"""Image normalisation and training-tile extraction for Cellpose training."""
import numpy as np
from scipy import ndimage as ndi


def norm_img(img, lo=1.0, hi=99.8):
    """Percentile-normalise a 16-bit image to ~[0,1] using non-zero (tissue) pixels."""
    a = img.astype(np.float32)
    v = a[a > 0]
    if v.size < 100:
        v = a.ravel()
    # subsample for speed
    if v.size > 2_000_000:
        v = v[:: v.size // 2_000_000]
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / max(p2 - p1, 1e-6), -0.5, 2.0).astype(np.float32)


def median_diameter(lab):
    areas = np.bincount(lab.ravel())[1:]
    areas = areas[areas > 0]
    return float(np.median(2 * np.sqrt(areas / np.pi))) if len(areas) else 10.0


def make_tiles(img, lab, tile=192, step=160, keep_empty=0.15, rng=None):
    """Cut an image into overlapping tiles; keep all with cells and some empty ones.
    Labels are relabelled consecutively inside each tile."""
    rng = rng or np.random.default_rng(0)
    H, W = img.shape
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    out_i, out_l = [], []
    for y in ys:
        for x in xs:
            li = lab[y:y + tile, x:x + tile]
            ii = img[y:y + tile, x:x + tile]
            if ii.shape != (tile, tile):
                pi = np.zeros((tile, tile), np.float32); pl = np.zeros((tile, tile), np.int32)
                pi[:ii.shape[0], :ii.shape[1]] = ii; pl[:li.shape[0], :li.shape[1]] = li
                ii, li = pi, pl
            if (ii > 0.02).mean() < 0.05:  # mostly outside tissue
                if rng.random() > keep_empty * 0.3:
                    continue
            if li.max() == 0 and rng.random() > keep_empty:
                continue
            u = np.unique(np.r_[0, li.ravel()])
            out_i.append(ii.copy())
            out_l.append(np.searchsorted(u, li).astype(np.int32))
    return out_i, out_l
