"""Shared utilities: data loading, RLE coding, and the competition metric."""
import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

DATA = os.environ.get("CM_DATA", "data")


# ----------------------------------------------------------------------------- data
def region_dir(sample_id, split):
    subj, reg = sample_id.split("__")
    return os.path.join(DATA, split, subj, reg)


def load_images(sample_id, split="training"):
    d = region_dir(sample_id, split)
    iv = tifffile.imread(os.path.join(d, "invivo.tif"))
    ex = tifffile.imread(os.path.join(d, "exvivo.tif"))
    return iv, ex


def load_gt():
    return pd.read_csv(os.path.join(DATA, "training", "train_ground_truth.csv"))


# ----------------------------------------------------------------------------- RLE
# Ground-truth format: "start length start length ..." over the row-major (C-order)
# flattened image. RLE_OFFSET is the index base (verified against image content).
RLE_OFFSET = 0


def rle_decode(rle, shape):
    s = np.asarray(rle.split(), dtype=np.int64)
    starts, lengths = s[0::2] - RLE_OFFSET, s[1::2]
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for a, l in zip(starts, lengths):
        flat[a:a + l] = True
    return flat.reshape(shape)


def rle_encode(mask):
    flat = np.concatenate([[0], mask.ravel().astype(np.int8), [0]])
    d = np.diff(flat)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return " ".join(f"{a + RLE_OFFSET} {b - a}" for a, b in zip(starts, ends))


def instances_to_label(inst, shape):
    """dict id->rle  ->  (label image int32, list of ids in label order)."""
    lab = np.zeros(shape, dtype=np.int32)
    ids = list(inst.keys())
    for k, cid in enumerate(ids, 1):
        m = rle_decode(inst[cid], shape)
        lab[m & (lab == 0)] = k
    return lab, ids


def _runs_from_flat(idx):
    """sorted flat indices -> 'start len start len ...'"""
    if len(idx) == 0:
        return ""
    br = np.where(np.diff(idx) != 1)[0]
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return " ".join(f"{a + RLE_OFFSET} {b - a + 1}" for a, b in zip(starts, ends))


def label_to_instances(lab, prefix):
    """label image -> (dict id->rle, dict label->id). Uses bounding boxes for speed."""
    from scipy import ndimage as ndi
    out, lmap = {}, {}
    H, W = lab.shape
    objs = ndi.find_objects(lab)
    for l0, sl in enumerate(objs):
        if sl is None:
            continue
        l = l0 + 1
        rr, cc = np.nonzero(lab[sl] == l)
        if len(rr) == 0:
            continue
        idx = np.sort((rr + sl[0].start) * W + (cc + sl[1].start))
        cid = f"{prefix}_{int(l):06d}"
        out[cid] = _runs_from_flat(idx)
        lmap[int(l)] = cid
    return out, lmap


# ----------------------------------------------------------------------------- metric
def _masks_from_instances(inst, shape):
    """Return list of ids and list of flat index arrays."""
    ids, pix = [], []
    for cid, rle in inst.items():
        s = np.asarray(rle.split(), dtype=np.int64)
        if len(s) == 0:
            continue
        starts, lengths = s[0::2] - RLE_OFFSET, s[1::2]
        idx = np.concatenate([np.arange(a, a + l) for a, l in zip(starts, lengths)])
        ids.append(cid)
        pix.append(idx)
    return ids, pix


def match_masks(pred_inst, gt_inst, shape, thr=0.75):
    """Return (tp list of (pred_id, gt_id, iou), n_pred, n_gt)."""
    pids, ppix = _masks_from_instances(pred_inst, shape)
    gids, gpix = _masks_from_instances(gt_inst, shape)
    npx = shape[0] * shape[1]
    glab = np.zeros(npx, dtype=np.int32)
    gcount = np.zeros(len(gids) + 1, dtype=np.int64)
    for k, idx in enumerate(gpix, 1):
        glab[idx] = k
        gcount[k] = len(idx)
    cand = []  # (iou, pi, gi)
    for pi, idx in enumerate(ppix):
        idx = np.unique(idx)
        g = glab[idx]
        g = g[g > 0]
        if len(g) == 0:
            continue
        u, c = np.unique(g, return_counts=True)
        for gk, inter in zip(u, c):
            iou = inter / (len(idx) + gcount[gk] - inter)
            if iou > thr:
                cand.append((iou, pi, gk - 1))
    # IoU>0.5 => unique match; with 0.75 each pred/gt has at most one partner
    cand.sort(reverse=True)
    usedp, usedg, tp = set(), set(), []
    for iou, pi, gi in cand:
        if pi in usedp or gi in usedg:
            continue
        usedp.add(pi); usedg.add(gi)
        tp.append((pids[pi], gids[gi], float(iou)))
    return tp, len(pids), len(gids)


def pq_from(tp, n_pred, n_gt):
    ntp = len(tp)
    fp, fn = n_pred - ntp, n_gt - ntp
    denom = ntp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return 1.0
    return sum(t[2] for t in tp) / denom


def score_region(pred_row, gt_row, iv_shape, ex_shape):
    piv = json.loads(pred_row["invivo_instances"]) if isinstance(pred_row["invivo_instances"], str) else pred_row["invivo_instances"]
    pex = json.loads(pred_row["exvivo_instances"]) if isinstance(pred_row["exvivo_instances"], str) else pred_row["exvivo_instances"]
    ppairs = json.loads(pred_row["match_pairs"]) if isinstance(pred_row["match_pairs"], str) else pred_row["match_pairs"]
    giv = json.loads(gt_row["invivo_instances"])
    gex = json.loads(gt_row["exvivo_instances"])
    gpairs = json.loads(gt_row["match_pairs"])
    tp_iv, npi, ngi = match_masks(piv, giv, iv_shape)
    tp_ex, npe, nge = match_masks(pex, gex, ex_shape)
    map_iv = {p: g for p, g, _ in tp_iv}
    map_ex = {p: g for p, g, _ in tp_ex}
    gset = {(a, b) for a, b in gpairs}
    correct = 0
    for a, b in ppairs:
        if a in map_iv and b in map_ex and (map_iv[a], map_ex[b]) in gset:
            correct += 1
    return dict(pq_iv=pq_from(tp_iv, npi, ngi), pq_ex=pq_from(tp_ex, npe, nge),
                correct=correct, n_pred_pairs=len(ppairs), n_gt_pairs=len(gpairs),
                tp_iv=len(tp_iv), n_pred_iv=npi, n_gt_iv=ngi,
                tp_ex=len(tp_ex), n_pred_ex=npe, n_gt_ex=nge)


def score_submission(pred_df, gt_df, split="training", verbose=False):
    gt = gt_df.set_index("sample_id")
    rows = []
    for _, pr in pred_df.iterrows():
        sid = pr["sample_id"]
        if sid not in gt.index:
            continue
        iv, ex = load_images(sid, split)
        r = score_region(pr, gt.loc[sid], iv.shape, ex.shape)
        r["sample_id"] = sid
        rows.append(r)
        if verbose:
            print(sid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'sample_id'})
    res = pd.DataFrame(rows)
    pq_iv, pq_ex = res.pq_iv.mean(), res.pq_ex.mean()
    c, p, g = res.correct.sum(), res.n_pred_pairs.sum(), res.n_gt_pairs.sum()
    prec = c / p if p else 0.0
    rec = c / g if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    S = 0.5 * (pq_iv + pq_ex) / 2 + 0.5 * f1
    return dict(S=S, pq_iv=pq_iv, pq_ex=pq_ex, f1=f1, prec=prec, rec=rec,
                correct=int(c), pred_pairs=int(p), gt_pairs=int(g)), res
