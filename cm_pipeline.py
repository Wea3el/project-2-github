"""Masks -> registration (per-region Hough + mouse-level consensus) -> candidate pairs ->
pair classifier -> submission rows; plus a fast implementation of the competition metric."""
import json
import numpy as np, pandas as pd

from cmutil import label_to_instances
from match import cell_table, register_region
from pipeline import consensus_pick, region_pairs
from consensus2 import group_register

# Pipeline variants, written "<pair data>[+option=value...]", e.g. "oof+weak=keep+bright=0.5":
#   weak     drop | keep   regions whose registration is not confident (default: drop them)
#   zwin     4.0           confidence needed for a registration that agrees with the mouse's consensus
#   zalone   8.0           confidence needed for a registration that does not
#   bright   1.0           fraction of the brightest in-vivo cells used to find the registration
#   grow_iv  0 / grow_ex 0 grow every predicted mask by this many pixels (without merging cells)
#   flow_iv / flow_ex 0.4, min_iv / min_ex 15   Cellpose flow-error threshold and minimum cell size (px)
#   cand     hung | gain | all   how candidate pairs are chosen (match.candidate_pairs); all = every nearby
#            pair is scored by the classifier, then a one-to-one assignment maximises the summed (p - thr)
#   u        10.0          gain: the cost of leaving a cell unmatched (a pair at distance d is worth u - d)
METHOD_DEFAULTS = dict(weak="drop", zwin=4.0, zalone=8.0, bright=1.0, grow_iv=0, grow_ex=0,
                       flow_iv=0.4, flow_ex=0.4, min_iv=15, min_ex=15, cand="hung", u=10.0)


def parse_method(spec):
    """'oof+weak=keep+grow_ex=1' -> ('oof', {options}); unknown options raise KeyError."""
    pairs, *kv = spec.split("+")
    o = dict(METHOD_DEFAULTS)
    for x in kv:
        k, v = x.split("=")
        o[k] = type(METHOD_DEFAULTS[k])(v)
    return pairs, o


def match_kw(opt):
    """the match_regions() arguments of a parsed pipeline variant"""
    return {k: opt[k] for k in ("weak", "zwin", "zalone", "bright", "cand", "u")}


def mod_masks(dP, cellprob, cp, up, opt, mod):
    """masks of one modality from cached network outputs, with the variant's mask options"""
    from common import masks_from_flows
    lab = masks_from_flows(dP, cellprob, cp, up, flow_threshold=opt[f"flow_{mod}"], min_size=opt[f"min_{mod}"])
    return grow_masks(lab, opt[f"grow_{mod}"])


def assign_by_score(la, lb, p, t):
    """one-to-one pairs maximising the summed (p - t) over the accepted pairs; an unmatched cell scores 0"""
    from scipy.optimize import linear_sum_assignment
    keep = p >= t
    if not keep.any():
        return []
    la, lb, w = np.asarray(la)[keep], np.asarray(lb)[keep], p[keep] - t + 1e-9
    ua, ia = np.unique(la, return_inverse=True)
    ub, ib = np.unique(lb, return_inverse=True)
    W = np.zeros((len(ua), len(ub)))
    np.maximum.at(W, (ia, ib), w)
    r, c = linear_sum_assignment(W, maximize=True)
    return [(int(ua[i]), int(ub[j])) for i, j in zip(r, c) if W[i, j] > 0]


def grow_masks(lab, px):
    if not px:
        return lab
    from skimage.segmentation import expand_labels
    return expand_labels(lab, int(px)).astype(lab.dtype)


def _reg_points(T, frac):
    """in-vivo points used for registration: the brightest `frac` of the cells (all cells if frac >= 1)"""
    xy = T["xy"]
    if frac >= 1 or len(xy) < 20 or "inten" not in T:
        return xy
    k = max(20, int(round(len(xy) * frac)))
    return xy[np.argsort(-T["inten"])[:k]]


def match_regions(items, clf_for, thrs=(0.1,), weak="drop", verbose=False, feats=None, zwin=4.0, zalone=8.0, bright=1.0,
                  cand="hung", u=10.0):
    """items: list of dict(sid, iv_img, ex_img, liv, lex). clf_for(sid) -> fitted classifier.
    Returns {thr: {sid: [(iv_label, ex_label), ...]}}, log DataFrame.
    feats (dict, optional) receives {sid: (iv labels, ex labels, features)} of every candidate pair."""
    info, data, cands, shapes = {}, {}, {}, {}
    for it in items:
        sid = it["sid"]
        Tiv, Tex = cell_table(it["liv"], it["iv_img"]), cell_table(it["lex"], it["ex_img"])
        civ_r = _reg_points(Tiv, bright)
        res = register_region(civ_r, Tex["xy"], it["ex_img"].shape) if len(civ_r) >= 3 and len(Tex["xy"]) >= 3 else []
        info[sid] = (Tiv, Tex, it["iv_img"].shape)
        data[sid] = (civ_r, Tex["xy"], it["ex_img"].shape); cands[sid] = res; shapes[sid] = it["ex_img"].shape
    groups = {s: s.split("__")[0] + str(shapes[s]) for s in cands}
    pick = consensus_pick(cands, groups=groups)
    new = group_register(data, cands, groups, pick, z_alone=zalone, z_win=zwin, verbose=verbose)
    out = {t: {} for t in thrs}
    log = []
    for sid in cands:
        Tiv, Tex, ivs = info[sid]
        n = new[sid]; reg = n["reg"]
        a = b = np.zeros(0, int); p = np.zeros(0)
        if reg is not None and not (n["status"] == "weak" and weak == "drop"):
            reg = dict(reg)
            others = [c["z"] for c in cands[sid] if np.linalg.norm(c["cen"] - reg["cen"]) > 40]
            reg["zgap"] = reg["z"] - (max(others) if others else 0)
            a, b, F = region_pairs(Tiv, Tex, reg, ivs, cand=cand, u=u)
            if feats is not None:
                feats[sid] = (Tiv["labels"][a], Tex["labels"][b], F)
            p = clf_for(sid).predict_proba(F)[:, 1] if len(a) else np.zeros(0)
        for t in thrs:
            if cand == "all":  # candidates overlap: choose the best one-to-one set by score
                out[t][sid] = assign_by_score(Tiv["labels"][a], Tex["labels"][b], p, t)
                continue
            pairs, ui, ue = [], set(), set()
            for x, y, pp in zip(a, b, p):
                if pp < t:
                    continue
                li, le = int(Tiv["labels"][x]), int(Tex["labels"][y])
                if li in ui or le in ue:
                    continue
                ui.add(li); ue.add(le); pairs.append((li, le))
            out[t][sid] = pairs
        log.append(dict(sid=sid, status=n["status"], z=reg["z"] if reg is not None else np.nan,
                        th=reg["th"] if reg is not None else np.nan, n_iv=len(Tiv["xy"]), n_ex=len(Tex["xy"]),
                        n_pairs=len(out[thrs[0]][sid])))
    return out, pd.DataFrame(log)


def to_rows(sid, liv, lex, pairs):
    (iv_inst, iv_map), (ex_inst, ex_map) = label_to_instances(liv, "IV"), label_to_instances(lex, "EX")
    mp = [[iv_map[a], ex_map[b]] for a, b in pairs if a in iv_map and b in ex_map]
    return dict(sample_id=sid, invivo_instances=json.dumps(iv_inst), exvivo_instances=json.dumps(ex_inst),
                match_pairs=json.dumps(mp))


# ------------------------------------------------------------------ metric (same rules as the competition)
def tp_map(pl, gl, thr=0.75):
    """pred label -> (gt label, IoU) for IoU > thr (unique because thr > 0.5); returns map, PQ."""
    pa = np.bincount(pl.ravel()); ga = np.bincount(gl.ravel())
    npred = int((pa[1:] > 0).sum()); ngt = int((ga[1:] > 0).sum())
    m = (pl > 0) & (gl > 0)
    mp = {}
    if m.any():
        pr, cnt = np.unique(np.stack([pl[m], gl[m]]), axis=1, return_counts=True)
        iou = cnt / (pa[pr[0]] + ga[pr[1]] - cnt)
        for (p, g), v in zip(pr.T, iou):
            if v > thr:
                mp[int(p)] = (int(g), float(v))
    tp = len(mp)
    den = tp + 0.5 * (npred - tp) + 0.5 * (ngt - tp)
    pq = sum(v for _, v in mp.values()) / den if den else 1.0
    return mp, pq


def score(pred, gts):
    """pred: {sid: (liv, lex, pairs)}, gts: {sid: (giv, gex, gpairs)} (label images, label pairs)."""
    rows = []
    for sid, (liv, lex, pairs) in pred.items():
        giv, gex, gp = gts[sid]
        miv, pqi = tp_map(liv, giv); mex, pqe = tp_map(lex, gex)
        gset = {(int(a), int(b)) for a, b in gp}
        correct = sum(1 for a, b in pairs if a in miv and b in mex and (miv[a][0], mex[b][0]) in gset)
        rows.append(dict(sid=sid, pq_iv=pqi, pq_ex=pqe, correct=correct, n_pred=len(pairs), n_gt=len(gset)))
    r = pd.DataFrame(rows)
    c, p, g = r.correct.sum(), r.n_pred.sum(), r.n_gt.sum()
    prec = c / p if p else 0.0; rec = c / g if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    S = 0.25 * (r.pq_iv.mean() + r.pq_ex.mean()) + 0.5 * f1
    return dict(S=S, pq_iv=r.pq_iv.mean(), pq_ex=r.pq_ex.mean(), f1=f1, prec=prec, rec=rec,
                correct=int(c), pred_pairs=int(p), gt_pairs=int(g)), r
