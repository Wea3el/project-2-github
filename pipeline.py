"""End-to-end: segmentation labels -> registration -> pair selection -> submission rows."""
import json
import numpy as np
from collections import defaultdict
from cmutil import label_to_instances
from match import (cell_table, register_region, local_refine, pair_features, apply_aff)


def consensus_pick(regs, lam=1.0, sth=3.0, scen=120.0, groups=None):
    """regs: dict sid -> list of candidate dicts (z, th, cen). Picks one per region using
    agreement with the other regions of the same group (mouse + ex-vivo frame size: sections
    imaged in the same frame put the in-vivo field of view at similar places)."""
    by = defaultdict(list)
    for sid in regs:
        by[groups[sid] if groups else sid.split("__")[0]].append(sid)
    pick = {}
    for subj, sids in by.items():
        for sid in sids:
            best = None
            for ci, c in enumerate(regs[sid]):
                sup = 0.0
                for s2 in sids:
                    if s2 == sid:
                        continue
                    v = 0.0
                    for c2 in regs[s2]:
                        w = max(c2["z"], 0) * np.exp(-(c["th"] - c2["th"]) ** 2 / (2 * sth ** 2)
                                                     - np.sum((c["cen"] - c2["cen"]) ** 2) / (2 * scen ** 2))
                        v = max(v, w)
                    sup += v
                sup /= max(len(sids) - 1, 1)
                sc = c["z"] + lam * sup
                if best is None or sc > best[0]:
                    best = (sc, ci)
            pick[sid] = best[1] if best else None
    return pick


def subject_modes(regs, pick, zmin=6.0):
    """Robust per-subject (theta, centre) from confidently registered regions."""
    by = defaultdict(list)
    for sid, ci in pick.items():
        if ci is None:
            continue
        c = regs[sid][ci]
        by[sid.split("__")[0]].append((c["z"], c["th"], c["cen"]))
    modes = {}
    for subj, v in by.items():
        good = [x for x in v if x[0] >= zmin]
        if len(good) < 2:
            continue
        th = np.median([x[1] for x in good]); cen = np.median(np.array([x[2] for x in good]), axis=0)
        modes[subj] = (float(th), cen, len(good))
    return modes


def second_pass(regs, pick, tables, zweak=7.0, dth=6.0, rad=250.0):
    """Re-search weakly registered regions inside the subject's consensus window."""
    from match import register_region
    modes = subject_modes(regs, pick)
    changed = []
    for sid, ci in pick.items():
        subj = sid.split("__")[0]
        if subj not in modes:
            continue
        if ci is not None and regs[sid][ci]["z"] >= zweak:
            continue
        th, cen, _ = modes[subj]
        Tiv, Tex, ex_shape = tables[sid]
        extra = register_region(Tiv["xy"], Tex["xy"], ex_shape, topk=6, angles=np.arange(th - dth, th + dth + 0.1, 1.0),
                                window=(cen, rad), per_hyp=4)
        regs[sid] = regs[sid] + extra
        changed.append(sid)
    return changed


def region_candidates(liv, lex, iv, ex):
    Tiv = cell_table(liv, iv)
    Tex = cell_table(lex, ex)
    res = register_region(Tiv["xy"], Tex["xy"], ex.shape) if len(Tiv["xy"]) >= 3 and len(Tex["xy"]) >= 3 else []
    return Tiv, Tex, res


def region_pairs(Tiv, Tex, reg, iv_shape, r=10.0):
    civ, cex = Tiv["xy"], Tex["xy"]
    P = local_refine(civ, cex, reg["M"])
    a, b, F = pair_features(civ, cex, P, Tiv, Tex, reg, r=r, iv_shape=iv_shape)
    return a, b, F


def build_row(sid, liv, lex, Tiv, Tex, a, b, keep, inst=None):
    if inst is None:
        inst = (label_to_instances(liv, "IV"), label_to_instances(lex, "EX"))
    (iv_inst, iv_map), (ex_inst, ex_map) = inst
    pairs, used_i, used_e = [], set(), set()
    for x, y, k in zip(a, b, keep):
        if not k:
            continue
        li, le = int(Tiv["labels"][x]), int(Tex["labels"][y])
        ci, ce = iv_map.get(li), ex_map.get(le)
        if ci is None or ce is None or ci in used_i or ce in used_e:
            continue
        used_i.add(ci); used_e.add(ce)
        pairs.append([ci, ce])
    return dict(sample_id=sid, invivo_instances=iv_inst, exvivo_instances=ex_inst,
                match_pairs=pairs)


def row_to_csv(row):
    return dict(sample_id=row["sample_id"], invivo_instances=json.dumps(row["invivo_instances"]),
                exvivo_instances=json.dumps(row["exvivo_instances"]), match_pairs=json.dumps(row["match_pairs"]))
