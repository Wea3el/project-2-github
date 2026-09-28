"""Mouse-level consensus registration.

Regions of the same mouse (and same ex-vivo frame size) put the in-vivo field of view at
nearly the same place/rotation in the ex-vivo frame. The first pass registers every region
independently; here we (1) find the dominant (theta, centre) modes of each group from the
first-pass picks, (2) re-search every region inside a window around each mode, and
(3) pick per region, flagging registrations that are neither confident on their own nor
consistent with a mode.
"""
import numpy as np
from collections import defaultdict
from match import register_region


def _k(a, b, sth, scen):
    return np.exp(-(a["th"] - b["th"]) ** 2 / (2 * sth ** 2) - np.sum((a["cen"] - b["cen"]) ** 2) / (2 * scen ** 2))


def find_modes(picks, zfloor=4.0, sth=4.0, scen=120.0, max_modes=2, min_support=1.0, dth=6.0, dcen=200.0):
    """picks: list of candidate dicts (one per region, may contain None).
    Returns list of modes dict(th, cen, n, support)."""
    P = [p for p in picks if p is not None]
    if not P:
        return []
    w = np.array([max(p["z"] - zfloor, 0.0) for p in P])
    used = np.zeros(len(P), bool)
    modes = []
    for _ in range(max_modes):
        best, bi = -1, -1
        for i, p in enumerate(P):
            if used[i] or w[i] <= 0:
                continue
            sup = sum(w[j] * _k(p, P[j], sth, scen) for j in range(len(P)) if j != i and not used[j])
            sc = w[i] + sup
            if sup >= min_support and sc > best:
                best, bi = sc, i
        if bi < 0:
            break
        # members: picks close to the seed
        mem = [j for j in range(len(P)) if not used[j]
               and abs(P[j]["th"] - P[bi]["th"]) <= dth and np.linalg.norm(P[j]["cen"] - P[bi]["cen"]) <= dcen]
        ww = np.array([max(w[j], 0.25) for j in mem])
        th = float(np.average([P[j]["th"] for j in mem], weights=ww))
        cen = np.average(np.array([P[j]["cen"] for j in mem]), axis=0, weights=ww)
        modes.append(dict(th=th, cen=cen, n=len(mem), support=float(best)))
        used[mem] = True
    return modes


def windowed_candidates(civ, cex, ex_shape, mode, dth=7.0, rad=220.0, step=1.0, topk=6):
    angles = np.arange(mode["th"] - dth, mode["th"] + dth + 1e-6, step)
    out = register_region(civ, cex, ex_shape, topk=topk, angles=angles, window=(mode["cen"], rad), per_hyp=4)
    for c in out:
        c["win"] = True
    return out


def agrees(c, mode, dth=6.0, dcen=200.0):
    return abs(c["th"] - mode["th"]) <= dth and np.linalg.norm(c["cen"] - mode["cen"]) <= dcen


def group_register(data, cands, groups, first_pick, z_alone=8.0, z_win=4.0, verbose=False):
    """data: sid -> (civ, cex, ex_shape); cands: sid -> first-pass candidate list;
    first_pick: sid -> index into cands. Returns sid -> dict(reg, status, modes)."""
    by = defaultdict(list)
    for sid in cands:
        by[groups[sid]].append(sid)
    out = {}
    for g, sids in by.items():
        picks = [cands[s][first_pick[s]] if cands[s] and first_pick[s] is not None else None for s in sids]
        # identical regions (same in-vivo and ex-vivo cells) must not vote twice
        seen, mp = set(), []
        for s, p in zip(sids, picks):
            key = (np.round(data[s][0], 1).tobytes(), np.round(data[s][1], 1).tobytes())
            mp.append(None if key in seen else p); seen.add(key)
        modes = find_modes(mp)
        if verbose:
            print(g, "modes:", [(round(m["th"], 1), m["cen"].round(0).tolist(), m["n"], round(m["support"], 1)) for m in modes])
        for s, p in zip(sids, picks):
            civ, cex, exs = data[s]
            best_win = None
            for mi, m in enumerate(modes):
                if len(civ) < 3 or len(cex) < 3:
                    continue
                wc = windowed_candidates(civ, cex, exs, m)
                # first-pass candidates that already agree with this mode count too
                wc += [c for c in cands[s] if agrees(c, m)]
                for c in wc:
                    if best_win is None or c["z"] > best_win[0]["z"]:
                        best_win = (c, mi)
            glob = p
            if best_win is not None and best_win[0]["z"] >= z_win and (glob is None or best_win[0]["z"] >= glob["z"] - 2.0 or glob["z"] < z_alone):
                reg, status = best_win[0], f"mode{best_win[1]}"
            elif glob is not None and glob["z"] >= z_alone:
                reg, status = glob, "alone"
            else:
                reg, status = (best_win[0] if best_win else glob), "weak"
            out[s] = dict(reg=reg, status=status, modes=modes, glob=glob, win=best_win[0] if best_win else None)
    return out
