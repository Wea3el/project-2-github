"""Cross-modal cell registration and matching.

Model: ex-vivo coords ~= A(in-vivo coords), A close to a similarity transform
(small rotation, scale ~0.95, no reflection) plus mild non-rigid deformation.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- features
def cell_table(lab, img=None):
    """Centroids (x,y), areas and mean normalised intensity for each label 1..n."""
    n = int(lab.max())
    idx = np.arange(1, n + 1)
    if n == 0:
        return dict(xy=np.zeros((0, 2)), area=np.zeros(0), inten=np.zeros(0), labels=idx)
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx)).T
    area = ndi.sum(np.ones_like(lab), lab, idx)
    keep = area > 0
    out = dict(xy=np.c_[cx, cy][keep], area=area[keep], labels=idx[keep])
    if img is not None:
        a = img.astype(np.float32)
        mu = ndi.mean(a, lab, idx)[keep]
        v = a[a > 0]
        p5, p99 = np.percentile(v, [5, 99.5]) if v.size else (0, 1)
        out["inten"] = (mu - p5) / max(p99 - p5, 1e-6)
    return out


def sim_matrix(theta_deg, s):
    t = np.deg2rad(theta_deg)
    return s * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def apply_aff(M, xy):
    """M is 2x3."""
    return xy @ M[:, :2].T + M[:, 2]


# ----------------------------------------------------------------------------- global search
def hough_register(civ, cex, angles=np.arange(-24, 24.1, 1.5), scales=np.arange(0.86, 1.08, 0.02),
                   bin_px=4.0, sigma_bins=1.0, wiv=None, wex=None, topk=5):
    """Vote for the translation of every (angle, scale) hypothesis.
    Returns list of candidates sorted by score: dict(theta, s, t, score)."""
    if len(civ) < 3 or len(cex) < 3:
        return []
    wiv = np.ones(len(civ)) if wiv is None else wiv
    wex = np.ones(len(cex)) if wex is None else wex
    W = (wiv[:, None] * wex[None, :]).ravel()
    lo = cex.min(0) - np.abs(civ).max() * 1.2 - 10
    hi = cex.max(0) + np.abs(civ).max() * 0.2 + 10
    nb = np.ceil((hi - lo) / bin_px).astype(int) + 1
    cands = []
    for th in angles:
        for s in scales:
            P = civ @ sim_matrix(th, s).T
            D = (cex[None, :, :] - P[:, None, :]).reshape(-1, 2)
            b = np.floor((D - lo) / bin_px).astype(int)
            ok = (b[:, 0] >= 0) & (b[:, 0] < nb[0]) & (b[:, 1] >= 0) & (b[:, 1] < nb[1])
            H = np.bincount(b[ok, 1] * nb[0] + b[ok, 0], weights=W[ok], minlength=nb[0] * nb[1]).reshape(nb[1], nb[0])
            Hs = ndi.gaussian_filter(H, sigma_bins, mode="constant") * (2 * np.pi * sigma_bins ** 2)
            k = np.argmax(Hs)
            yy, xx = divmod(k, nb[0])
            t = lo + (np.array([xx, yy]) + 0.5) * bin_px
            cands.append(dict(theta=float(th), s=float(s), t=t, score=float(Hs[yy, xx])))
    cands.sort(key=lambda c: -c["score"])
    # non-maximum suppression in translation space
    out = []
    for c in cands:
        if all(np.linalg.norm(c["t"] - o["t"]) > 30 or abs(c["theta"] - o["theta"]) > 6 for o in out):
            out.append(c)
        if len(out) >= topk:
            break
    return out


def cand_to_M(c):
    R = c["L"] if "L" in c else sim_matrix(c["theta"], c["s"])
    return np.c_[R, c["t"]]


# ----------------------------------------------------------------------------- refinement
def mutual_nn(A, B, r):
    """Mutual nearest neighbours between point sets A->B within radius r."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    ta, tb = cKDTree(A), cKDTree(B)
    dab, iab = tb.query(A)
    dba, iba = ta.query(B)
    i = np.arange(len(A))
    ok = (iba[iab] == i) & (dab < r)
    return i[ok], iab[ok], dab[ok]


def fit_affine(A, B, w=None):
    X = np.c_[A, np.ones(len(A))]
    if w is not None:
        X = X * w[:, None]; B = B * w[:, None]
    M, *_ = np.linalg.lstsq(X, B, rcond=None)
    return M.T  # 2x3


def fit_similarity(A, B):
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max((A0 ** 2).sum(), 1e-9)
    t = mb - s * R @ ma
    return np.c_[s * R, t]


def refine(civ, cex, M, radii=(12, 9, 7, 6, 6), min_pts=6):
    """ICP-style refinement: similarity first, then affine once enough inliers."""
    for k, r in enumerate(radii):
        P = apply_aff(M, civ)
        i, j, d = mutual_nn(P, cex, r)
        if len(i) < 3:
            break
        if len(i) >= min_pts and k >= 1:
            # robust affine: drop worst 10% residuals
            M1 = fit_affine(civ[i], cex[j])
            res = np.linalg.norm(apply_aff(M1, civ[i]) - cex[j], axis=1)
            keep = res <= np.percentile(res, 90) + 1e-6
            M = fit_affine(civ[i][keep], cex[j][keep]) if keep.sum() >= min_pts else M1
        else:
            M = fit_similarity(civ[i], cex[j])
    P = apply_aff(M, civ)
    i, j, d = mutual_nn(P, cex, radii[-1])
    return M, i, j, d


def local_refine(civ, cex, M, r=6.0, k=12, radius=120.0, iters=2):
    """Non-rigid correction: per in-vivo cell, shift by the robust mean residual of
    nearby inlier pairs (a simple smooth displacement field)."""
    P = apply_aff(M, civ)
    for _ in range(iters):
        i, j, d = mutual_nn(P, cex, r)
        if len(i) < 8:
            return P
        res = cex[j] - P[i]
        tree = cKDTree(P[i])
        newP = P.copy()
        for q in range(len(P)):
            dd, nn = tree.query(P[q], k=min(k, len(i)), distance_upper_bound=radius)
            nn = nn[np.isfinite(dd)]
            dd = dd[np.isfinite(dd)]
            if len(nn) < 3:
                continue
            w = np.exp(-(dd / (radius / 2)) ** 2)
            newP[q] = P[q] + (w[:, None] * res[nn]).sum(0) / w.sum()
        P = newP
    return P


def register(civ, cex, wiv=None, wex=None, **kw):
    cands = hough_register(civ, cex, wiv=wiv, wex=wex, **kw)
    best = None
    for c in cands:
        M, i, j, d = refine(civ, cex, cand_to_M(c))
        sc = (np.exp(-(d / 4.0) ** 2)).sum()
        if best is None or sc > best["score"]:
            best = dict(M=M, score=float(sc), n_inl=len(i), cand=c, hough=c["score"])
    if best is not None and len(cands) > 1:
        best["hough_ratio"] = cands[0]["score"] / max(cands[1]["score"], 1e-6)
    return best


# ----------------------------------------------------------------------------- assignment
def candidate_pairs(P, cex, r=10.0):
    """Hungarian assignment of transformed in-vivo points P to ex-vivo points with a
    gate r; returns (i, j, dist, d2_iv, d2_ex) with second-nearest distances."""
    if len(P) == 0 or len(cex) == 0:
        return [np.zeros(0)] * 5
    D = np.linalg.norm(P[:, None, :] - cex[None, :, :], axis=2)
    C = np.where(D < r, D, 1e4)
    a, b = linear_sum_assignment(C)
    ok = C[a, b] < 1e4
    a, b = a[ok], b[ok]
    # second nearest for uniqueness
    Ds = np.sort(D, axis=1)
    d2_iv = Ds[a, 1] if D.shape[1] > 1 else np.full(len(a), 99.0)
    DsT = np.sort(D, axis=0)
    d2_ex = DsT[1, b] if D.shape[0] > 1 else np.full(len(b), 99.0)
    return a, b, D[a, b], d2_iv, d2_ex


# ----------------------------------------------------------------------------- z-score Hough (FFT)
from scipy.signal import fftconvolve


def _points_img(xy, shape, b, off):
    q = np.floor((xy - off) / b).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < shape[0])
    im = np.zeros(shape, np.float32)
    np.add.at(im, (q[ok, 1], q[ok, 0]), 1.0)
    return im


def stretch_matrix(ratio, dir_deg):
    """Area-preserving stretch by sqrt(ratio) along dir and 1/sqrt(ratio) across."""
    t = np.deg2rad(dir_deg)
    u = np.array([np.cos(t), np.sin(t)]); v = np.array([-np.sin(t), np.cos(t)])
    r = np.sqrt(ratio)
    return r * np.outer(u, u) + (1 / r) * np.outer(v, v)


def hough_z(civ, cex, ex_shape, angles=np.arange(-24, 24.1, 2.0), scales=np.arange(0.86, 1.07, 0.03),
            b=3.0, sk_px=4.5, sr_px=60.0, topk=8, wiv=None, per_hyp=3, stretches=((1.0, 0.0),), window=None):
    """Translation voting for every (angle, scale); returns candidates ranked by
    z = (H - E) / sqrt(E + 1), where E is the vote count expected by chance given the
    local ex-vivo cell density."""
    from scipy import fft as sfft
    if len(civ) < 3 or len(cex) < 3:
        return []
    Hex, Wex = ex_shape
    gshape = (int(np.ceil(Hex / b)) + 1, int(np.ceil(Wex / b)) + 1)
    X = _points_img(cex, gshape, b, np.zeros(2))
    sk, sr = sk_px / b, sr_px / b
    A = ndi.gaussian_filter(X, sk, mode="constant") * (2 * np.pi * sk ** 2)
    R = ndi.gaussian_filter(X, sr, mode="constant") * (2 * np.pi * sk ** 2)
    ext = np.abs(civ - civ.mean(0)).max() * 2 * 1.15 / b + 4
    bmax = int(np.ceil(ext)) + 2
    L = (sfft.next_fast_len(gshape[0] + bmax), sfft.next_fast_len(gshape[1] + bmax))
    FA = sfft.rfft2(A, L); FR = sfft.rfft2(R, L)
    allc = []
    nms = max(int(40 / b), 3)
    hyps = [(th, s, st) for th in angles for s in scales for st in stretches]
    for th, s, st in hyps:
            Lm = sim_matrix(th, s) @ stretch_matrix(*st)
            P = civ @ Lm.T
            pmin = P.min(0) - b
            bshape = (int(np.ceil((P[:, 1].max() - pmin[1]) / b)) + 2, int(np.ceil((P[:, 0].max() - pmin[0]) / b)) + 2)
            B = np.zeros(bshape, np.float32)
            q = np.floor((P - pmin) / b).astype(int)
            np.add.at(B, (q[:, 1], q[:, 0]), 1.0 if wiv is None else wiv)
            FB = sfft.rfft2(B[::-1, ::-1], L)
            H = sfft.irfft2(FA * FB, L)
            E = sfft.irfft2(FR * FB, L)
            Z = (H - E) / np.sqrt(np.maximum(E, 0) + 1.0)
            if window is not None:
                # restrict the mapped FOV centre to lie within `radius` of `cen`
                wc, wr = window
                ky = np.arange(Z.shape[0]); kx = np.arange(Z.shape[1])
                ty = (ky - (bshape[0] - 1)) * b - pmin[1]
                tx = (kx - (bshape[1] - 1)) * b - pmin[0]
                cP = P.mean(0)
                dy = (cP[1] + ty - wc[1]) ** 2; dx = (cP[0] + tx - wc[0]) ** 2
                Z = np.where(dy[:, None] + dx[None, :] <= wr ** 2, Z, -1e9)
            mx = ndi.maximum_filter(Z, size=nms)
            pk = np.argwhere((Z == mx) & (Z > 1.0))
            if len(pk) == 0:
                continue
            vals = Z[pk[:, 0], pk[:, 1]]
            order = np.argsort(-vals)[:per_hyp]
            for o in order:
                ky, kx = pk[o]
                d = np.array([kx - (bshape[1] - 1), ky - (bshape[0] - 1)], float)
                t = d * b - pmin
                allc.append(dict(theta=float(th), s=float(s), t=t, score=float(vals[o]), L=Lm,
                                 H=float(H[ky, kx]), E=float(E[ky, kx])))
    allc.sort(key=lambda c: -c["score"])
    out = []
    for c in allc:
        cen = civ.mean(0) @ c["L"].T + c["t"]
        c["cen"] = cen
        if all(np.linalg.norm(cen - o["cen"]) > 40 for o in out):
            out.append(c)
        if len(out) >= topk:
            break
    return out


def density_at(cex, pts, sr=60.0):
    """Ex-vivo cell density (cells / px^2) at points, Gaussian-smoothed."""
    if len(cex) == 0:
        return np.zeros(len(pts))
    d2 = ((pts[:, None, :] - cex[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sr ** 2)).sum(1) / (2 * np.pi * sr ** 2)


def register_z(civ, cex, ex_shape, r_eval=6.0, **kw):
    cands = hough_z(civ, cex, ex_shape, **kw)
    res = []
    for c in cands:
        M, i, j, d = refine(civ, cex, cand_to_M(c))
        P = apply_aff(M, civ)
        E = (density_at(cex, P) * np.pi * r_eval ** 2).sum()
        n = (d < r_eval).sum()
        z = (n - E) / np.sqrt(E + 1)
        res.append(dict(M=M, n_inl=int(n), E=float(E), z=float(z), cand=c, hz=c["score"]))
    res.sort(key=lambda r: -r["z"])
    if not res:
        return None
    best = res[0]
    best["z2"] = res[1]["z"] if len(res) > 1 else 0.0
    best["all"] = res
    return best


# ----------------------------------------------------------------------------- local-descriptor voting
def _neighbor_maps(xy, R=60.0, b=3.0, M=None, blur=0.0):
    """For every point, rasterise the offsets of its neighbours (within R) into a
    (G x G) grid; optionally transform offsets by the 2x2 matrix M first."""
    G = int(np.ceil(2 * R / b)) + 1
    n = len(xy)
    maps = np.zeros((n, G, G), np.float32)
    if n < 2:
        return maps.reshape(n, -1)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(R * (1.15 if M is not None else 1.0), output_type="ndarray")
    if len(pairs) == 0:
        return maps.reshape(n, -1)
    a = np.r_[pairs[:, 0], pairs[:, 1]]
    c = np.r_[pairs[:, 1], pairs[:, 0]]
    off = xy[c] - xy[a]
    if M is not None:
        off = off @ M.T
    q = np.floor((off + R) / b).astype(int)
    ok = (q >= 0).all(1) & (q < G).all(1)
    np.add.at(maps, (a[ok], q[ok, 1], q[ok, 0]), 1.0)
    if blur > 0:
        maps = ndi.gaussian_filter(maps, (0, blur, blur), mode="constant") * (2 * np.pi * blur ** 2)
    return maps.reshape(n, -1)


def descriptor_register(civ, cex, angles=np.arange(-24, 24.1, 3.0), scales=(0.92, 1.0),
                        R=60.0, b=3.0, blur_px=4.0, tb=4.0, top_per_iv=2, topk=6):
    """Pairwise neighbourhood similarity -> weighted translation Hough per hypothesis."""
    if len(civ) < 4 or len(cex) < 4:
        return []
    EXm = _neighbor_maps(cex, R, b, None, blur=blur_px / b)       # (nex, G*G)
    nnb_ex = EXm.sum(1) / (2 * np.pi * (blur_px / b) ** 2 + 1e-9)
    cands = []
    for th in angles:
        for s in scales:
            Mrs = sim_matrix(th, s)
            IVm = _neighbor_maps(civ, R, b, Mrs, blur=0)             # (niv, G*G)
            S = IVm @ EXm.T                                            # soft neighbour agreement
            # expected by chance ~ n_iv_neighbours * ex density in the window
            nb_iv = IVm.sum(1)
            area = np.pi * R ** 2
            Ez = nb_iv[:, None] * (nnb_ex[None, :] / area) * (2 * np.pi * blur_px ** 2)
            Z = (S - Ez) / np.sqrt(Ez + 1.0)
            # keep the top matches per in-vivo cell
            jj = np.argsort(-Z, axis=1)[:, :top_per_iv]
            ii = np.repeat(np.arange(len(civ)), top_per_iv)
            jj = jj.ravel()
            w = np.maximum(Z[ii, jj], 0)
            keep = w > 1.5
            if keep.sum() < 3:
                continue
            ii, jj, w = ii[keep], jj[keep], w[keep]
            P = civ[ii] @ Mrs.T
            T = cex[jj] - P
            # weighted Hough on translation
            lo = T.min(0) - 3 * tb
            q = np.floor((T - lo) / tb).astype(int)
            shp = q.max(0) + 4
            Hh = np.zeros((shp[1], shp[0]))
            np.add.at(Hh, (q[:, 1], q[:, 0]), w)
            Hs = ndi.gaussian_filter(Hh, 1.0, mode="constant") * 2 * np.pi
            k = np.argmax(Hs); yy, xx = divmod(k, shp[0])
            t = lo + (np.array([xx, yy]) + 0.5) * tb
            cands.append(dict(theta=float(th), s=float(s), t=t, score=float(Hs[yy, xx])))
    cands.sort(key=lambda c: -c["score"])
    out = []
    for c in cands:
        cen = civ.mean(0) @ sim_matrix(c["theta"], c["s"]).T + c["t"]
        c["cen"] = cen
        if all(np.linalg.norm(cen - o["cen"]) > 40 for o in out):
            out.append(c)
        if len(out) >= topk:
            break
    return out


def register_desc(civ, cex, r_eval=6.0, **kw):
    cands = descriptor_register(civ, cex, **kw)
    res = []
    for c in cands:
        M, i, j, d = refine(civ, cex, cand_to_M(c))
        P = apply_aff(M, civ)
        E = (density_at(cex, P) * np.pi * r_eval ** 2).sum()
        n = (d < r_eval).sum()
        z = (n - E) / np.sqrt(E + 1)
        res.append(dict(M=M, n_inl=int(n), E=float(E), z=float(z), cand=c, hz=c["score"]))
    if not res:
        return None
    res.sort(key=lambda r: -r["hz"])
    best = res[0]
    best["z2"] = res[1]["hz"] if len(res) > 1 else 0.0
    best["all"] = res
    return best


# ----------------------------------------------------------------------------- full region matching
STRETCH = ((1.085, 158.0),)
SCALES = (0.88, 0.91, 0.94, 0.97, 1.0, 1.03)


def register_region(civ, cex, ex_shape, topk=12, angles=np.arange(-24, 24.1, 2.0), r_eval=6.0, window=None,
                    scales=SCALES, per_hyp=3):
    cands = hough_z(civ, cex, ex_shape, angles=angles, scales=scales, stretches=STRETCH, topk=topk, window=window,
                    per_hyp=per_hyp)
    res = []
    for c in cands:
        M, i, j, d = refine(civ, cex, cand_to_M(c))
        P = apply_aff(M, civ)
        E = (density_at(cex, P) * np.pi * r_eval ** 2).sum()
        n = (d < r_eval).sum()
        z = (n - E) / np.sqrt(E + 1)
        cen = apply_aff(M, civ.mean(0)[None])[0]
        th = float(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])))
        res.append(dict(M=M, n_inl=int(n), E=float(E), z=float(z), cen=cen, th=th, hz=c["score"]))
    res.sort(key=lambda r: -r["z"])
    return res


def pair_features(civ, cex, P, Tiv, Tex, reg, r=10.0, iv_shape=None):
    a, b, d, d2i, d2e = candidate_pairs(P, cex, r=r)
    if len(a) == 0:
        return a, b, np.zeros((0, 14))
    # local iv density (ambiguity) around mapped point
    tiv = cKDTree(P)
    n_iv_15 = np.array([len(tiv.query_ball_point(P[x], 15.0)) for x in a])
    tex = cKDTree(cex)
    n_ex_15 = np.array([len(tex.query_ball_point(cex[y], 15.0)) for y in b])
    H, W = iv_shape
    bd = np.minimum.reduce([civ[a, 0], civ[a, 1], W - civ[a, 0], H - civ[a, 1]])
    ri = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    iv_rank = ri(Tiv["inten"])[a] if "inten" in Tiv else np.zeros(len(a))
    ex_rank = ri(Tex["inten"])[b] if "inten" in Tex else np.zeros(len(a))
    F = np.c_[d, d2i, d2e, d / np.minimum(d2i, d2e).clip(0.5), n_iv_15, n_ex_15, bd,
              Tiv["area"][a], Tex["area"][b], iv_rank, ex_rank,
              np.full(len(a), reg["z"]), np.full(len(a), reg["n_inl"]), np.full(len(a), reg.get("zgap", 0.0))]
    return a, b, F


FEAT_NAMES = ["d", "d2i", "d2e", "dratio", "niv15", "nex15", "border", "area_iv", "area_ex",
              "rank_iv", "rank_ex", "regz", "reg_ninl", "zgap"]
