"""Per-region choice between several versions of the in-vivo masks.

Small changes to the in-vivo masks move the registration of some regions (a region then gains or loses
all of its pairs). This runs the whole pipeline once per version and keeps, for every region, the
version the pair classifier is most confident about (by default: the largest sum of accepted-pair
probabilities, i.e. the expected number of correct pairs).

  cv    leave-one-mouse-out: every version and selection rule scored with the competition metric
        python select_versions.py cv --versions cyto3_x3:-1,cyto3_x2:-0.5 --ex cyto3_x3:0 --pairs oof --thr 0.05
  test  build a submission with one rule
        python select_versions.py test --versions cyto3_x3_auto:-1,cyto3_x2:-0.5 --ex cyto3_x3:0 --pairs oof \\
            --thr 0.05 --rule pen --lam 0.2 --out submission_sel.csv
  Use the same --versions for cv and test.
A version is config:cellprob. Per-version results are saved in runs/select/ and reused when rerun.
"""
import os, sys, json, argparse, time
from multiprocessing import Pool
import numpy as np, pandas as pd

from common import DATA, RUNS, gt_labels, training_ids, load_images, load_flows, run_dir
from configs import SUBJECTS
from cm_pipeline import match_regions, score, to_rows, parse_method, mod_masks, match_kw

SDIR = os.path.join(RUNS, "select")
RULES = {  # rule -> per-region score to maximise (ties: the first version listed wins)
    "conf": lambda r: r.conf,        # expected number of correct pairs
    "npairs": lambda r: r.n_pred,    # most pairs
    "z": lambda r: r.z,              # most confident registration
    # "pen": conf - lam * pairs, i.e. the summed (probability - lam) over the accepted pairs: an extra pair only
    # helps if it is likely enough to be right; lam is chosen on the held-out mice (cv mode prints a sweep)
}
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def parse_version(v):
    c, cp = v.rsplit(":", 1)
    return c, float(cp)


def fit(pairs, fold):
    from evaluate_cv import fit_clf
    return fit_clf(pairs, fold)


def run_version(task):
    """One version on one held-out mouse (cv) or on the test set (test): per-region results."""
    mode, fold, version, ex, method, thr = task
    from evaluate_cv import fingerprint
    tag = (f"{mode}__{fold}__{version.replace(':', '_')}__{ex.replace(':', '_')}__{method}__{thr:g}"
           f"__{fingerprint(parse_method(method)[0])}")
    out_fn = os.path.join(SDIR, tag + ".pkl")
    if os.path.exists(out_fn):
        return out_fn
    t0 = time.time()
    (ivc, ivcp), (exc, excp) = parse_version(version), parse_version(ex)
    pairs_data, opt = parse_method(method)
    split = "training" if mode == "cv" else "hidden_test"
    ids = training_ids(subjects=[fold]) if mode == "cv" else list(pd.read_csv(os.path.join(DATA, "sample_submission.csv")).sample_id)
    rd_fold = fold if mode == "cv" else "full"
    items = []
    for sid in ids:
        iv, ex_img = load_images(sid, split)
        dP, cp, up = load_flows(run_dir(ivc, "iv", rd_fold), sid); liv = mod_masks(dP, cp, ivcp, up, opt, "iv")
        dP, cp, up = load_flows(run_dir(exc, "ex", rd_fold), sid); lex = mod_masks(dP, cp, excp, up, opt, "ex")
        items.append(dict(sid=sid, iv_img=iv, ex_img=ex_img, liv=liv, lex=lex))
    clf = fit(pairs_data, fold if mode == "cv" else "none")
    feats = {}
    pairs, log = match_regions(items, lambda s: clf, thrs=(thr,), feats=feats, **match_kw(opt))
    L = log.set_index("sid")
    rows = []
    for it in items:
        sid = it["sid"]; P = pairs[thr][sid]
        la, lb, F = feats.get(sid, ([], [], np.zeros((0, 14))))
        p = clf.predict_proba(F)[:, 1] if len(F) else np.zeros(0)
        prob = {(int(a), int(b)): float(q) for a, b, q in zip(la, lb, p)}
        r = dict(sid=sid, version=version, conf=float(sum(prob.get((int(a), int(b)), 0.0) for a, b in P)),
                 n_pred=len(P), z=float(L.loc[sid, "z"]) if np.isfinite(L.loc[sid, "z"]) else -1.0, status=L.loc[sid, "status"])
        if mode == "cv":
            _, per = score({sid: (it["liv"], it["lex"], P)}, {sid: gt_labels(sid)})
            r.update(per.iloc[0][["pq_iv", "pq_ex", "correct", "n_gt"]].to_dict())
        else:
            r["row"] = to_rows(sid, it["liv"], it["lex"], P)
        rows.append(r)
    tmp = out_fn + f".{os.getpid()}.tmp"
    pd.to_pickle(pd.DataFrame(rows), tmp); os.replace(tmp, out_fn)
    print(f"{mode} {fold} {version}: {sum(r['n_pred'] for r in rows)} pairs ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out_fn


def pooled_S(D):
    c, p, g = D.correct.sum(), D.n_pred.sum(), D.n_gt.sum()
    prec, rec = c / max(p, 1), c / max(g, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(S=0.25 * (D.pq_iv.mean() + D.pq_ex.mean()) + 0.5 * f1, pq_iv=D.pq_iv.mean(), pq_ex=D.pq_ex.mean(),
                f1=f1, prec=prec, rec=rec, pairs=int(p))


def choose(D, versions, rule, lam=0.0):
    """per region, the row of the version with the highest rule score (ties -> earlier version)"""
    order = {v: i for i, v in enumerate(versions)}
    sc = D.conf - lam * D.n_pred if rule == "pen" else D.apply(RULES[rule], axis=1)
    D = D.assign(_s=sc, _o=D.version.map(order))
    return D.sort_values(["sid", "_s", "_o"], ascending=[True, False, True]).groupby("sid").head(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "test"])
    ap.add_argument("--versions", required=True, help="comma list of in-vivo config:cellprob")
    ap.add_argument("--ex", default="cyto3_x3:0")
    ap.add_argument("--pairs", default="oof", help="pipeline variant (see cm_pipeline.parse_method)")
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--rule", default="pen", choices=list(RULES) + ["pen"])
    ap.add_argument("--lam", type=float, default=0.2, help="pen: penalty per accepted pair (pick it from the cv sweep)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="submission_sel.csv")
    args = ap.parse_args()
    os.makedirs(SDIR, exist_ok=True)
    versions = args.versions.split(",")
    folds = SUBJECTS if args.mode == "cv" else ["test"]
    tasks = [(args.mode, f, v, args.ex, args.pairs, args.thr) for f in folds for v in versions]
    with Pool(min(args.workers, len(tasks))) as pool:
        files = pool.map(run_version, tasks, chunksize=1)
    D = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)

    if args.mode == "cv":
        res = [dict(choice=v, **pooled_S(D[D.version == v])) for v in versions]
        for rule in RULES:
            res.append(dict(choice=f"per-region: {rule}", **pooled_S(choose(D, versions, rule))))
        for lam in LAMS:
            res.append(dict(choice=f"per-region: pen lam={lam:g}", **pooled_S(choose(D, versions, "pen", lam))))
        best = D.sort_values(["sid", "correct"], ascending=[True, False]).groupby("sid").head(1)
        res.append(dict(choice="per-region: oracle (upper bound)", **pooled_S(best)))
        R = pd.DataFrame(res)
        print("\n=== held-out score of each version and of each per-region rule ===")
        print(R.round(4).to_string(index=False))
        C = choose(D, versions, args.rule, args.lam)
        print(f"\nversions chosen by '{args.rule}' (lam {args.lam:g}):", C.version.value_counts().to_dict())
        P = R[R.choice.str.startswith("per-region: pen")]
        print("best penalty on the held-out mice:", P.loc[P.S.idxmax(), "choice"], f"(S {P.S.max():.4f})")
        R.to_csv(os.path.join(RUNS, "select_cv.csv"), index=False)
    else:
        C = choose(D, versions, args.rule, args.lam)
        sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
        C = C.set_index("sid").loc[sample.sample_id]
        sub = pd.DataFrame(list(C.row))
        assert list(sub.sample_id) == list(sample.sample_id)
        sub.to_csv(args.out, index=False)
        W = D.pivot(index="sid", columns="version", values="n_pred").loc[sample.sample_id]
        W["chosen"] = C.version
        print(W.to_string())
        print("versions chosen:", C.version.value_counts().to_dict(), "| pairs:", int(C.n_pred.sum()), "| wrote", args.out)
