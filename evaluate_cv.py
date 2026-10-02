"""Leave-one-mouse-out evaluation of every trained configuration (CPU only).

Stage A: segmentation quality per (config, modality, held-out mouse, cellprob threshold):
         PQ and the true-positive rate on *verified* (matchable) cells.
Stage B: full competition score S for combinations of in-vivo and ex-vivo settings (registration +
         consensus + pair classifier trained without the held-out mouse), for several pair thresholds.
         Every combination is saved to runs/cv_B/ as soon as it finishes, so an interrupted run
         loses nothing, and the work can be split across several jobs (--shard/--nshards).
Aggregation: runs/cv_stageB.csv and runs/best_config.json.

usage: python evaluate_cv.py                         # everything in one process (resumable)
       python evaluate_cv.py --stage a               # stage A (reused if runs/cv_stageA.csv is complete)
       python evaluate_cv.py --stage b --shard 3 --nshards 10   # one share of stage B
       python evaluate_cv.py --stage agg             # combine whatever stage B results exist
Options: --pairs gt,oof,both compares pair classifiers; --tag NAME keeps an evaluation's task list and
results (runs/cv_stageB_NAME.csv, runs/best_config_NAME.json) apart from the others.
"""
import os, sys, json, argparse, time
from multiprocessing import Pool
import numpy as np, pandas as pd

from common import RUNS, ROOT, gt_labels, training_ids, load_images, masks_from_flows, load_flows, run_dir, has_flows
from configs import CONFIGS, ENSEMBLES, SUBJECTS, CP_GRID, THR_GRID

BDIR = os.path.join(RUNS, "cv_B")
A_CSV = os.path.join(RUNS, "cv_stageA.csv")
tasks_json = lambda tag: os.path.join(BDIR, f"tasks{tag}.json")


def complete(cfg, mod, fold):
    rd = run_dir(cfg, mod, fold)
    return all(has_flows(rd, s) for s in training_ids(subjects=[fold]))


def trained_configs():
    """every config with cross-validation outputs in runs/, plus the ensembles of configs.py"""
    have = [c for c in sorted(os.listdir(RUNS)) if c not in ("shipped", "cv_B") and
            any(os.path.isdir(os.path.join(RUNS, c, m)) for m in ("iv", "ex"))] if os.path.isdir(RUNS) else []
    return have + [e for e in ENSEMBLES if e not in have]


PAIR_FILES = {"gt": "pairs_gt.pkl", "oof": "pairs_oof.pkl"}


def fit_clf(pairs, fold):
    """pair classifier fitted without the held-out mouse. pairs: gt (candidates from ground-truth masks,
    the shipped training set), oof (from out-of-fold predicted masks, build_pairs_oof.py), both, or any
    weights/pairs_<name>.pkl. Rows with an "outer" mouse (build_pairs_oof.py --strict) are used only when
    that mouse is the one held out, so no row comes from a segmentation model that saw it."""
    import pickle
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = []
    for k in (["gt", "oof"] if pairs == "both" else [pairs]):
        rows += pickle.load(open(os.path.join(ROOT, "weights", PAIR_FILES.get(k, f"pairs_{k}.pkl")), "rb"))
    use = lambda r: r["outer"] == fold if ("outer" in r and fold != "none") else r["sid"].split("__")[0] != fold
    tr = [r for r in rows if use(r) and len(r["y"])]
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0).fit(
        np.concatenate([r["F"] for r in tr]), np.concatenate([r["y"] for r in tr]))


_FP = {}


def fingerprint(pairs):
    """short hash of the matching / scoring code, the pair thresholds, the classifier fitting code and the pair-classifier
    data; part of every cached result's name, so a change to any of them recomputes the results instead of reusing stale ones"""
    if pairs not in _FP:
        import hashlib
        h = hashlib.sha1()
        import inspect
        for f in ("common.py", "cmutil.py", "match.py", "pipeline.py", "consensus2.py", "cm_pipeline.py"):
            h.update(open(os.path.join(ROOT, f), "rb").read())
        h.update(repr(THR_GRID).encode() + inspect.getsource(fit_clf).encode())   # what a cached result contains
        for k in (["gt", "oof"] if pairs == "both" else [pairs]):
            fn = os.path.join(ROOT, "weights", PAIR_FILES.get(k, f"pairs_{k}.pkl"))
            h.update(open(fn, "rb").read() if os.path.exists(fn) else b"missing")
        _FP[pairs] = h.hexdigest()[:8]
    return _FP[pairs]


def stage_a(task):
    cfg, mod, fold = task
    from cm_pipeline import tp_map
    rd = run_dir(cfg, mod, fold); rows = []
    for sid in training_ids(subjects=[fold]):
        giv, gex, gp = gt_labels(sid)
        gl = giv if mod == "iv" else gex
        ver = set(int(v) for v in (gp[:, 0] if mod == "iv" else gp[:, 1]))
        dP, cp, up = load_flows(rd, sid)
        for c in CP_GRID[mod]:
            lab = masks_from_flows(dP, cp, c, up)
            mp, pq = tp_map(lab, gl)
            hit = {g for g, _ in mp.values()}
            rows.append(dict(config=cfg, mod=mod, fold=fold, cp=c, sid=sid, pq=pq, n_pred=int(lab.max()),
                             ver_hits=len(ver & hit), n_ver=len(ver)))
    print("A", cfg, mod, fold, flush=True)
    return rows


def b_key(task):
    fold, (ivc, ivcp), (exc, excp), pairs = task
    from cm_pipeline import parse_method
    return (f"{fold}__{ivc}_{ivcp:g}__{exc}_{excp:g}" + ("" if pairs == "gt" else f"__pairs-{pairs}")
            + f"__{fingerprint(parse_method(pairs)[0])}")


def stage_b(task):
    out_fn = os.path.join(BDIR, b_key(task) + ".csv")
    if os.path.exists(out_fn):
        return out_fn
    fold, (ivc, ivcp), (exc, excp), ptag = task
    t0 = time.time()
    from cm_pipeline import match_regions, score, parse_method, region_item, match_kw
    pairs_data, opt = parse_method(ptag)
    clf = fit_clf(pairs_data, fold)
    items, gts = [], {}
    for sid in training_ids(subjects=[fold]):
        iv, ex = load_images(sid, "training")
        items.append(region_item(sid, iv, ex, load_flows(run_dir(ivc, "iv", fold), sid),
                                 load_flows(run_dir(exc, "ex", fold), sid), ivcp, excp, opt))
        gts[sid] = gt_labels(sid)
    pairs, log = match_regions(items, lambda s: clf, thrs=tuple(THR_GRID), **match_kw(opt))
    out = []
    for t in THR_GRID:
        pred = {it["sid"]: (it["liv"], it["lex"], pairs[t][it["sid"]]) for it in items}
        s, per = score(pred, gts)
        out.append(dict(fold=fold, iv_config=ivc, cp_iv=ivcp, ex_config=exc, cp_ex=excp, pairs=ptag, thr=t, **s,
                        n_regions=len(per), pq_iv_sum=per.pq_iv.sum(), pq_ex_sum=per.pq_ex.sum()))
    tmp = out_fn + f".{os.getpid()}.tmp"
    pd.DataFrame(out).to_csv(tmp, index=False)
    os.replace(tmp, out_fn)
    print("B", fold, ivc, ivcp, exc, excp, ptag, f"S={out[1]['S']:.4f}", f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out_fn


def stage_b_safe(task):
    # one failing combination must not stop the rest of this share
    try:
        return stage_b(task)
    except Exception as e:
        import traceback
        print("B FAILED", b_key(task), repr(e), flush=True); traceback.print_exc()
        return None


def summarize_a(A):
    # per fold: region-mean PQ, pooled verified TP rate; then average over folds (only settings run on all folds)
    g = A.groupby(["config", "mod", "cp", "fold"]).agg(pq=("pq", "mean"), vh=("ver_hits", "sum"), nv=("n_ver", "sum")).reset_index()
    g["ver_tp"] = g.vh / g.nv.clip(lower=1)
    s = g.groupby(["config", "mod", "cp"]).agg(pq=("pq", "mean"), ver_tp=("ver_tp", "mean"), folds=("fold", "count")).reset_index()
    s = s[s.folds == s.folds.max()] if len(s) else s
    return g, s


def run_stage_a(configs, workers):
    tasks = [(c, m, f) for c in configs for m in ("iv", "ex") for f in SUBJECTS if complete(c, m, f)]
    missing = [(c, m, f) for c in configs for m in ("iv", "ex") for f in SUBJECTS if (c, m, f) not in tasks]
    if missing:
        print("not finished (skipped):", missing)
    A = pd.read_csv(A_CSV) if os.path.exists(A_CSV) else pd.DataFrame(columns=["config", "mod", "fold"])
    # reused only if every threshold of CP_GRID is there (thresholds added to the grid are computed)
    have = {k for k, g in A.groupby(["config", "mod", "fold"]) if set(np.round(g.cp, 3)) >= set(np.round(CP_GRID[k[1]], 3))}
    todo = [t for t in tasks if t not in have]
    print(f"stage A: {len(tasks) - len(todo)} (config, modality, mouse) reused from {A_CSV}, {len(todo)} to compute", flush=True)
    if todo:
        for sid in training_ids(subjects=SUBJECTS):  # build the ground-truth label cache once, serially
            gt_labels(sid)
        with Pool(min(workers, len(todo))) as pool:
            new = pd.DataFrame([r for rows in pool.map(stage_a, todo) for r in rows])
        td = set(todo)
        A = A[[t not in td for t in zip(A.config, A["mod"], A.fold)]]   # their partial rows are replaced
        A = pd.concat([A, new], ignore_index=True) if len(A) else new
        A.to_csv(A_CSV + f".{os.getpid()}.tmp", index=False)
        os.replace(A_CSV + f".{os.getpid()}.tmp", A_CSV)   # other evaluations may be reading it
    keep = set(tasks)
    return A[[(c, m, f) in keep for c, m, f in zip(A.config, A["mod"], A.fold)]]


def make_b_tasks(A, iv_top, iv_extra, ex_top, ex_extra, ex_cps, pairs):
    g, s = summarize_a(A)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("\n=== Stage A (mean over held-out mice) ===")
    print(s.sort_values(["mod", "pq"], ascending=[True, False]).round(4).to_string(index=False))
    siv = s[s["mod"] == "iv"].sort_values("pq", ascending=False)
    iv_c = [(c, float(v)) for c, v in siv[["config", "cp"]].values[:iv_top]]
    for x in filter(None, iv_extra.split(",")):          # e.g. the settings of the current best submission
        c, v = x.split(":")
        if (c, float(v)) not in iv_c and ((siv.config == c) & np.isclose(siv.cp, float(v))).any():
            iv_c.append((c, float(v)))
    sex = s[s["mod"] == "ex"]
    ex_rank = sex.groupby("config").ver_tp.max().sort_values(ascending=False)   # verified cells drive the F1 term
    ex_cfgs = list(ex_rank.index[:ex_top])
    ex_cfgs += [c for c in filter(None, ex_extra.split(",")) if c not in ex_cfgs and c in ex_rank.index]
    ex_c = [(c, float(v)) for c in ex_cfgs for v in ex_cps.split(",")]
    folds = sorted(set(g.fold))
    tasks = [(f, i, e, p) for f in folds for i in iv_c for e in ex_c for p in pairs.split(",")
             if complete(i[0], "iv", f) and complete(e[0], "ex", f)]
    print(f"\nStage B: {len(tasks)} runs (iv settings {iv_c}; ex settings {ex_c}; pair classifiers {pairs})", flush=True)
    return tasks


def aggregate(tag):
    tasks = [(f, tuple(i), tuple(e), p) for f, i, e, p in json.load(open(tasks_json(tag)))]
    files = [os.path.join(BDIR, b_key(t) + ".csv") for t in tasks]
    have = [f for f in files if os.path.exists(f)]
    print(f"stage B results: {len(have)}/{len(files)} combinations finished")
    if not have:
        sys.exit("nothing to aggregate yet")
    B = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    if "pairs" not in B:
        B["pairs"] = "gt"
    B["pairs"] = B.pairs.fillna("gt")        # results saved before the pairs option existed
    B.to_csv(os.path.join(RUNS, f"cv_stageB{tag}.csv"), index=False)
    key = ["iv_config", "cp_iv", "ex_config", "cp_ex", "pairs", "thr"]
    # pooled over all held-out mice, like the leaderboard (PQ averaged over regions, F1 pooled over pairs)
    P = B.groupby(key).agg(folds=("fold", "count"), regions=("n_regions", "sum"), pqi=("pq_iv_sum", "sum"),
                           pqe=("pq_ex_sum", "sum"), c=("correct", "sum"), p=("pred_pairs", "sum"), gtp=("gt_pairs", "sum"),
                           S_foldmean=("S", "mean")).reset_index()
    P = P[P.folds == P.folds.max()]
    P["pq_iv"] = P.pqi / P.regions; P["pq_ex"] = P.pqe / P.regions
    P["prec"] = P.c / P.p.clip(lower=1); P["rec"] = P.c / P.gtp.clip(lower=1)
    P["f1"] = 2 * P.prec * P.rec / (P.prec + P.rec).clip(lower=1e-9)
    P["S_pooled"] = 0.25 * (P.pq_iv + P.pq_ex) + 0.5 * P.f1
    P = P.sort_values("S_pooled", ascending=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(f"\n=== Stage B{' [' + tag[1:] + ']' if tag else ''}: full score, pooled over held-out mice ({int(P.folds.max())} mice) ===")
    print(P[key + ["S_pooled", "S_foldmean", "pq_iv", "pq_ex", "f1", "prec", "rec"]].head(25).round(4).to_string(index=False))
    best = P.iloc[0]
    bj = dict(iv_config=best.iv_config, cp_iv=float(best.cp_iv), ex_config=best.ex_config, cp_ex=float(best.cp_ex),
              pairs=best.pairs, thr=float(best.thr), S_pooled=float(best.S_pooled), pq_iv=float(best.pq_iv), pq_ex=float(best.pq_ex),
              f1=float(best.f1), combinations_scored=len(have), combinations_total=len(files))
    json.dump(bj, open(os.path.join(RUNS, f"best_config{tag}.json"), "w"), indent=1)
    print("\nbest:", bj)
    if len(have) < len(files):
        print(f"WARNING: {len(files) - len(have)} combinations missing - rerun `bash submit_eval.sh` to finish them "
              "(finished ones are skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "agg"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--configs", default=None, help="default: every config trained in runs/ + configs.ENSEMBLES")
    ap.add_argument("--iv-top", type=int, default=2, help="best in-vivo settings (by PQ) carried into stage B")
    ap.add_argument("--iv-extra", default="cyto3_x3_auto:-1,cyto3_x3:-1", help="in-vivo settings always included (config:cellprob)")
    ap.add_argument("--ex-top", type=int, default=4, help="best ex-vivo configs (by verified-cell hits) carried into stage B")
    ap.add_argument("--ex-extra", default="cpsam2_x3,cyto3_x3", help="ex-vivo configs always included")
    ap.add_argument("--ex-cps", default="-1.5,-1.25,-1,-0.75,-0.5,0", help="ex-vivo thresholds (the leaderboard best is -1)")
    ap.add_argument("--pairs", default="gt", help="pipeline variants to compare (comma list): pair data gt / oof / both, "
                    "optionally with options, e.g. oof+weak=keep+bright=0.5 (see cm_pipeline.METHOD_DEFAULTS)")
    ap.add_argument("--tag", default="", help="name for this evaluation (keeps its task list / results apart)")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(BDIR, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    if args.stage in ("all", "a"):
        A = run_stage_a(args.configs.split(",") if args.configs else trained_configs(), args.workers)
        tasks = make_b_tasks(A, args.iv_top, args.iv_extra, args.ex_top, args.ex_extra, args.ex_cps, args.pairs)
        json.dump(tasks, open(tasks_json(tag), "w"), indent=0)
    if args.stage in ("all", "b"):
        tasks = [(f, tuple(i), tuple(e), p) for f, i, e, p in json.load(open(tasks_json(tag)))]
        mine = tasks[args.shard::args.nshards]
        todo = [t for t in mine if not os.path.exists(os.path.join(BDIR, b_key(t) + ".csv"))]
        print(f"stage B share {args.shard + 1}/{args.nshards}: {len(mine)} combinations, {len(todo)} still to run", flush=True)
        if todo:
            with Pool(min(args.workers, len(todo))) as pool:
                pool.map(stage_b_safe, todo, chunksize=1)
    if args.stage in ("all", "agg"):
        aggregate(tag)
    print(f"({(time.time() - t0) / 60:.1f} min)")
