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
"""
import os, sys, json, argparse, time
from multiprocessing import Pool
import numpy as np, pandas as pd

from common import RUNS, gt_labels, training_ids, load_images, masks_from_flows, load_flows, run_dir, flows_path
from configs import CV_CONFIGS, SUBJECTS, CP_GRID, THR_GRID

BDIR = os.path.join(RUNS, "cv_B")
A_CSV = os.path.join(RUNS, "cv_stageA.csv")
TASKS_JSON = os.path.join(BDIR, "tasks.json")


def complete(cfg, mod, fold):
    rd = run_dir(cfg, mod, fold)
    return all(os.path.exists(flows_path(rd, s)) for s in training_ids(subjects=[fold]))


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
    fold, (ivc, ivcp), (exc, excp) = task
    return f"{fold}__{ivc}_{ivcp:g}__{exc}_{excp:g}"


def stage_b(task):
    out_fn = os.path.join(BDIR, b_key(task) + ".csv")
    if os.path.exists(out_fn):
        return out_fn
    fold, (ivc, ivcp), (exc, excp) = task
    t0 = time.time()
    import pickle
    from sklearn.ensemble import HistGradientBoostingClassifier
    from cm_pipeline import match_regions, score
    rows_gt = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "pairs_gt.pkl"), "rb"))
    tr = [r for r in rows_gt if r["sid"].split("__")[0] != fold]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0).fit(
        np.concatenate([r["F"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    items, gts = [], {}
    for sid in training_ids(subjects=[fold]):
        iv, ex = load_images(sid, "training")
        dP, cp, up = load_flows(run_dir(ivc, "iv", fold), sid)
        liv = masks_from_flows(dP, cp, ivcp, up)
        dP, cp, up = load_flows(run_dir(exc, "ex", fold), sid)
        lex = masks_from_flows(dP, cp, excp, up)
        items.append(dict(sid=sid, iv_img=iv, ex_img=ex, liv=liv, lex=lex))
        gts[sid] = gt_labels(sid)
    pairs, log = match_regions(items, lambda s: clf, thrs=tuple(THR_GRID))
    out = []
    for t in THR_GRID:
        pred = {it["sid"]: (it["liv"], it["lex"], pairs[t][it["sid"]]) for it in items}
        s, per = score(pred, gts)
        out.append(dict(fold=fold, iv_config=ivc, cp_iv=ivcp, ex_config=exc, cp_ex=excp, thr=t, **s,
                        n_regions=len(per), pq_iv_sum=per.pq_iv.sum(), pq_ex_sum=per.pq_ex.sum()))
    tmp = out_fn + f".{os.getpid()}.tmp"
    pd.DataFrame(out).to_csv(tmp, index=False)
    os.replace(tmp, out_fn)
    print("B", fold, ivc, ivcp, exc, excp, f"S={out[1]['S']:.4f}", f"({(time.time() - t0) / 60:.1f} min)", flush=True)
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
    A = pd.read_csv(A_CSV) if os.path.exists(A_CSV) else None
    have = set() if A is None else set(map(tuple, A[["config", "mod", "fold"]].drop_duplicates().values))
    if A is not None and set(tasks) <= have:
        print(f"stage A: reusing {A_CSV}")
        A = A[A.apply(lambda r: (r.config, r["mod"], r.fold) in set(tasks), axis=1)]
    else:
        for sid in training_ids(subjects=SUBJECTS):  # build the ground-truth label cache once, serially
            gt_labels(sid)
        with Pool(workers) as pool:
            A = pd.DataFrame([r for rows in pool.map(stage_a, tasks) for r in rows])
        A.to_csv(A_CSV, index=False)
    return A


def make_b_tasks(A, iv_top, iv_extra, ex_top, ex_cps):
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
    ex_cfgs = list(ex_rank.index[:ex_top] if ex_top else ex_rank.index)
    ex_c = [(c, float(v)) for c in ex_cfgs for v in ex_cps.split(",")]
    folds = sorted(set(g.fold))
    tasks = [(f, i, e) for f in folds for i in iv_c for e in ex_c if complete(i[0], "iv", f) and complete(e[0], "ex", f)]
    print(f"\nStage B: {len(tasks)} runs (iv settings {iv_c}; ex settings {ex_c})", flush=True)
    return tasks


def aggregate():
    tasks = [(f, tuple(i), tuple(e)) for f, i, e in json.load(open(TASKS_JSON))]
    files = [os.path.join(BDIR, b_key(t) + ".csv") for t in tasks]
    have = [f for f in files if os.path.exists(f)]
    print(f"stage B results: {len(have)}/{len(files)} combinations finished")
    if not have:
        sys.exit("nothing to aggregate yet")
    B = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    B.to_csv(os.path.join(RUNS, "cv_stageB.csv"), index=False)
    key = ["iv_config", "cp_iv", "ex_config", "cp_ex", "thr"]
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
    print(f"\n=== Stage B: full score, pooled over held-out mice ({int(P.folds.max())} mice) ===")
    print(P[key + ["S_pooled", "S_foldmean", "pq_iv", "pq_ex", "f1", "prec", "rec"]].head(25).round(4).to_string(index=False))
    best = P.iloc[0]
    bj = dict(iv_config=best.iv_config, cp_iv=float(best.cp_iv), ex_config=best.ex_config, cp_ex=float(best.cp_ex),
              thr=float(best.thr), S_pooled=float(best.S_pooled), pq_iv=float(best.pq_iv), pq_ex=float(best.pq_ex),
              f1=float(best.f1), combinations_scored=len(have), combinations_total=len(files))
    json.dump(bj, open(os.path.join(RUNS, "best_config.json"), "w"), indent=1)
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
    ap.add_argument("--configs", default=",".join(CV_CONFIGS))
    ap.add_argument("--iv-top", type=int, default=2, help="best in-vivo settings (by PQ) carried into stage B")
    ap.add_argument("--iv-extra", default="cyto3_x3:-1", help="in-vivo settings always included (config:cellprob)")
    ap.add_argument("--ex-top", type=int, default=3, help="ex-vivo configs carried into stage B (0 = all)")
    ap.add_argument("--ex-cps", default="-0.5,0,0.5")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(BDIR, exist_ok=True)

    if args.stage in ("all", "a"):
        A = run_stage_a(args.configs.split(","), args.workers)
        tasks = make_b_tasks(A, args.iv_top, args.iv_extra, args.ex_top, args.ex_cps)
        json.dump(tasks, open(TASKS_JSON, "w"), indent=0)
    if args.stage in ("all", "b"):
        tasks = [(f, tuple(i), tuple(e)) for f, i, e in json.load(open(TASKS_JSON))]
        mine = tasks[args.shard::args.nshards]
        todo = [t for t in mine if not os.path.exists(os.path.join(BDIR, b_key(t) + ".csv"))]
        print(f"stage B share {args.shard + 1}/{args.nshards}: {len(mine)} combinations, {len(todo)} still to run", flush=True)
        if todo:
            with Pool(min(args.workers, len(todo))) as pool:
                pool.map(stage_b_safe, todo, chunksize=1)
    if args.stage in ("all", "agg"):
        aggregate()
    print(f"({(time.time() - t0) / 60:.1f} min)")
