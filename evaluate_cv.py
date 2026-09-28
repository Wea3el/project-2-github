"""Leave-one-mouse-out evaluation of every trained configuration (CPU only).

Stage A: segmentation quality per (config, modality, held-out mouse, cellprob threshold):
         PQ and the true-positive rate on *verified* (matchable) cells.
Stage B: full competition score S for combinations of the best in-vivo settings with every
         ex-vivo setting (registration + consensus + pair classifier trained without the
         held-out mouse), for several pair thresholds.
Writes runs/cv_stageA.csv, runs/cv_stageB.csv and runs/best_config.json.

usage: python evaluate_cv.py [--workers 8] [--configs base,cyto3_x3]
"""
import os, sys, json, argparse, itertools, time
from multiprocessing import Pool
import numpy as np, pandas as pd

from common import RUNS, gt_labels, training_ids, load_images, masks_from_flows, load_flows, run_dir, flows_path
from configs import CV_CONFIGS, SUBJECTS, CP_GRID, THR_GRID

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--configs", default=",".join(CV_CONFIGS))
ap.add_argument("--iv-top", type=int, default=2, help="how many in-vivo settings to carry into stage B")
ap.add_argument("--ex-cps", default="-0.5,0,0.5")
args = ap.parse_args()
configs = args.configs.split(",")


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


def stage_b(task):
    fold, (ivc, ivcp), (exc, excp) = task
    import pickle
    from sklearn.ensemble import HistGradientBoostingClassifier
    from cm_pipeline import match_regions, score
    rows_gt = pickle.load(open(os.path.join(os.path.dirname(__file__), "weights", "pairs_gt.pkl"), "rb"))
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
    print("B", fold, ivc, ivcp, exc, excp, f"S={out[1]['S']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    for sid in training_ids(subjects=SUBJECTS):  # build the ground-truth label cache once, serially
        gt_labels(sid)
    tasks = [(c, m, f) for c in configs for m in ("iv", "ex") for f in SUBJECTS if complete(c, m, f)]
    missing = [(c, m, f) for c in configs for m in ("iv", "ex") for f in SUBJECTS if (c, m, f) not in tasks]
    if missing:
        print("not finished (skipped):", missing)
    with Pool(args.workers) as pool:
        A = pd.DataFrame([r for rows in pool.map(stage_a, tasks) for r in rows])
    A.to_csv(os.path.join(RUNS, "cv_stageA.csv"), index=False)
    # per fold: region-mean PQ, pooled verified TP rate; then average over folds (only settings run on all folds)
    g = A.groupby(["config", "mod", "cp", "fold"]).agg(pq=("pq", "mean"), vh=("ver_hits", "sum"), nv=("n_ver", "sum")).reset_index()
    g["ver_tp"] = g.vh / g.nv.clip(lower=1)
    s = g.groupby(["config", "mod", "cp"]).agg(pq=("pq", "mean"), ver_tp=("ver_tp", "mean"), folds=("fold", "count")).reset_index()
    s = s[s.folds == s.folds.max()] if len(s) else s
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("\n=== Stage A (mean over held-out mice) ===")
    print(s.sort_values(["mod", "pq"], ascending=[True, False]).round(4).to_string(index=False))

    iv_c = [tuple(x) for x in s[s["mod"] == "iv"].sort_values("pq", ascending=False)[["config", "cp"]].values[:args.iv_top]]
    ex_cfgs = sorted(set(s[s["mod"] == "ex"].config))
    ex_c = [(c, float(v)) for c in ex_cfgs for v in args.ex_cps.split(",")]
    folds = sorted(set(g.fold))
    tasksB = [(f, i, e) for f in folds for i in iv_c for e in ex_c
              if complete(i[0], "iv", f) and complete(e[0], "ex", f)]
    print(f"\nStage B: {len(tasksB)} runs (iv settings {iv_c}; ex settings {ex_c})", flush=True)
    with Pool(args.workers) as pool:
        B = pd.DataFrame([r for rows in pool.map(stage_b, tasksB) for r in rows])
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
    print("\n=== Stage B: full score, pooled over held-out mice ===")
    print(P[key + ["S_pooled", "S_foldmean", "pq_iv", "pq_ex", "f1", "prec", "rec"]].head(25).round(4).to_string(index=False))
    best = P.iloc[0]
    bj = dict(iv_config=best.iv_config, cp_iv=float(best.cp_iv), ex_config=best.ex_config, cp_ex=float(best.cp_ex),
              thr=float(best.thr), S_pooled=float(best.S_pooled), pq_iv=float(best.pq_iv), pq_ex=float(best.pq_ex),
              f1=float(best.f1))
    json.dump(bj, open(os.path.join(RUNS, "best_config.json"), "w"), indent=1)
    print("\nbest:", bj, f"\n({(time.time() - t0) / 60:.1f} min)")
