"""Does the cell-probability threshold act the same on the test mice as on the training mice?
For each config, modality and threshold: cells per region, median cell area and the median network
cell probability inside the cells, per mouse. Training mice use the cross-validation outputs (model
trained without that mouse), test mice the all-mice model, so every mouse is unseen by its model.
Training rows also give the ground-truth cell count / area and PQ.

usage: python calib_check.py --configs cyto3_x3,cpsam2_x3 [--cps=-1.5,-1,-0.5,0,0.5,1] [--out runs/calib.csv]
"""
import os, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

from common import DATA, RUNS, gt_labels, training_ids, masks_from_flows, load_flows, run_dir, has_flows
from configs import SUBJECTS


def one(task):
    cfg, rd, mod, sid, cps = task
    from cm_pipeline import tp_map
    dP, prob, up = load_flows(rd, sid)
    test = rd.endswith(os.sep + "full")
    g = None if test else gt_labels(sid)[0 if mod == "iv" else 1]
    rows = []
    for c in cps:
        lab = masks_from_flows(dP, prob, c, up)
        a = np.bincount(lab.ravel())[1:]
        a = a[a > 0]
        r = dict(config=cfg, mod=mod, split="test" if test else "train", mouse=sid.split("__")[0][-6:], sid=sid, cp=c,
                 cells=len(a), area=float(np.median(a)) if len(a) else np.nan,
                 inside=float(np.median(prob[lab > 0])) if len(a) else np.nan)
        if g is not None:
            ga = np.bincount(g.ravel())[1:]
            ga = ga[ga > 0]
            r.update(gt_cells=len(ga), gt_area=float(np.median(ga)), pq=tp_map(lab, g)[1])
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="cyto3_x3,cpsam2_x3")
    ap.add_argument("--cps", default="-1.5,-1,-0.5,0,0.5,1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RUNS, "calib.csv"))
    args = ap.parse_args()
    cps = [float(c) for c in args.cps.split(",")]
    test_ids = list(pd.read_csv(os.path.join(DATA, "sample_submission.csv")).sample_id)
    tasks = []
    for cfg in args.configs.split(","):
        for mod in ("iv", "ex"):
            for f in SUBJECTS:
                rd = run_dir(cfg, mod, f)
                tasks += [(cfg, rd, mod, s, cps) for s in training_ids(subjects=[f]) if has_flows(rd, s)]
            # test outputs: the plain config, else its per-region rescaled variant (in-vivo runs as <config>_auto)
            for c in (cfg, cfg + "_auto"):
                rd = run_dir(c, mod, "full")
                have = [s for s in test_ids if has_flows(rd, s)]
                if have:
                    tasks += [(cfg, rd, mod, s, cps) for s in have]
                    break
    with Pool(min(args.workers, max(1, len(tasks)))) as pool:
        R = pd.DataFrame([r for rows in pool.map(one, tasks, chunksize=1) for r in rows])
    R.to_csv(args.out, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    agg = dict(regions=("sid", "nunique"), cells=("cells", "mean"), area=("area", "median"), inside=("inside", "median"))
    if "pq" in R:
        agg.update(gt_cells=("gt_cells", "mean"), gt_area=("gt_area", "median"), pq=("pq", "mean"))
    T = R.groupby(["config", "mod", "split", "mouse", "cp"]).agg(**agg).reset_index()
    for (cfg, mod), t in T.groupby(["config", "mod"]):
        print(f"\n=== {cfg} {mod}: per mouse and threshold (cells per region, median cell area px, median cell prob inside cells) ===")
        print(t.drop(columns=["config", "mod"]).round(3).to_string(index=False))
    print("\nsaved", args.out)
