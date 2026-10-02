"""Where do the points go? For one setting, follow every ground-truth pair of every training region
through the pipeline, using the models trained without that region's mouse (CPU only):
  segmented (both cells found at IoU > 0.75) -> proposed as a candidate pair (registration + radius)
  -> accepted by the pair classifier;
and sort every predicted pair into correct / wrong in-vivo cell / wrong ex-vivo cell / wrong partner.

usage: python diagnose_cv.py --iv cyto3_x3:-1 --ex cyto3_x3:0 --thr 0.1 [--pairs gt] [--out runs/diag_v6.csv]
       python diagnose_cv.py --iv cyto3_x3_auto:-1 --ex cpsam2_x3:-1 --thr 0.05 --pairs oof+flow_ex=0.3 --out runs/diag_f3.csv
  stages per verified pair: in-vivo / ex-vivo / both cells segmented (IoU > 0.75) -> the right pair among the
  candidates -> accepted by the classifier; --pairs takes any pipeline variant, e.g. oof+cand=gain+u=8
"""
import os, argparse
import numpy as np, pandas as pd

from common import RUNS, gt_labels, training_ids, load_images, load_flows, run_dir
from configs import SUBJECTS
from cm_pipeline import match_regions, tp_map, parse_method, region_item, match_kw
from evaluate_cv import fit_clf

ap = argparse.ArgumentParser()
ap.add_argument("--iv", required=True, help="config:cellprob")
ap.add_argument("--ex", required=True, help="config:cellprob")
ap.add_argument("--thr", type=float, default=0.1)
ap.add_argument("--pairs", default="gt", help="pipeline variant, e.g. oof or oof+weak=keep")
ap.add_argument("--out", default=os.path.join(RUNS, "diag.csv"))
args = ap.parse_args()
(ivc, ivcp), (exc, excp) = [(x.split(":")[0], float(x.split(":")[1])) for x in (args.iv, args.ex)]
pairs_data, opt = parse_method(args.pairs)

rows = []
for fold in SUBJECTS:
    items, gts = [], {}
    for sid in training_ids(subjects=[fold]):
        iv, ex = load_images(sid, "training")
        items.append(region_item(sid, iv, ex, load_flows(run_dir(ivc, "iv", fold), sid), load_flows(run_dir(exc, "ex", fold), sid),
                                 ivcp, excp, opt)); gts[sid] = gt_labels(sid)
    clf, feats = fit_clf(pairs_data, fold), {}
    pairs, log = match_regions(items, lambda s: clf, thrs=(args.thr,), feats=feats, **match_kw(opt))
    status = dict(zip(log.sid, log.status))
    for it in items:
        sid = it["sid"]; giv, gex, gp = gts[sid]
        miv, pq_iv = tp_map(it["liv"], giv); mex, pq_ex = tp_map(it["lex"], gex)
        g2p_iv = {g: p for p, (g, _) in miv.items()}; g2p_ex = {g: p for p, (g, _) in mex.items()}
        gset = {(int(a), int(b)) for a, b in gp}
        la, lb, _ = feats.get(sid, ([], [], None))
        cand = set(zip(map(int, la), map(int, lb)))
        pred = [(int(a), int(b)) for a, b in pairs[args.thr][sid]]
        found = [(g2p_iv[a], g2p_ex[b]) for a, b in gset if a in g2p_iv and b in g2p_ex]
        kinds = [("wrong_iv" if a not in miv else "wrong_ex" if b not in mex else
                  "correct" if (miv[a][0], mex[b][0]) in gset else "wrong_partner") for a, b in pred]
        rows.append(dict(
            mouse=fold.replace("subject_", ""), sid=sid, status=status.get(sid, "none"),
            gt_iv_cells=len(np.unique(giv)) - 1, gt_ex_cells=len(np.unique(gex)) - 1,
            pred_iv_cells=len(np.unique(it["liv"])) - 1, pred_ex_cells=len(np.unique(it["lex"])) - 1,
            pq_iv=pq_iv, pq_ex=pq_ex, gt_pairs=len(gset),
            iv_found=sum(a in g2p_iv for a, _ in gset), ex_found=sum(b in g2p_ex for _, b in gset),
            both_found=len(found), proposed=sum(p in cand for p in found), accepted=sum(p in set(pred) for p in found),
            candidates=len(cand), pred_pairs=len(pred), **{k: kinds.count(k) for k in ("correct", "wrong_iv", "wrong_ex", "wrong_partner")}))
        r = rows[-1]
        print(f"{sid:32s} {r['status']:6s} gt {r['gt_pairs']:3d} found {r['both_found']:3d} proposed {r['proposed']:3d} "
              f"accepted {r['accepted']:3d} | predicted {r['pred_pairs']:3d} correct {r['correct']:3d}", flush=True)

D = pd.DataFrame(rows)
D.to_csv(args.out, index=False)
S = D.groupby("mouse")[["gt_pairs", "iv_found", "ex_found", "both_found", "proposed", "accepted", "pred_pairs", "correct",
                        "wrong_iv", "wrong_ex", "wrong_partner"]].sum()
S.loc["all"] = S.sum()
# matching F1 now, and its ceilings: perfect pairing of the verified pairs whose two cells are both segmented
# (2B / (B + G)), and perfect classification of the candidates that contain them
S["f1"] = 2 * S.correct / (S.pred_pairs + S.gt_pairs).clip(lower=1)
S["f1_max_masks"] = 2 * S.both_found / (S.both_found + S.gt_pairs).clip(lower=1)
S["f1_max_candidates"] = 2 * S.proposed / (S.proposed + S.gt_pairs).clip(lower=1)
pd.set_option("display.width", 250)
print("\n", S.round(3).to_string())
print("\nlost before both_found -> segmentation | both_found -> proposed -> registration / candidates | "
      "proposed -> accepted -> classifier / threshold")
print("saved", args.out)
