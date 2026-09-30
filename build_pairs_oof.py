"""Pair-classifier training data from OUT-OF-FOLD predicted masks. The shipped classifier
(weights/pairs_gt.pkl) was fitted on candidate pairs built from ground-truth masks, but on the test set
it scores pairs built from predicted masks. Here each training region is segmented by the models trained
without its mouse, registered and paired exactly as at test time, and a candidate pair is labelled
correct with the competition's rule (both cells matched to ground truth at IoU > 0.75, and those two
ground-truth cells form a verified pair).

usage: python build_pairs_oof.py --iv cyto3_x2:-0.5 --ex cyto3_x3:0 [--out weights/pairs_oof.pkl]
Then compare classifiers with evaluate_cv.py --pairs gt,oof,both and use one with predict_test.py --pairs."""
import os, argparse, pickle
import numpy as np

from common import ROOT, gt_labels, training_ids, load_images, masks_from_flows, load_flows, run_dir
from configs import SUBJECTS
from cm_pipeline import match_regions, tp_map
from evaluate_cv import fit_clf

ap = argparse.ArgumentParser()
ap.add_argument("--iv", required=True, help="config:cellprob")
ap.add_argument("--ex", required=True, help="config:cellprob")
ap.add_argument("--out", default=os.path.join(ROOT, "weights", "pairs_oof.pkl"))
args = ap.parse_args()
(ivc, ivcp), (exc, excp) = [(x.split(":")[0], float(x.split(":")[1])) for x in (args.iv, args.ex)]

rows = []
for fold in SUBJECTS:
    items, gts = [], {}
    for sid in training_ids(subjects=[fold]):
        iv, ex = load_images(sid, "training")
        dP, cp, up = load_flows(run_dir(ivc, "iv", fold), sid); liv = masks_from_flows(dP, cp, ivcp, up)
        dP, cp, up = load_flows(run_dir(exc, "ex", fold), sid); lex = masks_from_flows(dP, cp, excp, up)
        items.append(dict(sid=sid, iv_img=iv, ex_img=ex, liv=liv, lex=lex)); gts[sid] = gt_labels(sid)
    feats = {}
    clf = fit_clf("gt", fold)  # only needed to run the pipeline; its scores are not used here
    match_regions(items, lambda s: clf, feats=feats)
    for it in items:
        sid = it["sid"]; giv, gex, gp = gts[sid]
        la, lb, F = feats.get(sid, (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 14))))
        miv, _ = tp_map(it["liv"], giv); mex, _ = tp_map(it["lex"], gex)
        gset = {(int(a), int(b)) for a, b in gp}
        y = np.array([(miv[a][0] if a in miv else -1, mex[b][0] if b in mex else -2) in gset for a, b in zip(la, lb)], bool)
        rows.append(dict(sid=sid, F=np.asarray(F, np.float64).reshape(-1, 14), y=y, ngt=len(gset)))
        print(sid, len(y), int(y.sum()), "of", len(gset), flush=True)
pickle.dump(rows, open(args.out, "wb"))
print("saved", args.out, sum(len(r["y"]) for r in rows), "candidate pairs,", sum(int(r["y"].sum()) for r in rows), "correct")
