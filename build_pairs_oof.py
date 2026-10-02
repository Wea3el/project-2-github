"""Pair-classifier training data from OUT-OF-FOLD predicted masks. The shipped classifier
(weights/pairs_gt.pkl) was fitted on candidate pairs built from ground-truth masks, but on the test set
it scores pairs built from predicted masks. Here each training region is segmented by the models trained
without its mouse, registered and paired exactly as at test time, and a candidate pair is labelled
correct with the competition's rule (both cells matched to ground truth at IoU > 0.75, and those two
ground-truth cells form a verified pair).

usage: python build_pairs_oof.py --iv cyto3_x2:-0.5 --ex cyto3_x3:0 [--method oof+cand=all] [--strict] [--out ...]
       python build_pairs_oof.py --iv cyto3_x3_auto:-1 --ex cpsam2_x3:-1 --out weights/pairs_oofs1.pkl   # for s1's masks
  --method  pipeline options for the masks and the candidates (cm_pipeline.parse_method; the name before the first
            "+" is ignored), e.g. oof+cand=all for the classifier of the all-candidates assignment
  --strict  nested: when mouse A is held out, the regions of mouse B are segmented by the model trained on the
            third mouse only (runs/<config>/<mod>/only_<mouse>, see README 6I), so no classifier row comes from a
            model that saw A. Rows carry outer=A, and evaluate_cv.fit_clf uses the rows of the mouse it evaluates.
  default output: weights/pairs_oof[strict][all].pkl, the name to give --pairs (e.g. oofstrict, oofall)
"""
import os, argparse, pickle
import numpy as np

from common import ROOT, gt_labels, training_ids, load_images, load_flows, run_dir
from configs import SUBJECTS
from cm_pipeline import match_regions, tp_map, parse_method, region_item, match_kw
from evaluate_cv import fit_clf

ap = argparse.ArgumentParser()
ap.add_argument("--iv", required=True, help="config:cellprob")
ap.add_argument("--ex", required=True, help="config:cellprob")
ap.add_argument("--method", default="oof")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()
(ivc, ivcp), (exc, excp) = [(x.rsplit(":", 1)[0], float(x.rsplit(":", 1)[1])) for x in (args.iv, args.ex)]
_, opt = parse_method(args.method)
name = "oof" + ("strict" if args.strict else "") + ("all" if opt["cand"] == "all" else "")
out_fn = args.out or os.path.join(ROOT, "weights", f"pairs_{name}.pkl")
if os.path.exists(out_fn):  # submitted files were made with it: keep them reproducible
    raise SystemExit(f"{out_fn} already exists: pair-classifier data is never overwritten, give a new --out")

# (outer held-out mouse or None, mouse whose regions are labelled, run folder of the models that segment it)
if args.strict:
    jobs = [(A, B, "only_" + [m for m in SUBJECTS if m not in (A, B)][0]) for A in SUBJECTS for B in SUBJECTS if B != A]
else:
    jobs = [(None, B, B) for B in SUBJECTS]

rows = []
for outer, mouse, rf in jobs:
    items, gts = [], {}
    for sid in training_ids(subjects=[mouse]):
        iv, ex = load_images(sid, "training")
        items.append(region_item(sid, iv, ex, load_flows(run_dir(ivc, "iv", rf), sid), load_flows(run_dir(exc, "ex", rf), sid),
                                 ivcp, excp, opt)); gts[sid] = gt_labels(sid)
    feats = {}
    clf = fit_clf("gt", mouse)  # only needed to run the pipeline; its scores are not used here
    match_regions(items, lambda s: clf, feats=feats, **match_kw(opt))
    for it in items:
        sid = it["sid"]; giv, gex, gp = gts[sid]
        la, lb, F = feats.get(sid, (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 14 + 4 * opt["q"]))))
        miv, _ = tp_map(it["liv"], giv); mex, _ = tp_map(it["lex"], gex)
        gset = {(int(a), int(b)) for a, b in gp}
        y = np.array([(miv[a][0] if a in miv else -1, mex[b][0] if b in mex else -2) in gset for a, b in zip(la, lb)], bool)
        r = dict(sid=sid, F=np.asarray(F, np.float64), y=y, ngt=len(gset))
        if outer:
            r["outer"] = outer
        rows.append(r)
        print(outer or "-", sid, len(y), int(y.sum()), "of", len(gset), flush=True)
pickle.dump(rows, open(out_fn, "wb"))
print("saved", out_fn, sum(len(r["y"]) for r in rows), "candidate pairs,", sum(int(r["y"].sum()) for r in rows), "correct")
