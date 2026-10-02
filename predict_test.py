"""Build the Kaggle submission from the cached test-set flows of the chosen models.

usage: python predict_test.py [--best runs/best_config.json] [--iv-config X --cp-iv -1 --ex-config Y --cp-ex 0 --thr 0.1]
       [--out submission.csv]
Flows are read from runs/<config>/<mod>/full/flows (written by infer_flows_hpc.py --fold full)."""
import os, json, argparse, pickle
import numpy as np, pandas as pd

from common import DATA, RUNS, ROOT, load_images, load_flows, run_dir, new_submission, log_submission
from cm_pipeline import match_regions, to_rows

ap = argparse.ArgumentParser()
ap.add_argument("--best", default=os.path.join(RUNS, "best_config.json"))
ap.add_argument("--iv-config"); ap.add_argument("--cp-iv", type=float)
ap.add_argument("--ex-config"); ap.add_argument("--cp-ex", type=float)
ap.add_argument("--thr", type=float)
ap.add_argument("--pairs", help="pair classifier training data: gt (shipped), oof or both (see evaluate_cv.fit_clf)")
ap.add_argument("--iv-fold", default="full"); ap.add_argument("--ex-fold", default="full")
ap.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
args = ap.parse_args()
new_submission(args.out)
b = json.load(open(args.best)) if os.path.exists(args.best) else {}
ivc = args.iv_config or b["iv_config"]; cpi = args.cp_iv if args.cp_iv is not None else b["cp_iv"]
exc = args.ex_config or b["ex_config"]; cpe = args.cp_ex if args.cp_ex is not None else b["cp_ex"]
thr = args.thr if args.thr is not None else b.get("thr", 0.1)
spec = args.pairs or b.get("pairs", "gt")
print(f"in-vivo: {ivc} cp={cpi} | ex-vivo: {exc} cp={cpe} | pair thr={thr} | pair classifier: {spec}", flush=True)

from cm_pipeline import parse_method, region_item, match_kw
pairs, opt = parse_method(spec)
pairs_name = pairs
clf = None
if pairs == "gt":
    try:
        clf = pickle.load(open(os.path.join(ROOT, "weights", "pair_clf.pkl"), "rb"))
    except Exception as e:  # scikit-learn version mismatch -> refit from the shipped features below
        print("refitting pair classifier (", e, ")")
if clf is None:
    from evaluate_cv import fit_clf
    clf = fit_clf(pairs, fold="none")   # all training mice

sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
items = []
for sid in sample.sample_id:
    iv, ex = load_images(sid, "hidden_test")
    items.append(region_item(sid, iv, ex, load_flows(run_dir(ivc, "iv", args.iv_fold), sid),
                             load_flows(run_dir(exc, "ex", args.ex_fold), sid), cpi, cpe, opt))
    print(sid, int(items[-1]["liv"].max()), int(items[-1]["lex"].max()), flush=True)
pairs, log = match_regions(items, lambda s: clf, thrs=(thr,), verbose=True, **match_kw(opt))
sub = pd.DataFrame([to_rows(it["sid"], it["liv"], it["lex"], pairs[thr][it["sid"]]) for it in items])

# format checks (same rules as the competition)
assert list(sub.sample_id) == list(sample.sample_id)
for _, r in sub.iterrows():
    iv_ids, ex_ids = set(json.loads(r.invivo_instances)), set(json.loads(r.exvivo_instances))
    pr = json.loads(r.match_pairs); a = [p[0] for p in pr]; bb = [p[1] for p in pr]
    assert len(set(a)) == len(a) and len(set(bb)) == len(bb) and set(a) <= iv_ids and set(bb) <= ex_ids
sub.to_csv(args.out, index=False)
pd.set_option("display.width", 200)
print(log.round(2).to_string(index=False))
print("pairs:", int(log.n_pairs.sum()), "| wrote", args.out, sub.shape)
from evaluate_cv import PAIR_FILES
log_submission(args.out, f"iv {ivc}:{cpi:g} | ex {exc}:{cpe:g} | thr {thr:g} | {spec}", int(log.n_pairs.sum()),
               [(ivc, "iv", args.iv_fold), (exc, "ex", args.ex_fold)],
               [PAIR_FILES.get(k, f"pairs_{k}.pkl") for k in (["gt", "oof"] if pairs_name == "both" else [pairs_name])])
