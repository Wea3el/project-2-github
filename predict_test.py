"""Build the Kaggle submission from the cached test-set flows of the chosen models.

usage: python predict_test.py [--best runs/best_config.json] [--iv-config X --cp-iv -1 --ex-config Y --cp-ex 0 --thr 0.1]
       [--out submission.csv]
Flows are read from runs/<config>/<mod>/full/flows (written by infer_flows_hpc.py --fold full)."""
import os, json, argparse, pickle
import numpy as np, pandas as pd

from common import DATA, RUNS, ROOT, load_images, masks_from_flows, load_flows, run_dir
from cm_pipeline import match_regions, to_rows

ap = argparse.ArgumentParser()
ap.add_argument("--best", default=os.path.join(RUNS, "best_config.json"))
ap.add_argument("--iv-config"); ap.add_argument("--cp-iv", type=float)
ap.add_argument("--ex-config"); ap.add_argument("--cp-ex", type=float)
ap.add_argument("--thr", type=float)
ap.add_argument("--iv-fold", default="full"); ap.add_argument("--ex-fold", default="full")
ap.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
args = ap.parse_args()
b = json.load(open(args.best)) if os.path.exists(args.best) else {}
ivc = args.iv_config or b["iv_config"]; cpi = args.cp_iv if args.cp_iv is not None else b["cp_iv"]
exc = args.ex_config or b["ex_config"]; cpe = args.cp_ex if args.cp_ex is not None else b["cp_ex"]
thr = args.thr if args.thr is not None else b.get("thr", 0.1)
print(f"in-vivo: {ivc} cp={cpi} | ex-vivo: {exc} cp={cpe} | pair thr={thr}", flush=True)

try:
    clf = pickle.load(open(os.path.join(ROOT, "weights", "pair_clf.pkl"), "rb"))
except Exception as e:  # scikit-learn version mismatch -> refit from the shipped features
    print("refitting pair classifier (", e, ")")
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = pickle.load(open(os.path.join(ROOT, "weights", "pairs_gt.pkl"), "rb"))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0).fit(
        np.concatenate([r["F"] for r in rows]), np.concatenate([r["y"] for r in rows]))

sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
items = []
for sid in sample.sample_id:
    iv, ex = load_images(sid, "hidden_test")
    dP, cp, up = load_flows(run_dir(ivc, "iv", args.iv_fold), sid); liv = masks_from_flows(dP, cp, cpi, up)
    dP, cp, up = load_flows(run_dir(exc, "ex", args.ex_fold), sid); lex = masks_from_flows(dP, cp, cpe, up)
    items.append(dict(sid=sid, iv_img=iv, ex_img=ex, liv=liv, lex=lex))
    print(sid, int(liv.max()), int(lex.max()), flush=True)
pairs, log = match_regions(items, lambda s: clf, thrs=(thr,), verbose=True)
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
