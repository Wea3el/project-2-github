"""Run a trained model and cache its raw outputs (flows + cell probability) per region, so mask
thresholds can be tuned later on CPU without the GPU.

usage: python infer_flows_hpc.py --mod ex --config cyto3_x3 --fold subject_db6b8b          # held-out mouse
       python infer_flows_hpc.py --mod ex --config cyto3_x3 --fold full --split hidden_test # test set
"""
import os, argparse, time
import numpy as np, pandas as pd, torch

from common import DATA, training_ids, load_images, run_dir, flows_path
from configs import CONFIGS
from seg_infer import load_cp, segment

ap = argparse.ArgumentParser()
ap.add_argument("--mod", required=True, choices=["iv", "ex"])
ap.add_argument("--config", required=True)
ap.add_argument("--fold", required=True)
ap.add_argument("--split", default=None, help="training (default for CV folds) or hidden_test")
ap.add_argument("--model", default=None, help="explicit weights path (default: run_dir/model)")
ap.add_argument("--max-regions", type=int, default=None)
args = ap.parse_args()

rd = run_dir(args.config, args.mod, args.fold)
split = args.split or ("hidden_test" if args.fold == "full" else "training")
up = float(CONFIGS[args.config]["up"])
if split == "training":
    ids = training_ids(subjects=[args.fold]) if args.fold != "full" else training_ids()
else:
    ids = list(pd.read_csv(os.path.join(DATA, "sample_submission.csv")).sample_id)
if args.max_regions:
    ids = ids[:args.max_regions]
os.makedirs(os.path.join(rd, "flows"), exist_ok=True)
model = load_cp(args.model or os.path.join(rd, "model"), up=up)
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
t0 = time.time()
for sid in ids:
    fn = flows_path(rd, sid)
    if os.path.exists(fn):
        continue
    iv, ex = load_images(sid, split)
    img = iv if args.mod == "iv" else ex
    _, fl = segment(model, img, up=up, cellprob=0.0, return_flows=True)
    tmp = fn + f".{os.getpid()}.tmp.npz"
    # float32 for final (test-set) outputs so the submission is reproduced exactly; float16 is plenty
    # for cross-validation folds and halves the disk space
    dt = np.float32 if args.fold == "full" else np.float16
    np.savez_compressed(tmp, dP=fl[1].astype(dt), cellprob=fl[2].astype(dt), up=up)
    os.replace(tmp, fn)
    print(sid, f"{time.time() - t0:.0f}s", flush=True)
open(os.path.join(rd, "flows", f"_done_{split}"), "w").write(str(len(ids)))
print("done", len(ids), "regions ->", os.path.join(rd, "flows"))
