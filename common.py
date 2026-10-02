"""Paths, ground-truth label cache and flow->mask conversion shared by all scripts."""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CM_DATA", os.path.join(ROOT, "data"))
os.environ["CM_DATA"] = DATA  # cmutil reads this at import
RUNS = os.environ.get("CM_RUNS", os.path.join(ROOT, "runs"))
CACHE = os.environ.get("CM_CACHE", os.path.join(ROOT, "cache"))

from cmutil import load_gt, load_images, instances_to_label  # noqa: E402


def gt_labels(sid):
    """Ground-truth label images (+ ids and verified pairs as label pairs), cached as npz."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{sid}_gt.npz")
    if not os.path.exists(p):
        gt = load_gt().set_index("sample_id").loc[sid]
        iv, ex = load_images(sid, "training")
        liv, ids_iv = instances_to_label(json.loads(gt.invivo_instances), iv.shape)
        lex, ids_ex = instances_to_label(json.loads(gt.exvivo_instances), ex.shape)
        piv = {c: k + 1 for k, c in enumerate(ids_iv)}; pex = {c: k + 1 for k, c in enumerate(ids_ex)}
        pairs = np.array([(piv[a], pex[b]) for a, b in json.loads(gt.match_pairs)], dtype=np.int64).reshape(-1, 2)
        tmp = p + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, liv=liv, lex=lex, pairs=pairs)
        os.replace(tmp, p)
    z = np.load(p)
    return z["liv"], z["lex"], z["pairs"]


def training_ids(subjects=None, exclude=None):
    ids = list(load_gt().sample_id)
    if subjects is not None:
        ids = [s for s in ids if s.split("__")[0] in subjects]
    if exclude is not None:
        ids = [s for s in ids if s.split("__")[0] not in exclude]
    lim = int(os.environ.get("CM_MAX_REGIONS", "0"))  # smoke tests only: first N regions per mouse
    if lim:
        per = {}
        ids = [s for s in ids if per.setdefault(s.split("__")[0], []).append(s) or len(per[s.split("__")[0]]) <= lim]
    return ids


def masks_from_flows(dP, cellprob, cp_thr, up=1.0, flow_threshold=0.4, min_size=15):
    """Same mask reconstruction Cellpose's eval() performs (niter scaled by 1/rescale)."""
    import inspect
    from cellpose import dynamics
    kw = dict(niter=int(round(200 / up)), cellprob_threshold=cp_thr, flow_threshold=flow_threshold, resize=None,
              min_size=min_size, max_size_fraction=0.4)
    if "interp" in inspect.signature(dynamics.resize_and_compute_masks).parameters:  # cellpose 3 (4 always interpolates)
        kw["interp"] = True
    m = dynamics.resize_and_compute_masks(np.asarray(dP, np.float32), np.asarray(cellprob, np.float32), **kw)
    return np.asarray(m).astype(np.int32)


def flows_path(run_dir, sid):
    return os.path.join(run_dir, "flows", f"{sid}.npz")


def members(rd):
    """Run dirs behind rd. A config named "a+b" is the ensemble of configs a and b (their flows are averaged)."""
    fold, mod = os.path.basename(rd), os.path.basename(os.path.dirname(rd))
    root, cfg = os.path.split(os.path.dirname(os.path.dirname(rd)))
    return [os.path.join(root, c, mod, fold) for c in cfg.split("+")]


def has_flows(rd, sid):
    return all(os.path.exists(flows_path(m, sid)) for m in members(rd))


def load_flows(run_dir, sid):
    fl = []
    for m in members(run_dir):
        z = np.load(flows_path(m, sid))
        fl.append((z["dP"].astype(np.float32), z["cellprob"].astype(np.float32), float(z["up"])))
    assert len({f[2] for f in fl}) == 1, f"ensemble members of {run_dir} use different upsampling"
    return sum(f[0] for f in fl) / len(fl), sum(f[1] for f in fl) / len(fl), fl[0][2]


def run_dir(config, mod, fold):
    """fold = held-out subject for cross-validation, or 'full' (trained on all mice)."""
    return os.path.join(RUNS, config, mod, fold)


# ---------------------------------------------------------------- submission files: never overwritten, always recorded
def new_submission(out):
    """submitted files must stay reproducible: stop before any work if the name is already taken"""
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: submission files are never overwritten, pick a new name")


def _sha(fn, n=10):
    import hashlib
    if not os.path.exists(fn):
        return "missing"
    h = hashlib.sha1()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:n]


def _git_commit():
    """commit checked out in ROOT, read from .git directly (git is not installed in the container)"""
    g = os.path.join(ROOT, ".git")
    try:
        head = open(os.path.join(g, "HEAD")).read().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        if os.path.exists(os.path.join(g, ref)):
            return open(os.path.join(g, ref)).read().strip()[:7]
        return next(l.split()[0][:7] for l in open(os.path.join(g, "packed-refs")) if l.rstrip().endswith(" " + ref))
    except Exception:
        return "unknown"


def log_submission(out, settings, n_pairs, models, pair_files):
    """append what a submission contains to submissions_log.tsv (see SUBMISSIONS.md): date, file, settings (ksub.sh sends
    them to Kaggle as the description), pairs, code (git commit + hash of the .py files, so uncommitted edits show), and
    sha1 of the model weights and of the pair-classifier data. models: (config, mod, fold); pair_files: names in weights/"""
    import glob, hashlib, time
    code = hashlib.sha1(b"".join(open(f, "rb").read() for f in sorted(glob.glob(os.path.join(ROOT, "*.py"))))).hexdigest()[:8]
    hashes = []
    for cfg, mod, fold in models:
        for c in cfg.split("+"):   # ensembles; "_auto" / "_tta" runs use their base config's weights
            base = c[:-5] if c.endswith("_auto") else c[:-4] if c.endswith("_tta") else c
            hashes.append(f"{base}/{mod}/{fold}={_sha(os.path.join(run_dir(base, mod, fold), 'model'))}")
    hashes += [f"{p}={_sha(os.path.join(ROOT, 'weights', p))}" for p in pair_files]
    with open(os.path.join(ROOT, "submissions_log.tsv"), "a") as f:
        f.write("\t".join([time.strftime("%Y-%m-%d %H:%M"), os.path.basename(out), settings, f"{n_pairs} pairs",
                           f"git {_git_commit()} code {code}", " ".join(hashes)]) + "\n")
