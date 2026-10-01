"""Train (or fine-tune) a Cellpose network for one modality. Resumable: a checkpoint is written
every few epochs, so a preempted/requeued SLURM job continues where it stopped.

usage: python train_seg_hpc.py --mod ex --config cyto3_x3 --fold subject_db6b8b
       (--fold = held-out subject for cross-validation, or 'full' to train on all mice)
"""
import os, sys, json, time, argparse
import numpy as np
import torch

from common import DATA, gt_labels, training_ids, load_images, run_dir, load_flows, masks_from_flows
from configs import CONFIGS
from segdata import norm_img, make_tiles, median_diameter

ap = argparse.ArgumentParser()
ap.add_argument("--mod", required=True, choices=["iv", "ex"])
ap.add_argument("--config", required=True)
ap.add_argument("--fold", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt-every", type=int, default=10)
ap.add_argument("--epochs", type=int, default=None, help="override (for quick tests)")
ap.add_argument("--max-regions", type=int, default=None, help="for quick tests")
args = ap.parse_args()

for suffix in ("_tta", "_auto"):   # inference-only variants: the model is the base config's
    if args.config.endswith(suffix):
        args.config = args.config[:-len(suffix)]
cfg = dict(CONFIGS[args.config])
args.seed = cfg.get("seed", args.seed)
if args.epochs:
    cfg["epochs"] = args.epochs
out = run_dir(args.config, args.mod, args.fold)
os.makedirs(out, exist_ok=True)
final_path = os.path.join(out, "model")
if os.path.exists(os.path.join(out, "train_done.json")):
    print("already trained:", final_path); sys.exit(0)

from cellpose import models, dynamics, transforms, train as cptrain

sam = str(cfg.get("pretrained")).startswith("cpsam")   # Cellpose-SAM: needs cellpose 4 (the SAM overlay)
np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, "| config", args.config, cfg, flush=True)

# ------------------------------------------------------------------ data: native-resolution tiles
holdout = set() if args.fold == "full" else {args.fold}
ids = training_ids(exclude=holdout)
if args.max_regions:
    ids = ids[:args.max_regions]
rng = np.random.default_rng(0)
tiles, tlabs, diams, seen = [], [], [], set()
for sid in ids:
    iv, ex = load_images(sid, "training")
    liv, lex, _ = gt_labels(sid)
    img, lab = (iv, liv) if args.mod == "iv" else (ex, lex)
    key = hash(img.tobytes())
    if key in seen:  # some regions share an identical image
        continue
    seen.add(key)
    ti, tl = make_tiles(norm_img(img), lab, tile=128, step=112, rng=rng)
    tiles += ti; tlabs += tl
    diams.append(median_diameter(lab))
print(f"{len(ids)} regions -> {len(tiles)} tiles, median diameter {np.median(diams):.2f}px", flush=True)
if cfg.get("pseudo"):  # self-training: the unlabelled test images, labelled with an earlier model's masks
    import pandas as pd
    pc, pcp = cfg["pseudo"][args.mod].rsplit(":", 1)
    for sid in pd.read_csv(os.path.join(DATA, "sample_submission.csv")).sample_id:
        img = load_images(sid, "hidden_test")[0 if args.mod == "iv" else 1]
        key = hash(img.tobytes())
        if key in seen:
            continue
        seen.add(key)
        dP, prob, pup = load_flows(run_dir(pc, args.mod, "full"), sid)
        ti, tl = make_tiles(norm_img(img), masks_from_flows(dP, prob, float(pcp), pup), tile=128, step=112, rng=rng)
        tiles += ti; tlabs += tl
    print(f"+ test images labelled by {pc} at cellprob {pcp} -> {len(tiles)} tiles", flush=True)
# flow targets are computed on the CPU: Cellpose 3.1.1.3's GPU version crashes on tiles that contain a
# single foreground pixel (np.stack on a 0-d array in _extend_centers_gpu); the CPU version is fine
flows = dynamics.labels_to_flows(tlabs, device=torch.device("cpu"))  # each (4,H,W): label, mask, flowY, flowX
# image + empty channels: 2 for Cellpose 3, 3 for Cellpose-SAM (as cellpose 4's own train_seg pads)
X = [np.stack([t] + [np.zeros_like(t)] * (2 if sam else 1)).astype(np.float32) for t in tiles]
Y = [f[1:] for f in flows]                                          # mask, flowY, flowX
nimg = len(X)

# ------------------------------------------------------------------ network
up = float(cfg["up"])
if sam:
    torch.backends.cuda.matmul.allow_tf32 = True   # float32 training (like train_seg), TF32 matmuls for speed
    net = models.CellposeModel(device=device, pretrained_model=cfg["pretrained"], use_bfloat16=False).net
elif cfg.get("pretrained"):
    pre = os.environ.get("CM_CYTO3", "") if cfg["pretrained"] == "cyto3" else ""
    pre = pre if pre and os.path.exists(pre) else cfg["pretrained"]
    kw = dict(pretrained_model=pre) if os.path.exists(str(pre)) else dict(model_type=pre)
    base = models.CellposeModel(gpu=device.type == "cuda", device=device, **kw)
    net = base.net
else:
    from cellpose.resnet_torch import CPnet
    net = CPnet([2, *cfg["nbase"]], 3, sz=3, mkldnn=False, max_pool=True, diam_mean=float(up * 10.0)).to(device)
net.diam_labels.data = torch.Tensor([np.median(diams) * up]).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-5))

# learning-rate schedule identical to cellpose.train.train_seg
E, lr = cfg["epochs"], cfg["lr"]
LR = np.linspace(0, lr, 10)
LR = np.append(LR, lr * np.ones(max(0, E - 10)))
if E > 300:
    LR = LR[:-100]
    for i in range(10):
        LR = np.append(LR, LR[-1] / 2 * np.ones(10))
elif E > 99:
    LR = LR[:-50]
    for i in range(10):
        LR = np.append(LR, LR[-1] / 2 * np.ones(5))

ckpt = os.path.join(out, "ckpt.pt")
start = 0
if os.path.exists(ckpt):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); start = st["epoch"] + 1
    print("resumed from epoch", start, flush=True)

rescale = np.full(cfg["batch"], 1.0 / up, np.float32)  # random_rotate_and_resize scales by 1/rescale = up
t0 = time.time()
for ep in range(start, E):
    np.random.seed(ep + 1000 * args.seed)
    perm = np.random.choice(nimg, size=cfg["nimg"], replace=nimg < cfg["nimg"])
    for g in opt.param_groups:
        g["lr"] = LR[ep]
    net.train()
    tot, n = 0.0, 0
    for k in range(0, cfg["nimg"], cfg["batch"]):
        inds = perm[k:k + cfg["batch"]]
        if sam:  # cellpose 4's augmentation runs on the GPU and returns tensors
            imgi, lbl = transforms.random_rotate_and_resize([X[i] for i in inds], lbls=[Y[i] for i in inds],
                                                            rescale=rescale[:len(inds)], scale_range=cfg.get("sr", 0.5),
                                                            bsize=cfg["bsize"], device=device)[:2]
            if cfg.get("aug"):  # same brightness/contrast jitter + noise as below, on the GPU tensors
                u = lambda lo, hi: torch.empty((len(inds), 1, 1), device=device).uniform_(lo, hi)
                imgi[:, 0] = imgi[:, 0] * u(0.7, 1.4) + u(-0.15, 0.15) + torch.randn_like(imgi[:, 0]) * u(0, 0.05)
            y = net(imgi)[0]
        else:
            imgi, lbl = transforms.random_rotate_and_resize([X[i] for i in inds], Y=[Y[i] for i in inds],
                                                            rescale=rescale[:len(inds)], scale_range=cfg.get("sr", 0.5),
                                                            xy=(cfg["bsize"], cfg["bsize"]))[:2]
            if cfg.get("aug"):  # per-crop brightness/contrast jitter + noise on the image channel
                k = len(inds)
                imgi[:, 0] = (imgi[:, 0] * np.random.uniform(0.7, 1.4, (k, 1, 1)) + np.random.uniform(-0.15, 0.15, (k, 1, 1))
                              + np.random.normal(0, 1, imgi[:, 0].shape) * np.random.uniform(0, 0.05, (k, 1, 1))).astype(np.float32)
            y = net(torch.from_numpy(imgi).to(device))[0]
        loss = cptrain._loss_fn_seg(lbl, y, device)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(inds); n += len(inds)
    if ep % 10 == 0 or ep == E - 1:
        print(f"epoch {ep}  loss {tot / n:.4f}  lr {LR[ep]:.3g}  {time.time() - t0:.0f}s", flush=True)
    if (ep + 1) % args.ckpt_every == 0 or ep == E - 1:
        tmp = ckpt + ".tmp"
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), epoch=ep), tmp)
        os.replace(tmp, ckpt)

net.save_model(final_path)
json.dump(dict(config=args.config, cfg=cfg, fold=args.fold, mod=args.mod, n_tiles=nimg, regions=len(ids),
               median_diam=float(np.median(diams)), minutes=(time.time() - t0) / 60),
          open(os.path.join(out, "train_done.json"), "w"), indent=1)
if os.path.exists(ckpt):
    os.remove(ckpt)  # only needed to resume; 3.6 GB for Cellpose-SAM
print("saved", final_path, flush=True)
