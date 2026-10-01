"""Pictures of the final cell matching, one per region, three panels side by side:
  1. the in-vivo image with every predicted cell outlined and the matched cells filled,
  2. the ex-vivo image resampled into the in-vivo frame (similarity fitted to the pairs), outlined and
     filled the same way: a matched cell sits in the same place, in the same colour, as its partner,
  3. both images overlaid (in-vivo green, ex-vivo magenta): where the registration is right the cells
     look white, where it is wrong they stay green and magenta.

  test: python render_pairs.py test submission_v13.csv [--out viz/submission_v13]
        each matched pair gets its own colour
  cv:   python render_pairs.py cv --iv cyto3_x3:-1 --ex cyto3_x3:0 --pairs oof --thr 0.05 [--out viz/cv_v13]
        held-out training regions (models trained without that mouse) checked against the ground truth,
        aligned with the true pairs: green = correct pair, red = wrong pair, yellow outline = a verified
        pair that was missed
Writes <out>/<region>.jpg (full size) and <out>/index.html (every region in one page, opens in a browser).
"""
import os, io, json, base64, colorsys, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from skimage.transform import SimilarityTransform, warp
from skimage.measure import ransac

from common import DATA, load_images
from cmutil import instances_to_label

GREEN, RED, YELLOW = (60, 220, 90), (240, 60, 60), (255, 210, 0)


def norm8(im):
    lo, hi = np.percentile(im, [1, 99.7])
    return (np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) ** 0.8 * 255).astype(np.uint8)


def centroids(lab, labels):
    if len(labels) == 0:
        return np.zeros((0, 2))
    c = ndi.center_of_mass(np.ones_like(lab), lab, labels)
    return np.array([(x, y) for y, x in c])  # (x, y)


def fit(src, dst):
    """similarity mapping in-vivo (x, y) -> ex-vivo (x, y), robust to wrong pairs; None if too few"""
    if len(src) < 3:
        return None, 0
    if len(src) == 3:
        T = SimilarityTransform(); T.estimate(src, dst); return T, 3
    T, inl = ransac((src, dst), SimilarityTransform, min_samples=2, residual_threshold=12, max_trials=500, rng=0)
    return (T, int(inl.sum())) if T is not None and inl.sum() >= 3 else (None, 0)


def paint(img8, lab, cols, outline=None, faint=0.35):
    """grey image, all cells outlined faintly, cells in cols ({label: rgb}) tinted with a bright outline;
    outline = (label image, labels, rgb): extra outlines (missed ground-truth cells)"""
    rgb = np.repeat(img8[..., None], 3, 2).astype(np.float32)
    b = find_boundaries(lab, mode="inner")
    rgb[b] = rgb[b] * (1 - faint) + 255 * faint
    if cols:
        C = np.zeros((int(lab.max()) + 1, 3), np.float32); on = np.zeros(len(C), bool)
        for l, c in cols.items():
            if l < len(C):
                C[l] = c; on[l] = True
        m = on[lab]
        rgb[m] = rgb[m] * 0.6 + C[lab[m]] * 0.4
        bb = find_boundaries(np.where(m, lab, 0), mode="inner")
        rgb[bb] = C[lab[bb]]
    if outline is not None:
        gl, keep, c = outline
        gb = find_boundaries(np.where(np.isin(gl, list(keep)), gl, 0), mode="inner")
        rgb[gb] = c
    return rgb.clip(0, 255).astype(np.uint8)


def render(sid, split, liv, lex, pairs, cols, T_from=None, gt=None):
    """pairs: [(iv label, ex label)]; cols: [rgb] per pair; T_from: pairs used for the alignment
    (default: the predicted ones); gt: (giv, gex, missed GT pairs) for the cv view"""
    iv, ex = load_images(sid, split)
    iv8, ex8 = norm8(iv), norm8(ex)
    ref = T_from if T_from is not None else pairs
    T, inl = fit(centroids(liv, [a for a, _ in ref]) if T_from is None else centroids(gt[0], [a for a, _ in ref]),
                 centroids(lex, [b for _, b in ref]) if T_from is None else centroids(gt[1], [b for _, b in ref]))
    civ = {a: c for (a, _), c in zip(pairs, cols)}
    cex = {b: c for (_, b), c in zip(pairs, cols)}
    out_iv = (gt[0], {a for a, _ in gt[2]}, YELLOW) if gt else None
    p1 = paint(iv8, liv, civ, out_iv)
    if T is not None:
        w = lambda a, order: warp(a, T, output_shape=iv.shape, order=order, preserve_range=True, mode="constant", cval=0)
        ex8w = w(ex8.astype(np.float32), 1).astype(np.uint8)
        lexw = w(lex.astype(np.float32), 0).astype(np.int32)
        out_ex = (w(gt[1].astype(np.float32), 0).astype(np.int32), {b for _, b in gt[2]}, YELLOW) if gt else None
        p2 = paint(ex8w, lexw, cex, out_ex)
        p3 = np.stack([ex8w, iv8, ex8w], 2)  # in-vivo green, ex-vivo magenta
        s, th = float(T.scale), float(np.degrees(T.rotation))
        reg = f"scale {s:.2f}, rotation {th:+.0f}°, {inl}/{len(ref)} pairs consistent"
    else:  # not registered: the whole ex-vivo image, shrunk to the in-vivo height
        p2 = paint(ex8, lex, cex)
        p3 = None
        reg = "not registered (fewer than 3 consistent pairs)"
    H = iv.shape[0]
    panels = [Image.fromarray(p) for p in (p1, p2, p3) if p is not None]
    panels = [p.resize((max(1, round(p.width * H / p.height)), H), Image.LANCZOS) for p in panels]
    gap = 8
    canvas = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), H), (24, 24, 24))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width + gap
    return canvas, reg


def save(canvas, out, sid, width=1500):
    canvas.save(os.path.join(out, sid + ".jpg"), quality=90)
    im = canvas.resize((width, round(canvas.height * width / canvas.width)), Image.LANCZOS) if canvas.width > width else canvas
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pair_colours(liv, pairs):
    """distinct colours, neighbouring cells (sorted by position) far apart in hue"""
    c = centroids(liv, [a for a, _ in pairs])
    order = np.lexsort((c[:, 0], c[:, 1])) if len(c) else []
    cols = [None] * len(pairs)
    for k, i in enumerate(order):
        cols[i] = tuple(int(255 * v) for v in colorsys.hsv_to_rgb((k * 0.618) % 1, 0.85, 1.0))
    return cols


def test_task(args):
    sid, iv_inst, ex_inst, mp, out = args
    iv, ex = load_images(sid, "hidden_test")
    liv, ids_iv = instances_to_label(json.loads(iv_inst), iv.shape)
    lex, ids_ex = instances_to_label(json.loads(ex_inst), ex.shape)
    li = {c: k + 1 for k, c in enumerate(ids_iv)}; le = {c: k + 1 for k, c in enumerate(ids_ex)}
    pairs = [(li[a], le[b]) for a, b in json.loads(mp) if a in li and b in le]
    canvas, reg = render(sid, "hidden_test", liv, lex, pairs, pair_colours(liv, pairs))
    return dict(sid=sid, img=save(canvas, out, sid), caption=f"{len(pairs)} pairs · {len(ids_iv)} in-vivo cells · "
                                                              f"{len(ids_ex)} ex-vivo cells · {reg}", n=len(pairs))


def page(title, sub, rows, out, back=None):
    cards = []
    for mouse in sorted({r["sid"].split("__")[0] for r in rows}):
        cards.append(f"<h2>{mouse.replace('subject_', 'Mouse ')}</h2>")
        for r in sorted((r for r in rows if r["sid"].startswith(mouse)), key=lambda r: r["sid"]):
            cards.append(f'<figure><img loading="lazy" src="{r["img"]}" alt="{r["sid"]}">'
                         f'<figcaption><b>{r["sid"].split("__region_")[1]}</b> · {r["caption"]}</figcaption></figure>')
    nav = f'<a class="back" href="{back}">← Cell Matching Diagnostics</a>' if back else ""
    dark = ("--bg:#0f1210; --surface:#181b19; --ink:#f3f5f3; --ink-2:#bec5c0; --ring:rgba(255,255,255,.10); "
            "--accent:#3fbf85; --ok:#4cc77a; --bad:#f07a6a; color-scheme:dark;")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: one wide column of region panels per mouse, caption under each. Tokens shared with Cell Matching Diagnostics. */
:root {{ --bg:#f5f7f5; --surface:#fcfcfb; --ink:#0f1411; --ink-2:#4c5550; --ring:rgba(15,20,17,.10); --accent:#178a57;
  --ok:#1f8a4c; --bad:#c2410c; --f-ui:"Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-data:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; color-scheme:light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ {dark} }} }}
:root[data-theme="dark"] {{ {dark} }}
* {{ box-sizing:border-box; }}
html {{ padding-block: env(safe-area-inset-top, 0px) env(safe-area-inset-bottom, 0px); }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 var(--f-ui); }}
main {{ max-width:1560px; margin:0 auto; padding-inline:16px; padding-block:24px 56px; display:grid; gap:14px; }}
h1, h2 {{ margin:0; line-height:1.2; text-wrap:balance; }} h1 {{ font-size:26px; }} h2 {{ font-size:18px; margin-top:18px; }}
p {{ margin:0; max-width:75ch; color:var(--ink-2); }}
.back {{ font:600 12px/1 var(--f-data); letter-spacing:.06em; text-transform:uppercase; color:var(--accent); text-decoration:none; }}
.back:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--ring); border-radius:10px; overflow:hidden; }}
img {{ display:block; width:100%; max-width:100%; height:auto; background:#181818; }}
figcaption {{ padding:8px 12px; font-size:13px; color:var(--ink-2); font-variant-numeric:tabular-nums; }}
figcaption b {{ color:var(--ink); font-family:var(--f-data); }}
.ok {{ color:var(--ok); font-weight:600; }} .bad {{ color:var(--bad); font-weight:600; }}
</style></head><body><main>{nav}<h1>{title}</h1>{sub}{''.join(cards)}</main></body></html>"""
    open(os.path.join(out, "index.html"), "w").write(html)


PANELS = ("<p>Left: in-vivo. Middle: ex-vivo resampled into the in-vivo frame with the fitted registration. "
          "Right: both overlaid, in-vivo green and ex-vivo magenta (white where they line up). "
          "Every predicted cell has a thin outline.</p>")


def cv_fold(task):
    fold, ivc, ivcp, exc, excp, method, thr, out = task
    from common import gt_labels, training_ids, load_flows, run_dir
    from cm_pipeline import match_regions, tp_map, parse_method, mod_masks, match_kw
    from evaluate_cv import fit_clf
    pairs_data, opt = parse_method(method)
    items = []
    for sid in training_ids(subjects=[fold]):
        iv, ex = load_images(sid, "training")
        dP, cp, up = load_flows(run_dir(ivc, "iv", fold), sid); liv = mod_masks(dP, cp, ivcp, up, opt, "iv")
        dP, cp, up = load_flows(run_dir(exc, "ex", fold), sid); lex = mod_masks(dP, cp, excp, up, opt, "ex")
        items.append(dict(sid=sid, iv_img=iv, ex_img=ex, liv=liv, lex=lex))
    clf = fit_clf(pairs_data, fold)
    P, log = match_regions(items, lambda s: clf, thrs=(thr,), **match_kw(opt))
    rows = []
    for it in items:
        sid = it["sid"]; giv, gex, gp = gt_labels(sid)
        miv, _ = tp_map(it["liv"], giv); mex, _ = tp_map(it["lex"], gex)
        gset = {(int(a), int(b)) for a, b in gp}
        pred = [(int(a), int(b)) for a, b in P[thr][sid]]
        ok = [a in miv and b in mex and (miv[a][0], mex[b][0]) in gset for a, b in pred]
        hit = {(miv[a][0], mex[b][0]) for (a, b), k in zip(pred, ok) if k}
        missed = [g for g in gset if g not in hit]
        canvas, _ = render(sid, "training", it["liv"], it["lex"], pred, [GREEN if k else RED for k in ok],
                           T_from=sorted(gset), gt=(giv, gex, missed))
        c = sum(ok)
        rows.append(dict(sid=sid, img=save(canvas, out, sid), n=len(pred),
                         caption=f"<span class='ok'>{c} correct</span> · <span class='bad'>{len(pred) - c} wrong</span> · "
                                 f"{len(missed)} of {len(gset)} verified pairs missed (yellow) · precision "
                                 f"{c / max(len(pred), 1):.2f}, recall {c / max(len(gset), 1):.2f}"))
        print(sid, f"correct {c}/{len(pred)}, GT {len(gset)}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "cv"])
    ap.add_argument("csv", nargs="?", help="test: the submission file")
    ap.add_argument("--iv", default="cyto3_x3:-1"); ap.add_argument("--ex", default="cyto3_x3:0")
    ap.add_argument("--pairs", default="oof"); ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--out", default=None); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--back", default=None, help="link back to a dashboard page (when the page is published)")
    a = ap.parse_args()
    if a.mode == "test":
        name = os.path.splitext(os.path.basename(a.csv))[0]
        out = a.out or os.path.join("viz", name); os.makedirs(out, exist_ok=True)
        sub = pd.read_csv(a.csv)
        tasks = [(r.sample_id, r.invivo_instances, r.exvivo_instances, r.match_pairs, out) for r in sub.itertuples()]
        with Pool(a.workers) as pool:
            rows = pool.map(test_task, tasks, chunksize=1)
        page(f"{name.replace('submission_', '')} test regions", f"<p>{sum(r['n'] for r in rows)} pairs on the 29 test regions. Each matched pair has its own color.</p>" + PANELS, rows, out, a.back)
    else:
        from configs import SUBJECTS
        (ivc, ivcp), (exc, excp) = [(x.rsplit(":", 1)[0], float(x.rsplit(":", 1)[1])) for x in (a.iv, a.ex)]
        out = a.out or os.path.join("viz", f"cv_{ivc}_{ivcp:g}_{exc}_{excp:g}_{a.pairs}_{a.thr:g}"); os.makedirs(out, exist_ok=True)
        with Pool(min(a.workers, len(SUBJECTS))) as pool:
            rows = [r for rr in pool.map(cv_fold, [(f, ivc, ivcp, exc, excp, a.pairs, a.thr, out) for f in SUBJECTS]) for r in rr]
        page("Held-out matching check", f"<p>in-vivo {a.iv}, ex-vivo {a.ex}, pair classifier {a.pairs}, threshold {a.thr:g}; "
             f"each mouse segmented and matched by models trained without it, then checked against the ground truth. "
             f"Green = correct pair, red = wrong pair, yellow outline = verified pair that was missed. The panels are aligned "
             f"with the true pairs.</p>" + PANELS, rows, out, a.back)
    print("wrote", os.path.join(out, "index.html"))
