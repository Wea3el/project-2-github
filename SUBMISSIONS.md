# Submissions: what each file contains, its score, and what it taught us

Check this before trying a variant. Every prediction also appends its exact settings to `submissions_log.tsv`
(on the cluster) together with the git commit, a hash of the code and the sha1 of the model weights and pair-classifier
data, and `bash ksub.sh NAME ...` sends the settings to Kaggle as the submission description. Submission names are
never reused: the scripts refuse to overwrite an existing file.

**Current best: f3, 0.51716.** In-vivo `cyto3_x3_auto` at cell probability −1, ex-vivo Cellpose-SAM `cpsam2_x3` at −1
with a stricter flow check (0.3), pair threshold 0.05, pair classifier `oof`:
`bash pv.sh f3 cyto3_x3_auto cpsam2_x3 "--cp-iv -1 --cp-ex -1 --thr 0.05 --pairs oof+flow_ex=0.3"`

**Small differences:** the final ranking uses the private part of the test set, so a small public gain may not carry
over. The 0.014–0.017 gaps between neighbouring thresholds are differences between predictions, not an estimate of
leaderboard noise, and there is no fixed size below which a gain is noise. Prefer changes that follow a consistent trend
over isolated small wins.

## Lessons (do not repeat)
1. **The ex-vivo cell-probability threshold is the biggest lever; −1 is the best value tested** for both cyto3_x3 and
   Cellpose-SAM (curve below). For Cellpose-SAM, −0.75 and −1.25 both lost 0.014–0.017; for cyto3, ±0.5 lost ~0.008.
   Another model (e.g. a self-trained one) can have a different best threshold.
2. **Thresholds chosen by cross-validation on the training mice do not transfer** (v9, v15). The right threshold
   depends on each mouse's cell size (calib_check: the best threshold is where the predicted median cell area matches
   the true one; true ex-vivo medians are 54, 79 and 98 px² on the three training mice). Tune thresholds on the leaderboard.
3. At −1 the ex-vivo masks are about the right size: growing them by 1 px (g1, −0.117) or loosening the flow check to
   0.8 (f8, −0.032) hurts; a stricter flow check (0.3) helped a little (f3, +0.005).
4. In-vivo: −1 is the best threshold tested (−2: −0.040 vs t2; −0.5: −0.018 and 0: −0.030 vs s1). Cellpose-SAM in-vivo
   (t5), in-vivo self-training (i1) and in-vivo wider size augmentation (i3s, −0.0075) do not help.
5. Ex-vivo models: Cellpose-SAM beats cyto3 at the same threshold (+0.021 at 0, +0.011 at −1). Wider size augmentation
   helped the cyto3 ex-vivo model (e3x1, +0.010 vs t2), so Cellpose-SAM with it (e5) is queued. Ensembles of cyto3 and
   Cellpose-SAM are worse than Cellpose-SAM alone (t7; n1, −0.008).
6. The matching variants tried (m1–m4) did not change the score with the older ex-vivo masks (threshold 0). That does
   not show matching has no headroom with the current masks (a classifier built on s1-style masks is queued: f3t1p).
   Choosing the in-vivo version per region lost (v14).
7. Ex-vivo self-training with cp-0 masks as labels hurt (e1): those labels were too small.
8. Pair threshold: 0.1 slightly better than 0.05 (thr1, +0.003), 0.02 worse (thr02, −0.0065).

Ex-vivo threshold curves (in-vivo cyto3_x3_auto at −1, pair threshold 0.05, oof):

| ex-vivo threshold | 0.5 | 0 | −0.5 | −0.75 | −1 | −1.25 | −1.5 | −2 | −3 |
|---|---|---|---|---|---|---|---|---|---|
| cyto3_x3 | 0.340 (v15) | 0.415 (v13) | 0.493 (t1) | | **0.501** (t2) | | 0.494 (x15) | 0.430 (x2) | 0.338 (x3) |
| cpsam2_x3 | | 0.436 (t6) | | 0.495 (s075) | **0.5125** (s1) | 0.498 (s125) | | 0.459 (s2) | |

## All submissions
"cp" = cell-probability threshold. Pair classifier `gt` = shipped (trained on ground-truth masks), `oof` = trained on out-of-fold predicted masks.

| file | in-vivo model : cp | ex-vivo model : cp | pair thr | other | public LB | verdict |
|---|---|---|---|---|---|---|
| v3 | shipped : −1 | shipped : 0 | 0.02 | original code, gt | 0.35329 | baseline |
| v4 | shipped : −1 | shipped : 0 | 0.1 | + mouse-level consensus registration | 0.36217 | +0.009 |
| v5 | shipped : −1 | cyto3_x3 : 0 | 0.1 | gt | 0.39429 | fine-tuned ex-vivo model +0.032 |
| v6 | cyto3_x3 : −1 | cyto3_x3 : 0 | 0.1 | gt | 0.40552 | +0.011 |
| v9 | cyto3_x2 : −0.5 | cyto3_x3 : 0 | 0.2 | gt (cross-validation winner) | 0.38002 | ✗ CV +0.006, LB −0.026 |
| v12 | cyto3_x3 : −1 | cyto3_x3 : 0 | 0.05 | oof | 0.41291 | ✓ +0.007 |
| v13 | cyto3_x3_auto : −1 | cyto3_x3 : 0 | 0.05 | oof | 0.41504 | ✓ size rescaling +0.002 |
| v14 (sel) | per-region choice of 7 versions | cyto3_x3 : 0 | 0.05 | oof, rule conf | 0.40953 | ✗ −0.006 |
| v15 | cyto3_x3_auto : −1 | cyto3_x3 : 0.5 | 0.05 | oof | 0.34008 | ✗ −0.075 |
| v16 | cpsam2_x3_auto : −0.5 | cpsam2_x3 : 0.5 | 0.05 | oof | 0.34863 | ✗ ex threshold 0.5 |
| i1 | cyto3_x3_pl_auto : −1 | cyto3_x3 : 0 | 0.05 | oof | 0.41540 | tie |
| e1 | cyto3_x3_auto : −1 | cyto3_x3_pl : 0 | 0.05 | oof | 0.40043 | ✗ −0.015 |
| t1 | cyto3_x3_auto : −1 | cyto3_x3 : −0.5 | 0.05 | oof | 0.49324 | ✓ +0.078 |
| t2 | cyto3_x3_auto : −1 | cyto3_x3 : −1 | 0.05 | oof | 0.50127 | ✓ +0.086 |
| t5 | cpsam2_x3_auto : −0.5 | cyto3_x3 : 0 | 0.05 | oof | 0.41493 | tie |
| t6 | cyto3_x3_auto : −1 | cpsam2_x3 : 0 | 0.05 | oof | 0.43635 | ✓ +0.021 vs v13 |
| t7 | cyto3_x3_auto : −1 | cyto3_x3+cpsam2_x3 : 0 | 0.05 | oof | 0.42955 | between the two |
| m1 | cyto3_x3_auto : −1 | cyto3_x3 : 0 | 0.05 | oof+cand=gain+u=10 | 0.41504 | no change |
| m2 | cyto3_x3_auto : −1 | cyto3_x3 : 0 | 0.05 | oof+cand=gain+u=6 | 0.40972 | ✗ −0.005 |
| m3 | cyto3_x3_auto : −1 | cyto3_x3 : 0 | 0.05 | oofall+cand=all | 0.41497 | no change |
| m4 | cyto3_x3_pl_auto : −1 | cyto3_x3 : 0 | 0.05 | oof+cand=gain+u=10 | 0.41540 | = i1 |
| x15 | cyto3_x3_auto : −1 | cyto3_x3 : −1.5 | 0.05 | oof | 0.49361 | −0.008 vs t2 |
| x2 | cyto3_x3_auto : −1 | cyto3_x3 : −2 | 0.05 | oof | 0.42971 | ✗ −0.072 vs t2 |
| x3 | cyto3_x3_auto : −1 | cyto3_x3 : −3 | 0.05 | oof | 0.33765 | ✗ −0.164 vs t2 |
| s1 | cyto3_x3_auto : −1 | cpsam2_x3 : −1 | 0.05 | oof | 0.51250 | ✓ +0.011 vs t2 |
| s2 | cyto3_x3_auto : −1 | cpsam2_x3 : −2 | 0.05 | oof | 0.45947 | ✗ −0.053 vs s1 |
| g1 | cyto3_x3_auto : −1 | cyto3_x3 : −1 | 0.05 | oof+grow_ex=1 | 0.38437 | ✗ −0.117 vs t2 |
| f8 | cyto3_x3_auto : −1 | cyto3_x3 : −1 | 0.05 | oof+flow_ex=0.8 | 0.46958 | ✗ −0.032 vs t2 |
| iv2 | cyto3_x3_auto : −2 | cyto3_x3 : −1 | 0.05 | oof | 0.46116 | ✗ −0.040 vs t2 |
| e3 | cyto3_x3_auto : −1 | cyto3_x3_sr : 0 | 0.05 | oof | not submitted | old ex threshold, superseded by e3x1 |
| i3 | cyto3_x3_sr_auto : −1 | cyto3_x3 : 0 | 0.05 | oof | not submitted | old ex threshold, superseded by i3s |
| s075 | cyto3_x3_auto : −1 | cpsam2_x3 : −0.75 | 0.05 | oof | 0.49515 | ✗ −0.017 vs s1 |
| s125 | cyto3_x3_auto : −1 | cpsam2_x3 : −1.25 | 0.05 | oof | 0.49843 | ✗ −0.014 vs s1 |
| thr1 | cyto3_x3_auto : −1 | cpsam2_x3 : −1 | 0.1 | oof | 0.51509 | ✓ +0.003 vs s1 |
| thr02 | cyto3_x3_auto : −1 | cpsam2_x3 : −1 | 0.02 | oof | 0.50599 | ✗ −0.0065 vs s1 |
| ivm05 | cyto3_x3_auto : −0.5 | cpsam2_x3 : −1 | 0.05 | oof | 0.49421 | ✗ −0.018 vs s1 |
| iv0 | cyto3_x3_auto : 0 | cpsam2_x3 : −1 | 0.05 | oof | 0.48269 | ✗ −0.030 vs s1 |
| n1 | cyto3_x3_auto : −1 | cyto3_x3+cpsam2_x3 : −1 | 0.05 | oof | 0.50418 | ✗ −0.008 vs s1 |
| **f3** | cyto3_x3_auto : −1 | cpsam2_x3 : −1 | 0.05 | oof+flow_ex=0.3 | **0.51716** | ✓ **best**, +0.005 vs s1 |
| i3s | cyto3_x3_sr_auto : −1 | cpsam2_x3 : −1 | 0.05 | oof | 0.50502 | ✗ −0.0075 vs s1 |
| e3x1 | cyto3_x3_auto : −1 | cyto3_x3_sr : −1 | 0.05 | oof | 0.51163 | ✓ +0.010 vs t2 (size augmentation helps the cyto3 ex model) |

## Queued (2026-10-02 afternoon), relative to f3 unless noted
| file | change | why |
|---|---|---|
| f3t1 | f3 + pair threshold 0.1 | combines the two small gains (f3, thr1) |
| f3t1p | f3t1 with the pair classifier `oofs1` (built from s1-style out-of-fold masks) | does a classifier matching the current masks help; replaces s1p |
| f2t1 | flow check 0.2 + pair threshold 0.1 | stricter still (0.8 lost, 0.3 won) |
| f3t15 / f3t2 | f3 + pair threshold 0.15 / 0.2 | 0.1 beat 0.05 and 0.02 lost: is the optimum higher? |
| diag_f3 (no submission) | failure breakdown of f3's settings on the held-out mice, with the F1 ceilings | how much is lost to segmentation vs registration vs the classifier decides the next work |
| tt1 (GPU inference only) | ex-vivo cpsam2_x3_tta (test-time augmentation) at f3's settings | another inference pass, no training |
| f3pf / f3pq | f3 with classifiers built from s1-style out-of-fold masks with the 0.3 flow check: `oofs1f` (the usual 14 features) and `oofs1q` (+ each cell's flow error and mask stability) | separates "classifier trained on matching masks" from "mask-confidence features" |
| e4 (GPU) | ex-vivo cpsam2_x3_pl1: Cellpose-SAM self-trained with s1's ex-vivo masks (teacher threshold −1) as labels; predicted at −1, then e4c0 / e4c05 at 0 / −0.5 | e1 failed with too-small labels; the student's best threshold is a new parameter |
| (e4h, only if e4 does not beat s1) | cpsam2_x3_pl1h: the same with 30% of the pseudo-labelled tiles (~15% of all tiles instead of ~40%) | before trying more self-trained models |
| e5 (GPU) | ex-vivo cpsam2_x3_sr: Cellpose-SAM with wider size augmentation | e3x1: size augmentation helped the cyto3 ex model by +0.010 |
| e6 (GPU) | ex-vivo cpsam2_x3_aug: Cellpose-SAM with brightness/contrast augmentation | robustness to the test images |
| e7 / e7n (GPU, after e5, e6 and tt1) | a second Cellpose-SAM run (cpsam2_x3_s1, other seed) alone, then averaged with cpsam2_x3 | averaging two runs of the same model is untested (the cyto3 + SAM ensembles were worse) |

The GPU runs (e4–e6) are first predicted with s1's settings (flow check 0.4, pair threshold 0.05), so compare them with s1;
re-predict the promising ones with the best settings of the moment.
