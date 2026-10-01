"""Maps a SLURM array index to one (config, modality, fold) training job.
usage: python tasks.py count            -> number of CV tasks
       python tasks.py get INDEX        -> "config mod fold"
       python tasks.py full CFG_IV CFG_EX count|get INDEX   -> tasks for final all-mice training"""
import os, sys
from configs import CV_CONFIGS, MODS, SUBJECTS

if sys.argv[1] == "full":
    cfg_iv, cfg_ex = sys.argv[2], sys.argv[3]
    # an ensemble "a+b" needs each member trained on all mice
    T = [(c, "iv", "full") for c in cfg_iv.split("+")] + [(c, "ex", "full") for c in cfg_ex.split("+")]
    cmd, rest = sys.argv[4], sys.argv[5:]
else:
    # CM_CV_TASKS="cyto3_x4:ex cyto3_x2_s1:iv ..." limits the run to those (config, modality) pairs
    spec = os.environ.get("CM_CV_TASKS", "").replace(",", " ").split()
    cm = [tuple(x.split(":")) for x in spec] or [(c, m) for c in CV_CONFIGS for m in MODS]
    # CM_CV_FOLDS="only_subject_5d294c ..." replaces the held-out mice (one-mouse models for the strict evaluation)
    folds = os.environ.get("CM_CV_FOLDS", "").split() or SUBJECTS
    T = [(c, m, f) for c, m in cm for f in folds]
    cmd, rest = sys.argv[1], sys.argv[2:]
if cmd == "count":
    print(len(T))
else:
    print(" ".join(T[int(rest[0])]))
