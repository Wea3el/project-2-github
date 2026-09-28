"""Maps a SLURM array index to one (config, modality, fold) training job.
usage: python tasks.py count            -> number of CV tasks
       python tasks.py get INDEX        -> "config mod fold"
       python tasks.py full CFG_IV CFG_EX count|get INDEX   -> tasks for final all-mice training"""
import sys
from configs import CV_CONFIGS, MODS, SUBJECTS

if sys.argv[1] == "full":
    cfg_iv, cfg_ex = sys.argv[2], sys.argv[3]
    T = [(cfg_iv, "iv", "full"), (cfg_ex, "ex", "full")]
    cmd, rest = sys.argv[4], sys.argv[5:]
else:
    T = [(c, m, f) for c in CV_CONFIGS for m in MODS for f in SUBJECTS]
    cmd, rest = sys.argv[1], sys.argv[2:]
if cmd == "count":
    print(len(T))
else:
    print(" ".join(T[int(rest[0])]))
