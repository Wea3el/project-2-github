#!/bin/bash
#SBATCH --job-name=cm_predict
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU only: masks from the cached test-set outputs -> registration -> pairs -> submission.csv
source "$CM_HOME/hpc_env.sh"
# then a picture of every test region: viz/<submission name>/index.html (see render_pairs.py)
cm_run "python predict_test.py --iv-config $CFG_IV --ex-config $CFG_EX ${PRED_ARGS:-} --out ${OUT:-submission.csv} && \
        (python render_pairs.py test ${OUT:-submission.csv} || echo 'pictures failed; the submission file is fine')"
