# Which SLURM account and partitions the jobs are submitted to. Edit these three lines for your cluster.
#
#   NYU Torch (torch-login-*):  your accounts:  sacctmgr -nP show assoc user=$USER format=account
#                               GPU partitions: l40s_public, h200_public, a100, ...   CPU: cpu_short
#   Course cloud bursting (ood.burst.hpc.nyu.edu, Lecture 1):
#                               CM_ACCOUNT=cs_gy_6643-2026fa  CM_GPU_PARTITION=g2-standard-12  CM_CPU_PARTITION=n2c48m24
CM_ACCOUNT="${CM_ACCOUNT:-torch_pr_355_general}"
CM_GPU_PARTITION="${CM_GPU_PARTITION:-l40s_public}"
CM_CPU_PARTITION="${CM_CPU_PARTITION:-cpu_short}"

# command-line options override the #SBATCH lines inside jobs/*.sbatch
GPU_OPTS="--account=$CM_ACCOUNT --partition=$CM_GPU_PARTITION --gres=gpu:1"
CPU_OPTS="--account=$CM_ACCOUNT --partition=$CM_CPU_PARTITION"
