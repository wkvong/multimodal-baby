# slurm job runner script
# adapted from: https://gist.github.com/willwhitney/e1509c86522896c6930d2fe9ea49a522

import os
import sys
import itertools
import argparse
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--basename", default="multimodal")
argparser.add_argument("--main-file", default="train")
argparser.add_argument("--logs", type=Path, default=Path("slurm_logs"))
argparser.add_argument("--scripts", type=Path, default=Path("slurm_scripts"))
argparser.add_argument("--code-dir", type=Path, default=Path())
argparser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
argparser.add_argument("--mail-type", default="BEGIN,END,FAIL")
argparser.add_argument("--mail-user", default="waikeenvong@gmail.com")
argparser.add_argument("--python", default="python")
argparser.add_argument("--conda", type=Path, default=Path("/home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh"))
argparser.add_argument("--dry-run", action="store_true")
args = argparser.parse_args()

# create slurm directories
args.logs.mkdir(parents=True, exist_ok=True)
args.scripts.mkdir(parents=True, exist_ok=True)

# try conda
try:
    conda_avail = args.conda.exists()
except:
    conda_avail = False
if not conda_avail:
    args.conda = None

# config
grids = [
    {
        "main_file": [args.main_file],
        "embedding_type": ["spatial", "flat"],
        "text_encoder": ["embedding", "lstm"],
        "embedding_dim": [128],
        "sim": ["mean"],
        "pretrained_cnn": [True],
        "multiple_frames": [True],
        "augment_frames": [True],
        # "normalize_features": [True, False],
        # self distillation?
        "gpus": [1],
        "num_workers": [4],
        "batch_size": [16, 64, 128],
        "max_epochs": [100],
        # learning rate?
        # weight decay?
        # seed?
        "checkpoint_callback": ["True"],
        "logger": ["True"]
    },
]

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if args.dry_run:
    print("NOT starting {} jobs:".format(len(jobs)))
else:
    print("Starting {} jobs:".format(len(jobs)))

all_keys = set().union(*[g.keys() for g in grids])
merged = {k: set() for k in all_keys}
for grid in grids:
    for key in all_keys:
        grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
        merged[key] = merged[key].union(grid_key_value)
varying_keys = {key for key in merged if len(merged[key]) > 1}

excluded_flags = {'main_file'}

for job in jobs:
    jobname = args.basename
    flagstring = ""
    for flag in job:

        # construct the string of arguments to be passed to the script
        if not flag in excluded_flags:
            if isinstance(job[flag], bool):
                if job[flag]:
                    flagstring = flagstring + " --" + flag
                else:
                    print("WARNING: Excluding 'False' flag " + flag)
            else:
                flagstring = flagstring + " --" + flag + " " + str(job[flag])

        # construct the job's name
        if flag in varying_keys:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
    flagstring = flagstring + " --exp_name " + jobname

    # create slurm script, slurm log and checkpoint dirs
    slurm_script_path = args.scripts / (jobname + '.slurm')
    slurm_script_path.parent.mkdir(parents=True, exist_ok=True)

    slurm_log_dir = args.logs / jobname
    slurm_log_dir.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoints / jobname
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    
    job_source_dir = args.code_dir

    # specify job command and create slurm file
    jobcommand = f"{args.python} {job['main_file']}.py{flagstring}"

    job_start_command = f"sbatch {slurm_script_path}"

    print(jobcommand)
    with slurm_script_path.open('w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --job-name={jobname}\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write(f"#SBATCH --output={(args.logs / (jobname + '.out')).absolute()}\n")
        slurmfile.write(f"#SBATCH --error={(args.logs / (jobname + '.err')).absolute()}\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --time=6:00:00\n")
        slurmfile.write("#SBATCH --mem=32GB\n")
        slurmfile.write("#SBATCH --cpus-per-task=4\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --constraint=pascal|turing|volta\n")
        slurmfile.write(f"#SBATCH --mail-type={args.mail_type}\n")
        slurmfile.write(f"#SBATCH --mail-user={args.mail_user}\n\n")

        if args.conda:
            slurmfile.write('source \n')
            slurmfile.write('conda activate pytorch\n')
        if job_source_dir != Path():
            slurmfile.write(f"cd {job_source_dir}\n")
        slurmfile.write("srun " + jobcommand)
        slurmfile.write("\n")

    if not args.dry_run:
        os.system(job_start_command + " &")
