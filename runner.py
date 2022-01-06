# slurm job runner script
# adapted from: https://gist.github.com/willwhitney/e1509c86522896c6930d2fe9ea49a522

import os
import sys
import itertools
import argparse
from pathlib import Path
import copy

argparser = argparse.ArgumentParser(
    description="Generate and optionally submit slurm jobs. runner_config.py is the configuration file.")
argparser.add_argument("--basename", default="multimodal",
                       help="The basename of jobs. All jobnames will start with this basename.")
argparser.add_argument("--scripts", type=Path, default=Path("scripts"),
                       help="The directory of scripts.")
argparser.add_argument("--logs", type=Path, default=Path("logs"),
                       help="The directory of logs.")
argparser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"),
                       help="The directory of checkpoints.")
argparser.add_argument("--code-dir", type=Path, default=Path(),
                       help="The working directory of the jobs.")
argparser.add_argument("--time", default="6:00:00",
                       help="The time limit of the jobs.")
argparser.add_argument("--mem", default="32GB",
                       help="The memory limit of the jobs.")
argparser.add_argument("--mail-type", default="END,FAIL",
                       help="What types of mails to send to the mail user.")
argparser.add_argument("--mail-user", default="waikeenvong@gmail.com",
                       help="The mail user to send mails to.")
argparser.add_argument("--python", default="python",
                       help="The python to run with; e.g., python3.")
argparser.add_argument("--conda", type=Path,
                       default=Path("/home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh"),
                       help="The path to the conda.sh; ignored if failed to access.")
argparser.add_argument("--dry-run", action="store_true",
                       help="Do not start jobs when running. Without this flag, jobs will be immediately submitted.")
argparser.add_argument("--auto-flag", action="store_true",
                       help="Automatically find varying flags and display them in job names; if not set, use designated ordered list of flags.")
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
from runner_config import grids, flags

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
        grid_key_value = grid.get(key, [])
        merged[key] = merged[key].union(grid_key_value)
varying_keys = {key for key in merged if len(merged[key]) > 1}

if args.auto_flag:
    # display all varying keys in jobname
    flags = list(varying_keys)
else: # use flags
    # check whether there are flags that are varying but omitted in flags
    omitted_flags = [key for key in varying_keys if key not in flags]
    if omitted_flags:
        print(f"ERROR: {', '.join(omitted_flags)} are varying but omitted in flags")
        sys.exit()


excluded_flags = {'main_file'}

for job in jobs:
    # construct the job's name
    jobname = args.basename
    for flag in flags:
        value = job[flag]
        jobname = jobname + f"_{flag}_{value}"

    # construct the string of arguments to be passed to the script
    flagstring = ""

    # use the order of flags first, then all other flags at the last
    # this order is actually unimportant and simply for elegency
    flags_order = copy.copy(flags)
    for flag in job:
        if (flag not in flags_order) and (flag not in excluded_flags):
            flags_order.append(flag)

    for flag in flags_order:
        value = job[flag]
        if isinstance(value, bool):
            if value:
                flagstring = flagstring + f" --{flag}"
            else:
                print("WARNING: Excluding 'False' flag " + flag)
        else:
            flagstring = flagstring + f" --{flag} {value}"

    flagstring = flagstring + f" --exp_name {jobname}"

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
        slurmfile.write(f"#SBATCH --time={args.time}\n")
        slurmfile.write(f"#SBATCH --mem={args.mem}\n")
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
