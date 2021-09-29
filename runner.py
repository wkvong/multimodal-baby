# slurm job runner script
# adapted from: https://gist.github.com/willwhitney/e1509c86522896c6930d2fe9ea49a522

import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv 

# create slurm directories
if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

# set code dir
code_dir = "/home/wv9/code/WaiKeen/multimodal-baby"

# config
basename = "multimodal"
grids = [
    {
        "main_file": ['train'],
        "embedding_type": ["spatial", "flat"],
        "text_encoder": ["embedding", "lstm"],
        "embedding_dim": [64, 128],
        "sim": ["mean"],
        "pretrained_cnn": [True],
        "multiple_frames": [True],
        "augment_frames": [True],
        # normalize features?
        # self distillation?
        "gpus": [1],
        "num_workers": [8],
        "batch_size": [16, 64, 256],
        "max_epochs": [50],
        # learning rate?
        # weight decay?
        # seed?
        "checkpoint_callback": ["False"],
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

if dry_run:
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
    jobname = basename
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

    slurm_script_path = 'slurm_scripts/' + jobname + '.slurm'
    slurm_script_dir = os.path.dirname(slurm_script_path)
    os.makedirs(slurm_script_dir, exist_ok=True)

    slurm_log_dir = 'slurm_logs/' + jobname 
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    job_source_dir = code_dir

    # don't need to copy code over
    # try:
    #     os.makedirs(job_source_dir)
    #     os.system('cp -R ./* ' + job_source_dir)
    # except FileExistsError:
    #     # with the 'clear' flag, we're starting fresh
    #     # overwrite the code that's already here
    #     if clear:
    #         print("Overwriting existing files.")
    #         os.system('cp -R ./* ' + job_source_dir)

    jobcommand = "python {}.py{}".format(job['main_file'], flagstring)

    job_start_command = "sbatch " + slurm_script_path

    print(jobcommand)
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" +
                        jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --signal=USR1@600\n")
        slurmfile.write("#SBATCH --time=1-00\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32GB\n")

        slurmfile.write("#SBATCH -c 4\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")

        slurmfile.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
        slurmfile.write("#SBATCH --mail-user=waikeenvong@gmail.com\n")

        slurmfile.write("cd " + job_source_dir + '\n')
        slurmfile.write("srun " + jobcommand)
        slurmfile.write("\n")

    # if not dry_run:
    #     os.system(job_start_command + " &")
