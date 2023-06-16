#!/bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=object_categories_linear_decoding
#SBATCH --output=slurm_logs/object_categories_linear_decoding.out
#SBATCH --error=slurm_logs/object_categories_linear_decoding.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=waikeenvong@gmail.com

source /home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-baby

seeds=(0)
splits=("last")

for split in ${splits[@]}; do
    for seed in ${seeds[@]}; do
        srun python object_categories_linear_decoding.py --train_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/object_categories' --num-classes 64 --epochs 100 --seed ${seed} --split ${split}
    done
done

