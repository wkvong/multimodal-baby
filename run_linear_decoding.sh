#!/bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=linear_decoding
#SBATCH --output=slurm_logs/linear_decoding_all.out
#SBATCH --error=slurm_logs/linear_decoding_all.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=waikeenvong@gmail.com

source /home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-baby

subsets=(1 0.1 0.01)

for subset in ${subsets[@]}; do
    srun python linear_decoding.py --train_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/dev' --test_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/test' --num-classes 22 --epochs 100 --subset ${subset}
done
