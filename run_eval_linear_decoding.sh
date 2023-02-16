#!/bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=linear_decoding
#SBATCH --output=slurm_logs/linear_decoding_subset.out
#SBATCH --error=slurm_logs/linear_decoding_subset.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=waikeenvong@gmail.com

source /home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-baby

linear_probe_checkpoints=("self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_1.0_seed_0"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_1.0_seed_1"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_1.0_seed_2"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.1_seed_0"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.1_seed_1"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.1_seed_2"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.01_seed_0"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.01_seed_1"
                          "self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_subset_0.01_seed_2")

for checkpoint in ${linear_probe_checkpoints[@]}; do
    python eval_linear_decoding.py --checkpoint ${checkpoint} --save_predictions
done

