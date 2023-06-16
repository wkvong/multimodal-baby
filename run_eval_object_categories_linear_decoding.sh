#!/bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=object_categories_eval_linear_decoding
#SBATCH --output=slurm_logs/object_categories_eval_linear_decoding.out
#SBATCH --error=slurm_logs/object_categories_eval_linear_decoding.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=waikeenvong@gmail.com

source /home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-baby

linear_probe_checkpoints=("object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_0_split_first"
"object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_1_split_first"
"object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_2_split_first"
"object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_0_split_last"
"object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_1_split_last"
"object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_2_split_last")

for checkpoint in ${linear_probe_checkpoints[@]}; do
    srun python eval_object_categories_linear_decoding.py --checkpoint ${checkpoint} --save_predictions
done
