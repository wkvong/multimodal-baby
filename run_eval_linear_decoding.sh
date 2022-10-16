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

linear_probe_checkpoints=("embedding_frozen_pretrained_contrastive_labeled_s_linear_probe_seed_0" "embedding_frozen_pretrained_contrastive_labeled_s_linear_probe_seed_1" "embedding_frozen_pretrained_contrastive_labeled_s_linear_probe_seed_2")

for checkpoint in ${linear_probe_checkpoints[@]}; do
    python eval_linear_decoding.py --checkpoint ${checkpoint}
done

