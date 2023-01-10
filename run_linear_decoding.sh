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

# frozen_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
# finetuned_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
# subsets=(0.1 0.01)
checkpoint='models/TC-S-resnext.tar'
subsets=(1 0.1 0.01)

for subset in ${subsets[@]}; do
    srun python linear_decoding.py --train_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/dev' --test_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/test' --checkpoint ${checkpoint} --num-classes 22 --epochs 100 --subset ${subset}
done


# for checkpoint in ${frozen_pretrained_checkpoints[@]}; do
#     for subset in ${subsets[@]}; do
#         srun python linear_decoding.py --train_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/dev' --test_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/test' --checkpoint ${checkpoint} --num-classes 22 --epochs 100 --subset ${subset}
#     done
# done

# for checkpoint in ${finetuned_pretrained_checkpoints[@]}; do
#     for subset in ${subsets[@]}; do
#         srun python linear_decoding.py --train_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/dev' --test_dir '/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval/test' --checkpoint ${checkpoint} --num-classes 22 --epochs 100 --subset ${subset}
#     done
# done
