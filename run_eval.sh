#!/bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=saycam_eval
#SBATCH --output=slurm_logs/saycam_eval.out
#SBATCH --error=slurm_logs/saycam_eval.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=waikeenvong@gmail.com

eval_types=("image")
eval_datasets=("saycam")
stage="test"
eval_metadata_filename="eval_test.json"
# eval_metadata_filename="eval_manual_filtered_test.json"

frozen_pretrained_checkpoints=("multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2"
"multimodal_cnn_dino_True_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
"multimodal_cnn_dino_True_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
"multimodal_cnn_dino_True_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

finetune_random_init_checkpoints=("multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

frozen_random_init_checkpoints=("multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

shuffled_checkpoints=("multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

single_frame_checkpoints=("multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

transformer_checkpoints=("multimodal_vit_dino_True_text_encoder_transformer_embedding_dim_512_batch_size_8_dropout_i_0.5_pos_embed_type_learned_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_vit_dino_True_text_encoder_transformer_embedding_dim_512_batch_size_8_dropout_i_0.5_pos_embed_type_learned_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_vit_dino_True_text_encoder_transformer_embedding_dim_512_batch_size_8_dropout_i_0.5_pos_embed_type_learned_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

source /home/wv9/code/WaiKeen/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-baby

# for eval_type in ${eval_types[@]}; do
#     for eval_dataset in ${eval_datasets[@]}; do
#       # frozen pre-trained eval
        # for checkpoint in ${frozen_pretrained_checkpoints[@]}; do
        #     srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --eval_metadata_filename ${eval_metadata_filename} --use_kitty_label --save_predictions
        # done

#         # fine-tune random-init eval
#         for checkpoint in ${finetune_random_init_checkpoints[@]}; do
#             srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         done

#         # frozen random-init eval
#         for checkpoint in ${frozen_random_init_checkpoints[@]}; do
#             srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         done

#         # fine-tuned pre-trained eval
#         # for checkpoint in ${finetuned_pretrained_checkpoints[@]}; do
#         #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         # done

#         # shuffled eval
#         for checkpoint in ${shuffled_checkpoints[@]}; do
#             srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         done

#         # no data augmentation eval
#         # for checkpoint in ${no_data_aug_checkpoints[@]}; do
#         #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         # done

#         # single frame eval
#         for checkpoint in ${single_frame_checkpoints[@]}; do
#             srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
#         done

#         # clip eval
        #         # python eval.py --clip_eval --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --eval_metadata_filename ${eval_metadata_filename}

#         # transformer eval
#         for checkpoint in ${transformer_checkpoints[@]}; do
#             srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --eval_metadata_filename ${eval_metadata_filename} --use_kitty_label --save_predictions
#         done
#     done
# done

# object categories eval
eval_datasets=("object_categories")
for eval_type in ${eval_types[@]}; do
    for eval_dataset in ${eval_datasets[@]}; do
        # frozen pre-trained eval
        # for checkpoint in ${frozen_pretrained_checkpoints[@]}; do
        #     srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # frozen random-init eval
        # for checkpoint in ${frozen_random_init_checkpoints[@]}; do
        #     srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done
        
        # shuffled eval
        # for checkpoint in ${shuffled_checkpoints[@]}; do
        #     srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # clip eval
#         srun python eval.py --clip_eval --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --save_predictions

        # transformer eval
        for checkpoint in ${transformer_checkpoints[@]}; do
            srun python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --eval_metadata_filename ${eval_metadata_filename} --use_kitty_label --save_predictions
        done
    done
done
        
