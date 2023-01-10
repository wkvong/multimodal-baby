#!/bin/bash

eval_types=("image")
eval_datasets=("saycam")
stage="test"
eval_metadata_filename="eval_test.json"

frozen_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
finetune_random_init_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
frozen_random_init_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
finetuned_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
shuffled_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
no_data_aug_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_augment_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_augment_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_augment_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
single_frame_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_multiple_frames_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

for eval_type in ${eval_types[@]}; do
    for eval_dataset in ${eval_datasets[@]}; do
        # frozen pre-trained eval
        # for checkpoint in ${frozen_pretrained_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # fine-tune random-init eval
        # for checkpoint in ${finetune_random_init_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # frozen random-init eval
        # for checkpoint in ${frozen_random_init_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # fine-tuned pre-trained eval
        for checkpoint in ${finetuned_pretrained_checkpoints[@]}; do
            python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        done

        # shuffled eval
        # for checkpoint in ${shuffled_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # no data augmentation eval
        # for checkpoint in ${no_data_aug_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # single frame eval
        # for checkpoint in ${single_frame_checkpoints[@]}; do
        #     python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --use_kitty_label --save_predictions
        # done

        # clip eval
        # python eval.py --clip_eval --eval_type ${eval_type} --eval_dataset ${eval_dataset} --stage ${stage} --eval_metadata_filename ${eval_metadata_filename}
    done
done
