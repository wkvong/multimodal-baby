#!/bin/bash

eval_types=("image" "text")
eval_datasets=("saycam" "object_categories")
models=("embedding" "lstm")

frozen_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
random_init_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
finetuned_pretrained_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")
shuffled_checkpoints=("multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1" "multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_2")

for eval_type in ${eval_types[@]}; do
    for eval_dataset in ${eval_datasets[@]}; do
        for model in ${models[@]}; do
            for checkpoint in ${frozen_pretrained_checkpoints[@]}; do
                python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --model ${model} --stage dev --use_kitty_label --save_predictions
            done

            for checkpoint in ${random_init_checkpoints[@]}; do
                python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --model ${model} --stage dev --use_kitty_label --save_predictions
            done

            for checkpoint in ${finetuned_pretrained_checkpoints[@]}; do
                python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --model ${model} --stage dev --use_kitty_label --save_predictions
            done

            for checkpoint in ${shuffled_checkpoints[@]}; do
                python eval.py --checkpoint ${checkpoint} --eval_type ${eval_type} --eval_dataset ${eval_dataset} --model ${model} --stage dev --use_kitty_label --save_predictions
            done
        done
    done
done
