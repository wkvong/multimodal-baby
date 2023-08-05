all_groups = {
    "LSTM 0": ["LSTM 0"],
    "LSTM": ["LSTM 0", "LSTM 1", "LSTM 2"],
    "Captioning LSTM": ["Captioning LSTM 0", "Captioning LSTM 1", "Captioning LSTM 2"],
    "CBOW": ["CBOW 0", "CBOW 1", "CBOW 2"],
    "Contrastive": ["Contrastive 0", "Contrastive 1", "Contrastive 2"],
    "Joint bs16": ["Joint bs16 0", "Joint bs16 1", "Joint bs16 2"],
    "Joint bs512": ["Joint bs512 0", "Joint bs512 1", "Joint bs512 2"],
    "unigram": ["1-gram"],
    "bigram": ["2-gram"],
    "trigram": ["3-gram"],
    "4-gram": ["4-gram"],
}

all_checkpoint_paths = {
    "childes": {
        "LSTM 0": "checkpoints/lm_dataset_childes_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_patience_5_weight_decay_0.04_seed_0/epoch=35.ckpt",
    },
    "saycam": {
        "LSTM 0": "checkpoints/lm_dataset_saycam_captioning_False_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_0/epoch=29.ckpt",
        "LSTM 1": "checkpoints/lm_dataset_saycam_captioning_False_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_1/epoch=38.ckpt",
        "LSTM 2": "checkpoints/lm_dataset_saycam_captioning_False_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_2/epoch=28.ckpt",
        "Captioning LSTM 0": "checkpoints/lm_dataset_saycam_captioning_True_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_0/epoch=29.ckpt",
        "Captioning LSTM 1": "checkpoints/lm_dataset_saycam_captioning_True_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_1/epoch=42.ckpt",
        "Captioning LSTM 2": "checkpoints/lm_dataset_saycam_captioning_True_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_16_lr_0.006_lr_scheduler_True_weight_decay_0.04_seed_2/epoch=38.ckpt",
        "CBOW 0": "checkpoints/lm_dataset_saycam_text_encoder_cbow_embedding_dim_512_tie_False_bias_False_crange_1_dropout_i_0.0_dropout_o_0.1_batch_size_8_lr_0.003_lr_scheduler_True_patience_2_weight_decay_0.04_seed_0/epoch=31.ckpt",
        "CBOW 1": "checkpoints/lm_dataset_saycam_text_encoder_cbow_embedding_dim_512_tie_False_bias_False_crange_1_dropout_i_0.0_dropout_o_0.1_batch_size_8_lr_0.003_lr_scheduler_True_patience_2_weight_decay_0.04_seed_1/epoch=65.ckpt",
        "CBOW 2": "checkpoints/lm_dataset_saycam_text_encoder_cbow_embedding_dim_512_tie_False_bias_False_crange_1_dropout_i_0.0_dropout_o_0.1_batch_size_8_lr_0.003_lr_scheduler_True_patience_2_weight_decay_0.04_seed_2/epoch=58.ckpt",
        "Contrastive 0": "checkpoints/multimodal_dataset_saycam_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_fix_temperature_True_temperature_0.1_batch_size_512_lr_0.01_lr_scheduler_True_weight_decay_0.05_eval_include_sos_eos_True_seed_0/epoch=117.ckpt",
        "Contrastive 1": "checkpoints/multimodal_dataset_saycam_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_fix_temperature_True_temperature_0.1_batch_size_512_lr_0.01_lr_scheduler_True_weight_decay_0.05_eval_include_sos_eos_True_seed_1/epoch=109.ckpt",
        "Contrastive 2": "checkpoints/multimodal_dataset_saycam_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_fix_temperature_True_temperature_0.1_batch_size_512_lr_0.01_lr_scheduler_True_weight_decay_0.05_eval_include_sos_eos_True_seed_2/epoch=108.ckpt",
        "Joint bs16 0": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_16_optimizer_AdamW_lr_0.006_lr_scheduler_True_weight_decay_0.04_val_batch_size_16_eval_include_sos_eos_True_seed_0/epoch=93.ckpt",
        "Joint bs16 1": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_16_optimizer_AdamW_lr_0.006_lr_scheduler_True_weight_decay_0.04_val_batch_size_16_eval_include_sos_eos_True_seed_1/epoch=104.ckpt",
        "Joint bs16 2": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_16_optimizer_AdamW_lr_0.006_lr_scheduler_True_weight_decay_0.04_val_batch_size_16_eval_include_sos_eos_True_seed_2/epoch=119.ckpt",
        "Joint bs512 0": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_512_optimizer_AdamW_lr_0.01_lr_scheduler_True_weight_decay_0.05_val_batch_size_16_eval_include_sos_eos_True_seed_0/epoch=118.ckpt",
        "Joint bs512 1": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_512_optimizer_AdamW_lr_0.01_lr_scheduler_True_weight_decay_0.05_val_batch_size_16_eval_include_sos_eos_True_seed_1/epoch=103.ckpt",
        "Joint bs512 2": "checkpoints/joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_batch_size_512_optimizer_AdamW_lr_0.01_lr_scheduler_True_weight_decay_0.05_val_batch_size_16_eval_include_sos_eos_True_seed_2/epoch=81.ckpt",

        "1-gram": "1-gram",
        "2-gram": "2-gram",
        "3-gram": "3-gram",
        "4-gram": "4-gram",
    },
    "coco": {
        "lm": "checkpoints/lm_dataset_coco_captioning_False_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=109.ckpt",
        "lm_bs512": "checkpoints/lm_dataset_coco_captioning_False_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_512_lr_0.01_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=67.ckpt",
        "capt": "checkpoints/lm_dataset_coco_captioning_True_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=149.ckpt",
        "capt_ft": "checkpoints/lm_dataset_coco_captioning_True_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0_finetune_cnn/epoch=210.ckpt",
        "capt_bs512": "checkpoints/lm_dataset_coco_captioning_True_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_512_lr_0.003_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=65.ckpt",
        "capt_attn": "checkpoints/lm_dataset_coco_captioning_True_attention_True_pretrained_cnn_True_cnn_model_resnext50_32x4d_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.0003_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=198.ckpt",
        "capt_attn_gt": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_pretrained_cnn_True_cnn_model_resnext50_32x4d_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=143.ckpt",
        "capt_attn_gt_ft": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_pretrained_cnn_True_cnn_model_resnext50_32x4d_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0_finetune_cnn/epoch=198.ckpt",
        "capt_attn_gt_reg": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_lambda_ar_1.0_pretrained_cnn_True_cnn_model_resnext50_32x4d_tie_True_bias_True_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=149.ckpt",
        "capt_attn_gt_reg_ft": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_lambda_ar_1.0_pretrained_cnn_True_cnn_model_resnext50_32x4d_tie_True_bias_True_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0_finetune_cnn/epoch=211.ckpt",
        "capt_attn_gt_reg_untie": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_lambda_ar_1.0_pretrained_cnn_True_cnn_model_resnext50_32x4d_tie_False_bias_True_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=136.ckpt",
        "capt_attn_gt_reg_untie_ft": "checkpoints/lm_dataset_coco_captioning_True_attention_True_attention_gate_True_lambda_ar_1.0_pretrained_cnn_True_cnn_model_resnext50_32x4d_tie_False_bias_True_batch_size_8_lr_0.001_lr_scheduler_True_weight_decay_0.01_seed_0_finetune_cnn/epoch=187.ckpt",

        "cbow": "checkpoints/lm_dataset_coco_captioning_False_text_encoder_cbow_embedding_dim_512_tie_False_bias_False_crange_2_dropout_i_0.0_dropout_o_0.0_batch_size_8_lr_0.0003_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=153.ckpt",

        "contrastive": "checkpoints/multimodal_dataset_coco_captioning_False_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_fix_temperature_True_temperature_0.05_batch_size_512_lr_0.0003_weight_decay_0.01_seed_0/epoch=51.ckpt",

        "joint": "checkpoints/joint_dataset_coco_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_dropout_i_0.0_dropout_o_0.0_fix_temperature_True_temperature_0.05_batch_size_512_lr_0.01_lr_scheduler_True_weight_decay_0.01_seed_0/epoch=191.ckpt",
    },
}
