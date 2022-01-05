grids = [
    {
        "main_file": ["train"],
        "lambda_mm": [0.],
        "lambda_lm": [1.],
        "embedding_type": ["spatial"],
        "text_encoder": ["lstm"],
        "embedding_dim": [32],
        "tie": ["True"],
        "bias": ["True"],
        "crange": [1],
        "dropout_i": [.0],
        "dropout_o": [.1],
        "sim": ["mean"],
        "pretrained_cnn": [True],
        "multiple_frames": [True],
        "augment_frames": [True],
        # "normalize_features": [True, False],
        # self distillation?
        "gpus": [1],
        "num_workers": [4],
        "batch_size": [128, 256],
        "drop_last": [True],
        "max_epochs": [50],
        "optimizer": ["AdamW"],
        "lr": [0.03],
        "weight_decay": [0.00, 0.01, 0.02, 0.03],
        "seed": [0, 1],
        "optimize_unused": [True],
        "checkpoint_callback": ["True"],
        "logger": ["True"]
    },
]
# ordered flags to display in jobname
flags = [
    "text_encoder", "embedding_dim", "tie", "bias", "dropout_i", "dropout_o",
    "batch_size", "drop_last", "optimizer", "lr", "weight_decay", "seed",
]
