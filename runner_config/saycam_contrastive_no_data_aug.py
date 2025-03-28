grids = [
    {
        "main_file": ["train"],
        "dataset": ["saycam"],
        "lambda_mm": [1.],
        "lambda_lm": [0.],
        "embedding_type": ["flat"],
        "text_encoder": ["embedding"],
        "embedding_dim": [512],
        "dropout_i": [.5],
        "dropout_o": [.0],
        "pretrained_cnn": [True],
        "multiple_frames": [True],
        "augment_frames": [False],
        "normalize_features": [True],
        "fix_temperature": [True],
        "temperature": [0.07],
        "gpus": [1],
        "num_workers": [8],
        "batch_size": [8],
        "drop_last": [True],
        "optimizer": ["AdamW"],
        "lr": [1e-4],
        "lr_scheduler": [True],
        "weight_decay": [0.1],
        "val_batch_size": [16],
        "eval_include_sos_eos": [True],
        "seed": [0, 1, 2],
        "optimize_unused": [True],
        "max_epochs": [400],
        "check_val_every_n_epoch": [1],
        "checkpoint_callback": ["True"],
        "logger": ["True"]
    },
]
# ordered flags to display in jobname
flags = [
    "text_encoder",
    "embedding_dim",
    "batch_size",
    "dropout_i",
    "augment_frames",
    "fix_temperature",
    "lr",
    "lr_scheduler",
    "weight_decay",
    "max_epochs",
    "seed",
]
