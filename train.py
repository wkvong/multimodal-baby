import argparse
from pathlib import Path
import json
import os

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from multimodal.multimodal_data_module import MultiModalDataModule
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from multimodal.multimodal import VisionEncoder, TextEncoder, MultiModalModel, LanguageModel
from multimodal.multimodal_lit import MultiModalLitModel


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser()

    # add trainer specific arguments
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # get data, model and litmodel specific arguments
    data_group = parser.add_argument_group("Data Args")
    MultiModalDataModule.add_to_argparse(data_group)
    MultiModalSAYCamDataModule.add_additional_to_argparse(data_group)
    COCOCaptionsDataModule.add_additional_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    VisionEncoder.add_to_argparse(model_group)
    TextEncoder.add_to_argparse(model_group)
    MultiModalModel.add_to_argparse(model_group)
    LanguageModel.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--exp_name", type=str, default="multimodal_test",
                        help="experiment name for logging")
    parser.add_argument("--dataset", type=str, choices=["saycam", "coco"],
                        default="saycam",
                        help="which dataset to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for everything")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="saves best k models; 0 saves none; -1 saves all")
    parser.add_argument("--resume_ckpt", type=Path, default=None,
                        help="path to the checkpoint to resume from; if it's "
                             "\"last\", resume from the last checkpoint.")
    parser.add_argument('--save_every_n_steps', type=int, default=50)

    return parser


def main():
    # parse args
    parser = _setup_parser()
    args = parser.parse_args()

    # checkpoint paths
    ckpt_dir = Path('/scratch/nk3351/projects/multimodal-baby/checkpoints') / args.exp_name
    if str(args.resume_ckpt) == "last":
        args.resume_ckpt = ckpt_dir / 'last.ckpt'

    # set random seed
    pl.seed_everything(args.seed)

    # set up data module and models
    DataModuleClass = {
        "saycam": MultiModalSAYCamDataModule,
        "coco": COCOCaptionsDataModule,
    }[args.dataset]
    data = DataModuleClass(args)
    vocab = data.read_vocab()
    vision_encoder = VisionEncoder(args=args)
    text_encoder = TextEncoder(
        vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
    lit_model = MultiModalLitModel(vision_encoder, text_encoder, args)

    # Save config and word2idx for AoA eval to ckpt_dir
    args_d = {}
    for key, val in args.__dict__.items():
        if type(val) == type:
            args_d[key] = str(val)
        else:
            args_d[key] = val

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(args_d, f, indent=2)
    with open(os.path.join(ckpt_dir, 'word2idx.json'), 'w') as f:
        json.dump(text_encoder.word2idx, f)

    # setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        # monitor='val_loss',
        # save_last=True,
        save_top_k=args.save_top_k,
        dirpath=ckpt_dir,
        every_n_train_steps=args.save_every_n_steps,
        filename='{step}')

    # create trainer (with checkpoint and logger if specified)
    if args.logger:
        # add checkpoint callback and wandb logging
        wandb_logger = WandbLogger(project='multimodal-saycam', name=args.exp_name,
                                   log_model=True)
        trainer = pl.Trainer.from_argparse_args(args,
                                                enable_checkpointing=args.checkpoint_callback,
                                                callbacks=[
                                                    checkpoint_callback],
                                                logger=wandb_logger)
    else:
        trainer = pl.Trainer.from_argparse_args(args)

    print(args)
    print(ckpt_dir)

    # fit model
    trainer.fit(lit_model, data, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
