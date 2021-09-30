import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from multimodal.multimodal_data_module import MultiModalSAYCamDataModule, read_vocab
from multimodal.multimodal import MultiModalModel
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
    MultiModalSAYCamDataModule.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    MultiModalModel.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--exp_name", type=str, default="multimodal_test",
                        help="experiment name for logging")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for everything")

    return parser

def main():
    # parse args
    parser = _setup_parser()
    args = parser.parse_args()

    # set random seed
    pl.seed_everything(args.seed)

    # set up data module and models
    data = MultiModalSAYCamDataModule(args)
    args.input_dim = len(read_vocab())
    model = MultiModalModel(args)
    lit_model = MultiModalLitModel(model, args)

    # create trainer (with logger if specified)
    if args.logger:
        # add wandb logging
        wandb_logger = WandbLogger(project='multimodal-saycam', name=args.exp_name,
                                   log_model=True)
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    else:
        trainer = pl.Trainer.from_argparse_args(args)

    print(args)

    # fit model
    trainer.fit(lit_model, data)
    
if __name__ == "__main__":
    main()
