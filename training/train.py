import argparse

import numpy as np
import torch
import pytorch_lightning as pl

from multimodal_saycam.data.multimodal_data_module import MultiModalSAYCamDataModule
from multimodal_saycam.models.multimodal_cnn_embedding import MultiModalCNNEmbedding
from multimodal_saycam.lit_models.multimodal import MultiModalLitModel


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
    MultiModalCNNEmbedding.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_model_group)

    return parser

def main():
    # set random seed
    pl.seed_everything(0)  # TODO: change seed to be a command line argument

    # parse args, set up data module and models
    parser = _setup_parser()
    args = parser.parse_args()
    data = MultiModalSAYCamDataModule(args)
    model = MultiModalCNNEmbedding(args)
    lit_model = MultiModalLitModel(model, args)

    # TOOD: add logging

    # create trainer
    trainer = pl.Trainer.from_argparse_args(args)

    print(args)

    # fit model
    trainer.fit(lit_model, data)
    
if __name__ == "__main__":
    main()
