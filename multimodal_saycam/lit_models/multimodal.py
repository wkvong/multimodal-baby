import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

LR = 3e-4

class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Ligtning class for multimodal SAYCam model
    """

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr", LR)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=LR)
        # TODO: add argument for loss function type?
        
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer        

    def forward(self, x, y, y_len):
        return self.model(x, y, y_len)

    def training_step(self, batch, batch_idx):
        x, y, y_len = batch
        logits_per_image, logits_per_text = self(x, y, y_len)

        # create ground truth labels
        batch_size = x.size(0)
        ground_truth = torch.tensor(np.arange(batch_size), dtype=torch.long, device=self.device)

        # calculate infonce loss
        loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_len = batch
        logits_per_image, logits_per_text = self(x, y, y_len)

        # create ground truth labels
        batch_size = x.size(0)
        ground_truth = torch.tensor(np.arange(batch_size), dtype=torch.long, device=self.device)

        # calculate infonce loss
        loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)
        
        self.log("val_loss", loss)
        return loss
