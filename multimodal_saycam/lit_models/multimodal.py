import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

LR = 3e-4

class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for MultiModal SAYCam model
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        train_loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)
        
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # batch of image-text pairs (same as training)
            x, y, y_len = batch

            logits_per_image, logits_per_text = self(x, y, y_len)
     
            # create ground truth labels
            batch_size = x.size(0)
            ground_truth = torch.tensor(np.arange(batch_size), dtype=torch.long, device=self.device)
     
            # calculate infonce loss
            val_loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)
            
            # self.log("val_loss", val_loss, on_step=False, on_epoch=True)
            return val_loss
        elif dataloader_idx == 1:
            # batch of evaluation trials
            x, y, y_len = batch

            # resize x so images from the same trial are in the batch dim
            # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
            x = x.view(-1, 3, 224, 224)  

            logits_per_image, logits_per_text = self.model(x, y, y_len)
            logits = logits_per_text[0]  # get logits per trial
            pred = torch.argmax(logits).item()

            label = 0  # correct answer is always the first item 

            if pred == label:
                val_accuracy = 1
            else:
                val_accuracy = 0

            # TODO: figure out how to log per-category accuracies separately
                
            # self.log("val_accuracy", val_accuracy, on_step=False, on_epoch=True) 
            return val_accuracy

    # def validation_epoch_end(self, validation_step_outputs):
    #     # collate validation results and log
    #     assert len(validation_step_outputs) == 2  # check we have results from both val dataloaders
        
    #     val_losses = validation_step_outputs[0]
    #     val_accuracies = validation_step_outputs[1]

    #     print(val_losses)
    #     print(val_accuracies)
