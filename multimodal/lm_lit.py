import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from multimodal.multimodal_data_module import read_vocab, PAD_TOKEN_ID

LR = 3e-4

class LMLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for LM model
    """

    def __init__(self, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr", LR)

        self.text_encoder = text_encoder

        # load vocab and create dict to map indices back to words
        self.vocab = read_vocab()
        self.word2idx = self.vocab
        self.idx2word = dict((v,k) for k,v in self.vocab.items())

        # build output layer
        self.output_layer = nn.Linear(self.text_encoder.hidden_dim, len(self.vocab), bias=self.args.get("bias", True))
        if self.args.get("tie", True):
            self.output_layer.weight = self.text_encoder.embedding.weight

        # save hyperparameters to logger
        self.save_hyperparameters()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=LR, help="learning rate")
        parser.add_argument("--tie", type=lambda s: bool(eval(s)), default=True, help="whether to tie the input embedding and output layer matrix")
        parser.add_argument("--bias", type=lambda s: bool(eval(s)), default=True, help="whether to use bias for output layer")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer        

    def forward(self, y, y_len):
        outputs = self.text_encoder(y, y_len)
        logits = self.output_layer(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, y_len = batch
        logits = self(y, y_len)

        # calculate CE loss
        if self.text_encoder.text_encoder in ['cbow', 'bert']:
            labels = y
        else:
            logits = logits[:, :-1]
            labels = y[:, 1:1+logits.size(1)]
        loss = F.cross_entropy(logits.transpose(-2, -1), labels, ignore_index=PAD_TOKEN_ID, reduction="mean")
        mean_perplexity = loss.exp()

        # log train loss and temperature
        self.log("ce_loss", loss)
        self.log("perplexity", mean_perplexity)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # batch of image-text pairs (same as training)
            x, y, y_len = batch

            logits = self(y, y_len)
     
            # calculate CE loss
            if self.text_encoder.text_encoder in ['cbow', 'bert']:
                labels = y
            else:
                logits = logits[:, :-1]
                labels = y[:, 1:1+logits.size(1)]
            loss = F.cross_entropy(logits.transpose(-2, -1), labels, ignore_index=PAD_TOKEN_ID, reduction="mean")
            mean_perplexity = loss.exp()

            self.log("val_ce_loss", loss, on_step=False, on_epoch=True)
            self.log("val_perplexity", mean_perplexity, on_step=False, on_epoch=True)

            return loss

        elif dataloader_idx == 1:
            # batch of evaluation trials (only one trial at a time)
            x, y, y_len = batch
