import argparse
from pathlib import Path
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

LR = 3e-4
SELF_DISTILLATION = False
ALPHA = 1
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
VOCAB_FILENAME = DATA_DIR / "vocab.json"

class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for MultiModal SAYCam model
    """

    def __init__(self, model, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr", LR)

        self.model = model

        # self-distillation
        self.self_distillation = self.args.get("self_distillation", SELF_DISTILLATION)
        self.teacher = copy.deepcopy(self.model)
        self.alpha = self.args.get("alpha", ALPHA)

        # set teacher to be non-trainable
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # load vocab and create dict to map indices back to words
        with open(VOCAB_FILENAME) as f:
            self.vocab = json.load(f)
            self.word2idx = self.vocab
            self.idx2word = dict((v,k) for k,v in self.vocab.items())

        # save hyperparameters to logger
        self.save_hyperparameters()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=LR, help="learning rate")
        parser.add_argument("--self_distillation", action='store_true',
                            help="include self-distillation loss during training")
        parser.add_argument("--alpha", type=float, default=1.0,
                            help="coefficient for KLdiv loss in self-distillation")
        
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
        infonce_loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)

        # calculate accuracy (image and text separately)
        train_image_pred = torch.argmax(logits_per_image, dim=-1)
        train_text_pred = torch.argmax(logits_per_text, dim=-1)
        train_image_accuracy = (train_image_pred == ground_truth).sum() / batch_size
        train_text_accuracy = (train_text_pred == ground_truth).sum() / batch_size

        # calculate self-distillation loss
        kl_loss = 0
        if self.self_distillation:
            # get teacher targets and student predictions
            teacher_logits_per_image, teacher_logits_per_text = self.teacher(
                x, y, y_len, self_distillation=True, teacher=True)
            student_logits_per_image, student_logits_per_text = self.model(
                x, y, y_len, self_distillation=True, teacher=False)

            # calculate kl div loss 
            kl_loss = (F.kl_div(F.log_softmax(student_logits_per_image, dim=-1), teacher_logits_per_image, reduction='batchmean') + F.kl_div(F.log_softmax(student_logits_per_text, dim=-1), teacher_logits_per_text, reduction='batchmean')).div(2) * self.alpha

            # update teacher model via ema
            self.update_teacher()

        # calculate joint train loss
        train_loss = infonce_loss + kl_loss
        
        # log train loss and temperature
        self.log("train_loss", train_loss)
        self.log("train_infonce_loss", infonce_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_image_accuracy", train_image_accuracy)
        self.log("train_text_accuracy", train_text_accuracy)
        self.log("temperature", self.model.logit_scale.item())
        self.log("kl_temperature", self.model.kl_logit_scale.item())
        
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

            # calculate accuracy (image and text separately)
            val_image_pred = torch.argmax(logits_per_image, dim=-1)
            val_text_pred = torch.argmax(logits_per_text, dim=-1)
            val_image_accuracy = (val_image_pred == ground_truth).sum() / batch_size
            val_text_accuracy = (val_text_pred == ground_truth).sum() / batch_size

            self.log("val_loss", val_loss, on_step=False, on_epoch=True)
            self.log("val_image_accuracy", val_image_accuracy, on_step=False, on_epoch=True)
            self.log("val_text_accuracy", val_text_accuracy, on_step=False, on_epoch=True)
            
            return val_loss
        elif dataloader_idx == 1:
            # batch of evaluation trials (only one trial at a time)
            x, y, y_len = batch

            # resize x so images from the same trial are in the batch dim
            # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
            x = x.view(-1, 3, 224, 224)  

            # calculate accuracy
            logits_per_image, logits_per_text = self.model(x, y, y_len)
            logits = logits_per_text[0]  # get logits per trial
            pred = torch.argmax(logits).item()
            label = 0  # correct answer is always the first item 

            if pred == label:
                val_accuracy = 1
            else:
                val_accuracy = 0

            # log evaluation accuracy
            self.log("val_accuracy", val_accuracy, on_step=False, on_epoch=True)

            # log category-level evaluation accuracies as a separate metric
            category_label = self.idx2word[y.item()]
            self.log(f"val_accuracy_{category_label}", val_accuracy, on_step=False, on_epoch=True)

            return val_accuracy

    def update_teacher(self):
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.data.copy_(self.ema(teacher.data, student.data))

    def ema(self, s, t):
        return s * (1 - 0.999) + t * 0.999
