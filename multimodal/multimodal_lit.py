import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from multimodal.multimodal import MultiModalModel, LanguageModel
from multimodal.multimodal_data_module import read_vocab
from utils import get_entropy

OPTIMIZER = torch.optim.AdamW
LR = 3e-4
WEIGHT_DECAY = 0.01
SELF_DISTILLATION = False
ALPHA = 1

class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for MultiModal SAYCam model
    """

    def __init__(self, vision_encoder, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.optimizer_class = self.args.get("optimizer", OPTIMIZER)
        self.lr = self.args.get("lr", LR)
        self.weight_decay = self.args.get("weight_decay", WEIGHT_DECAY)
        self.lambda_mm = self.args.get("lambda_mm", 1.)
        self.lambda_lm = self.args.get("lambda_lm", 0.)

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.model = MultiModalModel(self.vision_encoder, self.text_encoder, args)
        self.language_model = LanguageModel(self.text_encoder, args)

        # self-distillation
        self.self_distillation = self.args.get("self_distillation", SELF_DISTILLATION)
        self.teacher = copy.deepcopy(self.model)
        self.alpha = self.args.get("alpha", ALPHA)

        # set teacher to be non-trainable
        for param in self.teacher.parameters():
            param.requires_grad = False

        # save hyperparameters to logger
        self.save_hyperparameters()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=lambda o: getattr(torch.optim, o), default=OPTIMIZER,
                            help="optimizer class under toch.optim")
        parser.add_argument("--lr", type=float, default=LR, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="weight decay on all parameters")
        parser.add_argument("--lambda_mm", type=float, default=1.,
                            help="multimodal loss *= lambda_mm")
        parser.add_argument("--lambda_lm", type=float, default=0.,
                            help="lm loss *= lambda_lm")
        parser.add_argument("--self_distillation", action='store_true',
                            help="include self-distillation loss during training")
        parser.add_argument("--alpha", type=float, default=1.0,
                            help="coefficient for KLdiv loss in self-distillation")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
        train_image_entropy = get_entropy(logits_per_image, dim=-1).mean()
        train_text_entropy = get_entropy(logits_per_text, dim=-1).mean()

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

        # calculate language model ce loss
        lm_ce_loss, perplexity, _, _, _ = self.language_model.calculate_ce_loss(y, y_len)

        # calculate joint train loss
        train_loss = self.lambda_mm * (infonce_loss + kl_loss) + self.lambda_lm * lm_ce_loss
        
        # log train loss and temperature
        self.log("train_loss", train_loss)
        self.log("train_infonce_loss", infonce_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_image_accuracy", train_image_accuracy)
        self.log("train_text_accuracy", train_text_accuracy)
        self.log("train_image_entropy", train_image_entropy)
        self.log("train_text_entropy", train_text_entropy)
        self.log("temperature", self.model.logit_scale.item())
        self.log("kl_temperature", self.model.kl_logit_scale.item())
        self.log("ce_loss", lm_ce_loss)
        self.log("perplexity", perplexity)
        
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
            val_image_entropy = get_entropy(logits_per_image, dim=-1).mean()
            val_text_entropy = get_entropy(logits_per_text, dim=-1).mean()

            # calculate language model ce loss
            lm_ce_loss, perplexity, _, _, _ = self.language_model.calculate_ce_loss(y, y_len)

            self.log("val_loss", val_loss, on_step=False, on_epoch=True)
            self.log("val_image_accuracy", val_image_accuracy, on_step=False, on_epoch=True)
            self.log("val_text_accuracy", val_text_accuracy, on_step=False, on_epoch=True)
            self.log("val_image_entropy", val_image_entropy, on_step=False, on_epoch=True)
            self.log("val_text_entropy", val_text_entropy, on_step=False, on_epoch=True)
            self.log("val_ce_loss", lm_ce_loss, on_step=False, on_epoch=True)
            self.log("val_perplexity", perplexity, on_step=False, on_epoch=True)
            
            return val_loss
        elif dataloader_idx == 1:
            # batch of evaluation trials (only one trial at a time)
            x, y, y_len = batch

            # resize x so images from the same trial are in the batch dim
            # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
            x = x.view(-1, *x.shape[-3:])

            # calculate accuracy
            logits_per_image, logits_per_text = self.model(x, y, y_len)
            logits = logits_per_text[0]  # get logits per trial
            pred = torch.argmax(logits).item()
            label = 0  # correct answer is always the first item 
            val_accuracy = int(pred == label)
            val_entropy = get_entropy(logits)

            # log evaluation accuracy and entropy
            self.log("val_accuracy", val_accuracy, on_step=False, on_epoch=True)
            self.log("val_entropy", val_entropy, on_step=False, on_epoch=True)

            # log category-level evaluation accuracies as a separate metric
            category_label = self.text_encoder.idx2word[y.item()]
            self.log(f"val_accuracy_{category_label}", val_accuracy, on_step=False, on_epoch=True)

            return val_accuracy

    def update_teacher(self):
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.data.copy_(self.ema(teacher.data, student.data))

    def ema(self, s, t):
        return s * (1 - 0.999) + t * 0.999
