import argparse
import functools
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from multimodal.multimodal import MultiModalModel, LanguageModel
from multimodal.utils import get_entropy

OPTIMIZER = torch.optim.AdamW
LR = 3e-4
WEIGHT_DECAY = 0.01
# SELF_DISTILLATION = False
# ALPHA = 1


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
        # self.alpha = self.args.get("alpha", ALPHA)
        self.lambda_mm = self.args.get("lambda_mm", 1.)
        self.lambda_lm = self.args.get("lambda_lm", 0.)
        self.optimize_unused = self.args.get("optimize_unused", False)

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.model = MultiModalModel(
            self.vision_encoder, self.text_encoder, args)
        self.language_model = LanguageModel(self.text_encoder, args)

        # self-distillation
        # self.self_distillation = self.args.get(
        #     "self_distillation", SELF_DISTILLATION)

        # if self.self_distillation:
        #     # only instantiate teacher model if self-distillation is on
        #     self.teacher = copy.deepcopy(self.model)

        #     # set teacher to be non-trainable
        #     for param in self.teacher.parameters():
        #         param.requires_grad = False

        # save hyperparameters to logger
        self.save_hyperparameters()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=lambda o: getattr(torch.optim, o), default=OPTIMIZER,
                            help="optimizer class under toch.optim")
        parser.add_argument("--lr", type=float, default=LR,
                            help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="weight decay on all parameters")
        parser.add_argument("--lambda_mm", type=float, default=1.,
                            help="multimodal loss *= lambda_mm")
        parser.add_argument("--lambda_lm", type=float, default=0.,
                            help="lm loss *= lambda_lm")
        parser.add_argument("--optimize_unused", action="store_true",
                            help="optimize the computation for unused loss")
        # parser.add_argument("--self_distillation", action='store_true',
        #                     help="include self-distillation loss during training")
        # parser.add_argument("--alpha", type=float, default=1.0,
        #                     help="coefficient for KLdiv loss in self-distillation")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x, y, y_len):
        return self.model(x, y, y_len)

    # def calculate_self_distillation_loss(self, x, y, y_len):
    #     # get teacher targets and student predictions
    #     teacher_logits_per_image, teacher_logits_per_text = self.teacher(
    #         x, y, y_len, self_distillation=True, teacher=True)
    #     student_logits_per_image, student_logits_per_text = self.model(
    #         x, y, y_len, self_distillation=True, teacher=False)

    #     # calculate kl div loss
    #     kl_loss = (F.kl_div(F.log_softmax(student_logits_per_image, dim=-1), teacher_logits_per_image, reduction='batchmean') +
    #                F.kl_div(F.log_softmax(student_logits_per_text, dim=-1), teacher_logits_per_text, reduction='batchmean')).div(2) * self.alpha

    #     # update teacher model via ema
    #     self.update_teacher()

    #     return kl_loss

    def calculate_joint_loss(self, batch, stage, log):
        # batch of image-text pairs
        x, y, y_len = batch

        if self.lambda_mm or not self.optimize_unused:
            infonce_loss, image_accuracy, text_accuracy, \
            image_entropy, text_entropy, logits_per_image, logits_per_text = \
            self.model.calculate_contrastive_loss(x, y, y_len)

            # if self.self_distillation:
            #     kl_loss = self.calculate_self_distillation_loss(x, y, y_len)
            # else:
            #     kl_loss = 0.

            # log
            log(f"{stage}_infonce_loss", infonce_loss)
            # log(f"{stage}_kl_loss", kl_loss)
            log(f"{stage}_image_accuracy", image_accuracy)
            log(f"{stage}_text_accuracy", text_accuracy)
            log(f"{stage}_image_entropy", image_entropy)
            log(f"{stage}_text_entropy", text_entropy)
            log("temperature",
                     (-self.model.logit_neg_log_temperature).exp().item())
            # log("kl_temperature", (-self.model.kl_logit_neg_log_temperature).exp().item())

        else:
            infonce_loss = 0.

        if self.lambda_lm or not self.optimize_unused:
            # calculate language model ce loss
            lm_ce_loss, perplexity, _, _, _ = \
                self.language_model.calculate_ce_loss(y, y_len)

            # log
            log(f"{stage}_ce_loss", lm_ce_loss)
            log(f"{stage}_perplexity", perplexity)

        else:
            lm_ce_loss = 0.

        # calculate joint loss
        loss = self.lambda_mm * infonce_loss + self.lambda_lm * lm_ce_loss

        # log
        log(f"{stage}_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_joint_loss(batch, 'train', self.log)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        stage = 'val'
        log = functools.partial(self.log, on_step=False, on_epoch=True)

        if dataloader_idx == 0:
            return self.calculate_joint_loss(batch, stage, log)

        elif dataloader_idx == 1:
            # TODO: check whether adding special tokens will make a difference

            # batch of evaluation trials (only one trial at a time)
            x, y, y_len = batch

            if self.lambda_mm or not self.optimize_unused:
                # resize x so images from the same trial are in the batch dim
                # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
                x = x.view(-1, *x.shape[-3:])

                # calculate accuracy
                logits_per_image, logits_per_text = self.model(x, y, y_len)
                logits = logits_per_text[0]  # get logits per trial
                pred = torch.argmax(logits).item()
                label = 0  # correct answer is always the first item
                accuracy = int(pred == label)
                entropy = get_entropy(logits)

                # log evaluation accuracy and entropy
                log(f"{stage}_accuracy", accuracy)
                log(f"{stage}_entropy", entropy)

                # log category-level evaluation accuracies as a separate metric
                category_label = self.text_encoder.idx2word[y.item()]
                log(f"{stage}_accuracy_{category_label}", accuracy)

            else:
                accuracy = 0.

            return accuracy

    # def update_teacher(self):
    #     for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
    #         teacher.data.copy_(self.ema(teacher.data, student.data))

    # def ema(self, s, t):
    #     return s * (1 - 0.999) + t * 0.999
