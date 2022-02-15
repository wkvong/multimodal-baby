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
from multimodal.textgen_eval import evaluate as textgen_eval
from multimodal.multimodal_data_module import \
    N_VAL_DATALOADERS_PER_SPLIT, MAX_LEN_UTTERANCE, \
    PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID

OPTIMIZER = torch.optim.AdamW
LR = 3e-4
FACTOR = 0.1
PATIENCE = 20
WEIGHT_DECAY = 0.01

# text generation evaluation arguments
BEAM_WIDTH = 3
DECODE_LENGTH = MAX_LEN_UTTERANCE
LENGTH_PENALTY_ALPHA = 0.0
# print arguments
PRINT_EVAL_TEXTGEN_EXAMPLE_IDS = range(10)

class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for MultiModal SAYCam model
    """

    def __init__(self, vision_encoder, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.optimizer_class = self.args.get("optimizer", OPTIMIZER)
        self.lr = self.args.get("lr", LR)
        self.lr_scheduler = self.args.get("lr_scheduler", False)
        self.factor = self.args.get("factor", FACTOR)
        self.patience = self.args.get("patience", PATIENCE)
        self.weight_decay = self.args.get("weight_decay", WEIGHT_DECAY)
        # self.alpha = self.args.get("alpha", ALPHA)
        self.lambda_mm = self.args.get("lambda_mm", 1.)
        self.lambda_lm = self.args.get("lambda_lm", 0.)
        self.optimize_unused = self.args.get("optimize_unused", False)
        self.eval_textgen = self.args.get("eval_textgen", False)
        self.beam_width = self.args.get("beam_width", BEAM_WIDTH)
        self.decode_length = self.args.get("decode_length", DECODE_LENGTH)
        self.length_penalty_alpha = self.args.get(
            "length_penalty_alpha", LENGTH_PENALTY_ALPHA)

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.model = MultiModalModel(
            self.vision_encoder, self.text_encoder, args)
        self.language_model = LanguageModel(self.text_encoder, args)

        # save hyperparameters to logger
        self.save_hyperparameters()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=lambda o: getattr(torch.optim, o), default=OPTIMIZER,
                            help="optimizer class under torch.optim")
        parser.add_argument("--lr", type=float, default=LR,
                            help="learning rate")
        parser.add_argument("--lr_scheduler", action="store_true",
                            help="use ReduceLROnPlateau lr scheduler")
        parser.add_argument("--factor", type=float, default=FACTOR,
                            help="factor by which the learning rate will be "
                                 "reduced")
        parser.add_argument("--patience", type=int, default=PATIENCE,
                            help="number of epochs with no improvement after "
                                 "which learning rate will be reduced")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="weight decay on all parameters")
        parser.add_argument("--lambda_mm", type=float, default=1.,
                            help="multimodal contrastive loss *= lambda_mm")
        parser.add_argument("--lambda_lm", type=float, default=0.,
                            help="language modeling loss *= lambda_lm")
        parser.add_argument("--optimize_unused", action="store_true",
                            help="optimize the computation for unused loss "
                                 "(i.e., lambda=0)")
        parser.add_argument("--eval_textgen", action="store_true",
                            help="evaluate text generation")
        parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH,
                            help="beam width in beam search text generation")
        parser.add_argument("--decode_length", type=int, default=DECODE_LENGTH,
                            help="beam search maximum decode length")
        parser.add_argument("--length_penalty_alpha", type=float,
                            default=LENGTH_PENALTY_ALPHA,
                            help="beam search length penalty (alpha); "
                                 "0 for no length penalty.")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if not self.lr_scheduler:
            return optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            }
        }

    def forward(self, x, y, y_len):
        return self.model(x, y, y_len)

    def calculate_joint_loss(self, batch, stage, log, eval_textgen=False):
        # batch of image-text pairs
        x, y, y_len, raw_y = batch

        # dict of results to return
        ret = {
            'batch_size': x.size(0),
        }

        # reuse image_features and text_outputs if possible
        image_features, text_outputs = None, None

        if self.lambda_mm or not self.optimize_unused:
            infonce_loss, image_accuracy, text_accuracy, \
                image_entropy, text_entropy, logits_per_image, logits_per_text, \
                image_features, text_outputs = \
                self.model.calculate_contrastive_loss(x, y, y_len)

            # log
            log(f"{stage}_infonce_loss", infonce_loss)
            log(f"{stage}_image_accuracy", image_accuracy)
            log(f"{stage}_text_accuracy", text_accuracy)
            log(f"{stage}_image_entropy", image_entropy)
            log(f"{stage}_text_entropy", text_entropy)
            log("temperature",
                (-self.model.logit_neg_log_temperature).exp().item())

            ret.update({
                'infonce_loss': infonce_loss.detach(),
                'image_accuracy': image_accuracy,
                'text_accuracy': text_accuracy,
                'image_entropy': image_entropy.detach(),
                'text_entropy': text_entropy.detach(),
            })

        else:
            infonce_loss = 0.

        if self.lambda_lm or not self.optimize_unused:
            if self.language_model.text_encoder.captioning:
                # get image_features if needed
                if image_features is None:
                    image_features = self.vision_encoder(x)
                    if self.model.normalize_features:
                        image_features = F.normalize(image_features, p=2, dim=1)  # normalize image features
                # text_outputs is not reusable since it's not obtained from captioning in the contrastive module
                text_outputs = None

            # calculate language model ce loss
            ce_loss, _, _, labels = self.language_model.calculate_ce_loss(
                y, y_len, outputs=text_outputs, image_features=image_features,
                tokenwise=True)

            # get all kinds of losses with/without special tokens
            # standard loss including all special tokens
            mask = (labels != PAD_TOKEN_ID)
            n_tokens = mask.sum()
            lm_ce_loss = ce_loss.sum() / n_tokens
            # excluding SOS_TOKEN
            mask = mask & (labels != SOS_TOKEN_ID)
            n_tokens_wo_sos = mask.sum()
            lm_ce_loss_wo_sos = (ce_loss * mask).sum() / n_tokens_wo_sos
            # further excluding EOS_TOKEN
            mask = mask & (labels != EOS_TOKEN_ID)
            n_tokens_wo_sos_eos = mask.sum()
            lm_ce_loss_wo_sos_eos = (ce_loss * mask).sum() / n_tokens_wo_sos_eos

            # log
            log(f"{stage}_ce_loss", lm_ce_loss)
            log(f"{stage}_ce_loss_wo_sos", lm_ce_loss_wo_sos)
            log(f"{stage}_ce_loss_wo_sos_eos", lm_ce_loss_wo_sos_eos)

            ret.update({
                'ce_loss': lm_ce_loss.detach(),
                'ce_loss_wo_sos': lm_ce_loss_wo_sos.detach(),
                'ce_loss_wo_sos_eos': lm_ce_loss_wo_sos_eos.detach(),
                'n_tokens': n_tokens,
                'n_tokens_wo_sos': n_tokens_wo_sos,
                'n_tokens_wo_sos_eos': n_tokens_wo_sos_eos,
            })

            if eval_textgen:
                beam_seq, log_prob = self.language_model.beam_search_decode(
                    batch_size=ret['batch_size'],
                    beam_width=self.beam_width,
                    decode_length=self.decode_length,
                    length_penalty_alpha=self.length_penalty_alpha,
                    image_features=image_features
                        if self.language_model.text_encoder.captioning else
                        None,
                )

                def ids_to_sentence(y):
                    y = y.tolist()
                    y_len = 0
                    while y_len < len(y) and y[y_len] != PAD_TOKEN_ID:
                        y_len += 1
                    y = y[:y_len]
                    if len(y) > 0 and y[-1] == EOS_TOKEN_ID:
                        y = y[:-1]
                    if len(y) > 0 and y[0] == SOS_TOKEN_ID:
                        y = y[1:]
                    return ' '.join(
                        self.text_encoder.idx2word[idx] for idx in y)

                gen_text_ids = beam_seq[:, 0]
                gen_text = [ids_to_sentence(y) for y in gen_text_ids]

                ret.update({
                    'raw_y': raw_y,
                    'gen_text': gen_text,
                })

        else:
            lm_ce_loss = 0.

        # calculate joint loss
        loss = self.lambda_mm * infonce_loss + self.lambda_lm * lm_ce_loss

        # log
        log(f"{stage}_loss", loss)

        ret.update({
            'loss': loss,
        })

        return ret

    def joint_loss_epoch_end(self, outputs, stage, log, eval_textgen=False):
        def mean_over_examples(name):
            # mean over examples
            n_examples = 0
            value_sum = 0.
            for output in outputs:
                batch_size = output['batch_size']
                value = output[name].item()
                n_examples += batch_size
                value_sum += value * batch_size
            value_mean = value_sum / n_examples
            return value_mean

        def mean_over_tokens(name, n_tokens_name):
            # mean over tokens
            n_tokens_sum = 0
            value_sum = 0.
            for output in outputs:
                n_tokens = output[n_tokens_name].item()
                value = output[name].item()
                n_tokens_sum += n_tokens
                value_sum += value * n_tokens
            value_mean = value_sum / n_tokens_sum
            return value_mean

        if self.lambda_mm or not self.optimize_unused:
            for name in (
                'infonce_loss', 'image_accuracy', 'text_accuracy',
                'image_entropy', 'text_entropy',):
                log(f"{stage}_{name}", mean_over_examples(name))

        if self.lambda_lm or not self.optimize_unused:
            for suffix in ('', '_wo_sos', '_wo_sos_eos'):
                value_mean = mean_over_tokens(
                    f'ce_loss{suffix}', f'n_tokens{suffix}')
                log(f"{stage}_ce_loss{suffix}", value_mean)

                # perplexity
                perplexity = np.exp(value_mean)
                log(f"{stage}_perplexity{suffix}", perplexity)

            if eval_textgen:
                list_of_references, hypotheses = [], []
                for output in outputs:
                    list_of_references += output['raw_y']
                    hypotheses += output['gen_text']

                for example_id in PRINT_EVAL_TEXTGEN_EXAMPLE_IDS:
                    print(f"example #{example_id}:")
                    references = list_of_references[example_id]
                    hypothesis = hypotheses[example_id]
                    print("references:")
                    print("\n".join(references))
                    print("hypothesis:")
                    print(hypothesis)

                score_dict = textgen_eval(list_of_references, hypotheses)

                for metric, score in score_dict.items():
                    log(f"{stage}_{metric}", score)

        for name in ('loss',):
            log(f"{stage}_{name}", mean_over_examples(name))

    def training_step(self, batch, batch_idx):
        return self.calculate_joint_loss(
            batch, 'train', self.log, eval_textgen=False)

    def training_epoch_end(self, outputs):
        log = lambda name, value, *args, **kwargs: self.log(
            f'{name}_epoch', value, on_step=False, on_epoch=True,
            *args, **kwargs)
        return self.joint_loss_epoch_end(
            outputs, 'train', log, eval_textgen=False)

    def validation_test_step(self, stage, batch, batch_idx, dataloader_idx=0):
        log = functools.partial(self.log, on_step=False, on_epoch=True)

        ret = {}

        if dataloader_idx == 0:
            empty_log = lambda *args, **kwargs: None
            ret.update(self.calculate_joint_loss(
                batch, stage, empty_log, eval_textgen=self.eval_textgen))

        elif dataloader_idx == 1:
            # TODO: check whether adding special tokens will make a difference

            # batch of evaluation trials (only one trial at a time)
            x, y, y_len, raw_y = batch

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

                ret.update({'accuracy': accuracy})

            else:
                accuracy = 0.

        return ret

    def validation_test_epoch_end(self, stage, outputs):
        # only deal with outputs of the first dataset
        log = functools.partial(self.log, on_step=False, on_epoch=True)
        return self.joint_loss_epoch_end(
            outputs[0], stage, log, eval_textgen=self.eval_textgen)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx < N_VAL_DATALOADERS_PER_SPLIT:  # as normal
            return self.validation_test_step(
                'val', batch, batch_idx, dataloader_idx=dataloader_idx)
        else:  # actually a test_step
            return self.test_step(
                batch, batch_idx,
                dataloader_idx=dataloader_idx - N_VAL_DATALOADERS_PER_SPLIT)

    def validation_epoch_end(self, outputs):
        self.validation_test_epoch_end(
            'val', outputs[:N_VAL_DATALOADERS_PER_SPLIT])
        if len(outputs) > N_VAL_DATALOADERS_PER_SPLIT:
            self.test_epoch_end(outputs[N_VAL_DATALOADERS_PER_SPLIT:])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_test_step(
            'test', batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs):
        return self.validation_test_epoch_end(
            'test', outputs)