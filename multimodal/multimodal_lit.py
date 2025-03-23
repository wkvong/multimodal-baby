import argparse
import functools
import copy
import json
import numpy as np
import os
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from multimodal.multimodal import MultiModalModel, LanguageModel, \
    calculate_attn_reg_loss
from multimodal.utils import get_entropy
from multimodal.textgen_eval import evaluate as textgen_eval
from multimodal.multimodal_data_module import \
    N_VAL_DATALOADERS_PER_SPLIT, MAX_LEN_UTTERANCE, \
    PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID
from huggingface_hub import hf_hub_download

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
        self.lambda_ar = self.args.get("lambda_ar", 0.)
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

        # get vocab
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
        with open(self.vocab_path) as f:
            self.vocab = json.load(f)
        self.nlp = spacy.load("en_core_web_sm")

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
        parser.add_argument("--lambda_ar", type=float, default=0.,
                            help="attention regularization loss *= lambda_ar")
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

    @staticmethod
    def load_model(model_name="cvcl"):
        """Load pre-trained CVCL model from HuggingFace Hub"""
        if model_name == "cvcl":
            checkpoint_name = "cvcl_s_dino_resnext50_embedding"
            checkpoint = hf_hub_download(repo_id="wkvong/"+checkpoint_name, filename=checkpoint_name+".ckpt")
            model = MultiModalLitModel.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            raise ValueError("Model name not found.")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
        return model, preprocess

    def encode_image(self, x):
        """Encode images to obtain image features"""
        image_features, _ = self.model.encode_image(x)
        return image_features

    def encode_text(self, y, y_len=None):
        """Encode text to obtain text features"""
        text_features, _ = self.model.encode_text(y, y_len)
        return text_features

    def tokenize(self, texts):
        """Tokenize texts to obtain tokens and token lengths"""
        max_seq_len = 25

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        token_lengths = []

        for text in texts:
            doc = self.nlp(text)
            word_tokens = [token.text for token in doc]
               
            # Truncate if too long (leaving room for special tokens)
            if len(word_tokens) > max_seq_len - 2:  # -2 for <sos> and <eos>
                word_tokens = word_tokens[:max_seq_len - 2]
               
            # Calculate correct token length
            token_length = len(word_tokens) + 2  # +2 for <sos> and <eos>
               
            # Add special tokens and padding if necessary
            tokens = [self.vocab["<sos>"]] + [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens] + [self.vocab["<eos>"]] + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
               
            all_tokens.append(tokens)
            token_lengths.append(token_length)
        
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        token_lengths = torch.tensor(token_lengths, dtype=torch.long)
        return tokens, token_lengths
    
    def calculate_ce_loss(
        self, y, y_len, x=None,
        outputs=None,
        image_features=None,
        image_feature_map=None,
        return_image_features=False,
        **kwargs
    ):
        """Wraps self.language_model.calculate_ce_loss
        """
        if self.language_model.text_encoder.captioning or \
            self.language_model.text_encoder.has_attention:
            # get image_features and image_feature_map if needed
            if image_features is None:
                image_features, image_feature_map = self.model.encode_image(x)
            # text_outputs is not reusable since it's not obtained from
            # captioning in the contrastive module
            outputs = None
        else:
            image_features, image_feature_map = None, None

        # calculate language model ce loss
        ret = self.language_model.calculate_ce_loss(
            y, y_len,
            outputs=outputs,
            image_features=image_features
                if self.language_model.text_encoder.captioning else None,
            image_feature_map=image_feature_map
                if self.language_model.text_encoder.has_attention else None,
            **kwargs
        )
        if return_image_features:
            ret = ret + (image_features, image_feature_map)
        return ret

    def calculate_joint_loss(self, batch, stage, log, eval_textgen=False,
                             ce_weight=None):
        # batch of image-text pairs
        x, y, y_len, raw_y = batch

        # dict of results to return
        ret = {
            'batch_size': x.size(0),
        }

        # reuse image_features, image_feature_map and text_outputs if possible
        image_features, image_feature_map, text_outputs = None, None, None

        if self.lambda_mm or not self.optimize_unused:
            infonce_loss, image_accuracy, text_accuracy, \
                image_entropy, text_entropy, logits_per_image, logits_per_text, \
                image_features, image_feature_map, text_outputs = \
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
            # calculate language model ce loss
            ce_loss, _, _, attns, labels, image_features, image_feature_map = \
            self.calculate_ce_loss(
                y, y_len, x=x,
                outputs=text_outputs,
                image_features=image_features,
                image_feature_map=image_feature_map,
                return_image_features=True,
                tokenwise=True,
                weight=ce_weight,
            )

            # get all kinds of losses with/without special tokens
            # Actually in torch.nn.CrossEntropyLoss the sum of loss should be
            # divided by the sum of mask weighted by the weight. Here I ignored
            # the weight for simplicity, since it is not used in the main code.

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

            # attention regularization loss
            if self.language_model.text_encoder.has_attention:
                attn_reg_loss = calculate_attn_reg_loss(attns)

                # log
                log(f"{stage}_attn_reg_loss", attn_reg_loss)

                ret.update({
                    'attn_reg_loss': attn_reg_loss.detach(),
                })

            else:
                attn_reg_loss = 0.

            if eval_textgen:
                beam_seq, log_prob = self.language_model.beam_search_decode(
                    batch_size=ret['batch_size'],
                    beam_width=self.beam_width,
                    decode_length=self.decode_length,
                    length_penalty_alpha=self.length_penalty_alpha,
                    image_features=image_features
                        if self.language_model.text_encoder.captioning else
                        None,
                    image_feature_map=image_feature_map
                        if self.language_model.text_encoder.has_attention else
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
            attn_reg_loss = 0.

        # calculate joint loss
        loss = self.lambda_mm * infonce_loss + self.lambda_lm * lm_ce_loss \
            + self.lambda_ar * attn_reg_loss

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

            if self.language_model.text_encoder.has_attention:
                for name in ('attn_reg_loss',):
                    log(f"{stage}_{name}", mean_over_examples(name))

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
            # batch of evaluation trials (only one trial at a time)
            x, y, y_len, raw_y = batch

            # resize x so images from the same trial are in the batch dim
            # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
            x = x.view(-1, *x.shape[-3:])

            if self.lambda_mm:
                logits_per_image, logits_per_text = self.model(x, y, y_len)
                logits = logits_per_text[0]  # get logits per trial

            elif self.lambda_lm and (
                    self.language_model.text_encoder.captioning or
                    self.language_model.text_encoder.has_attention) \
                    and y[0, 0].item() == SOS_TOKEN_ID:
                # tile y to match the batch size
                y = y.expand(x.size(0), -1)
                y_len = y_len.expand(x.size(0))

                # calculate language model ce loss
                ce_loss, _, _, _, labels = self.calculate_ce_loss(
                    y, y_len, x=x, tokenwise=True)

                # use - ce_loss on the word as logits
                logits = - ce_loss[:, 0]

            else:
                logits = None

            if logits is not None:
                # calculate accuracy
                pred = torch.argmax(logits).item()
                label = 0  # correct answer is always the first item
                accuracy = int(pred == label)
                entropy = get_entropy(logits)

                # log evaluation accuracy and entropy
                log(f"{stage}_accuracy", accuracy)
                log(f"{stage}_entropy", entropy)

                # log category-level evaluation accuracies as a separate metric
                category_label = raw_y[0][0]
                log(f"{stage}_accuracy_{category_label}", accuracy)

                ret.update({'accuracy': accuracy})

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
