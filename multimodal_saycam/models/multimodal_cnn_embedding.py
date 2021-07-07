import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNN
from .embedding import Embedding

INPUT_DIM = 10000
EMBEDDING_DIM = 128
PRETRAINED_CNN = True
FINETUNE_CNN = False


class MultiModalCNNEmbedding(nn.Module):
    """
    Model description
    """
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.input_dim = self.args.get("input_dim", INPUT_DIM)
        self.embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        self.pretrained_cnn = self.args.get("pretrained_cnn", PRETRAINED_CNN)
        self.finetune_cnn = self.args.get("finetune_cnn", FINETUNE_CNN)

        self.image_embed = CNN(args=args)
        self.text_embed = Embedding(args=args)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.image_embed(image)

    def encode_text(self, text):
        return self.text_embed(text)
        
    def forward(self, image, text, text_length):
        # encode image and text
        image_features = self.encode_image(image)  # (B, E, H, W,)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = self.encode_text(text)  # (B, L, E)
        text_features = F.normalize(text_features, p=2, dim=2)

        # calculate batched similarity 
        match_sum = torch.einsum('iehw,tle->it', [image_features, text_features])  # calculate matchmap
        match_avg = match_sum / (7 * 7 * text_length)  # divide by h, w, l

        # transform to logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = match_avg * logit_scale
        logits_per_text = match_avg.t() * logit_scale

        print(logits_per_image)
        print(f'temp: {self.logit_scale}')
        
        return logits_per_image, logits_per_text

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM,
                            help="size of input embedding")
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        parser.add_argument("--pretrained_cnn", action="store_true",
                            help="use pretrained CNN")
        parser.add_argument("--finetune_cnn", action="store_true",
                            help="finetune CNN (frozen by default)")

    
