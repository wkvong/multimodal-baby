import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EMBEDDING_DIM = 128
PRETRAINED_CNN = True
FINETUNE_CNN = False
CNN_OUTPUT = "spatial"
INPUT_DIM = 10000
TEXT_ENCODER = "embedding"
HIDDEN_DIM = 128
SIM = "max"

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VisionEncoder(nn.Module):
    """
    Visual encoder (pre-trained self-supervised ResNeXt CNN from Orhan et al. (2020))
    """

    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.embedding_dim = self.args.get("embedding_dim")
        self.pretrained_cnn = self.args.get("pretrained_cnn")
        self.finetune_cnn = self.args.get("finetune_cnn")
        self.model = self._load_pretrained_cnn()

    def forward(self, x):
        return self.model(x)

    def _load_pretrained_cnn(self):
        # initialize resnext model and replace fc layer to match pretrained model
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=2765, bias=True)

        # load checkpoint
        if self.pretrained_cnn:
            # rename checkpoint keys since we are not using DataParallel
            print('Loading pretrained CNN!')

            checkpoint = torch.load('models/TC-S-resnext.tar',
                                    map_location=torch.device("cpu"))

            prefix = 'module.'
            n_clip = len(prefix)
            renamed_checkpoint = {k[n_clip:]: v for k, v in
                                  checkpoint['model_state_dict'].items()}
            
            model.load_state_dict(renamed_checkpoint)

        if not self.finetune_cnn:
            print('Freezing CNN layers!')
            set_parameter_requires_grad(model)  # freeze cnn layers
        else:
            print('Fine-tuning CNN layers!')  # fine-tune cnn

        # remove classifier head and add 1x1 convolution to map original embedding (2048) to embedding_dim
        model = torch.nn.Sequential(*list(model.children())[:-2],
                                    nn.Conv2d(2048, self.embedding_dim, 1))
        return model


class TextEncoder(nn.Module):
    """
    Text encoder
    """
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder")
        self.input_dim = self.args.get("input_dim")
        self.embedding_dim = self.args.get("embedding_dim")
        self.hidden_dim = self.args.get("hidden_dim")
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)

    def forward(self, x, x_len):
        if self.text_encoder == "embedding":
            output = self.embedding(x)  # (B, L, E)
            return output
        elif self.text_encoder == "lstm":
            # initialize hidden state
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)

            # embed padded sequence and pack
            embedding = self.embedding(x)  # (B, L, E)
            embedding = embedding.view(-1, batch_size, self.embedding_dim)  # (L, B, E)
            # need to move x_len to cpu for this line to work
            embedding = pack_padded_sequence(embedding, x_len.cpu(), enforce_sorted=False)

            # pass through lstm
            output, _ = self.lstm(embedding, hidden)  # output: (L, B, E)

            # unpack and reshape
            output, _  = pad_packed_sequence(output)  # (L, B, E)
            output = output.view(batch_size, -1, self.embedding_dim)  # (B, L, E)
            return output

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim, device="cuda"),
                torch.zeros(1, batch_size, self.hidden_dim, device="cuda"))


class MultiModalModel(nn.Module):
    """
    Model description
    """
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder", TEXT_ENCODER)
        self.sim = self.args.get("sim", SIM)

        self.image_embed = VisionEncoder(args=args)
        self.text_embed = TextEncoder(args=args)

        # learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.image_embed(image)

    def encode_text(self, text, text_length):
        return self.text_embed(text, text_length)
        
    def forward(self, image, text, text_length):
        # encode image and text
        image_features = self.encode_image(image)  # (B, E, H, W,)
        # image_features = F.normalize(image_features, p=2, dim=1)  # normalize image features
        text_features = self.encode_text(text, text_length)  # (B, L, E)
        # text_features = F.normalize(text_features, p=2, dim=2)  # normalize text features

        # calculate batched similarity
        if self.sim == "mean":
            # mean similarity takes the dot product (sum) of all spatial image and word
            # embeddings, and then takes the average across all words and spatial locations
            match_sum = torch.einsum('iehw,tle->it', [image_features, text_features])  # calculate matchmap
            match_avg = match_sum / (7 * 7 * text_length)  # divide by h, w, l
        elif self.sim == "max":
            # max similarity takes the maximum dot product for all spatial embeddings
            # for a given word, and then averages across words
            match_max = torch.einsum('iehw,tle->itlhw', [image_features, text_features])  # calculate matchmap
            match_max = torch.amax(match_max, dim=(3, 4))  # amax to apply over multiple dims
            match_avg = torch.sum(match_max, dim=2) / text_length  # divide by h, w, l

        # transform to logits and scale with temperature param
        logit_scale = self.logit_scale.exp()
        logits_per_image = match_avg * logit_scale
        logits_per_text = match_avg.t() * logit_scale

        return logits_per_image, logits_per_text

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--text_encoder", type=str, default=TEXT_ENCODER, choices=["embedding", "lstm"],
                            help="type of text encoder to use (embedding only or LSTM)")
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM,
                            help="size of input embedding")
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM,
                            help="size of lstm hidden layer")        
        parser.add_argument("--pretrained_cnn", action="store_true",
                            help="use pretrained CNN")
        parser.add_argument("--finetune_cnn", action="store_true",
                            help="finetune CNN (frozen by default)")
        parser.add_argument("--cnn_output", type=str, default=CNN_OUTPUT, choices=["spatial", "flat"],
                            help="type of output from CNN (spatial or flat embedding)")
        parser.add_argument("--sim", type=str, default=SIM, choices=["mean", "max"],
                            help="type of similarity to use (mean or max over image patches per word)")

    
