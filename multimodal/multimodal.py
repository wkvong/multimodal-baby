import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TEXT_ENCODER = "embedding"
EMBEDDING_TYPE = "spatial"
INPUT_DIM = 10000  # TODO: fix to size of vocab
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
PRETRAINED_CNN = True
FINETUNE_CNN = False
NORMALIZE_FEATURES = False
SIM = "max"

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

        self.embedding_type = self.args.get("embedding_type")
        self.embedding_dim = self.args.get("embedding_dim")
        self.pretrained_cnn = self.args.get("pretrained_cnn")
        self.finetune_cnn = self.args.get("finetune_cnn")
        self.model = self._load_pretrained_cnn()

    def forward(self, x):
        return self.model(x)

    def _forward_unbatched(self, x):
        outputs = []
        for i in x:
            i = i.unsqueeze(0)  # add fake batch dim
            output = self.model(i)  # pass through model
            output = output.squeeze()  # remove batch dim
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs

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

        if self.embedding_type == "spatial":
            # remove classifier head and add 1x1 convolution to map original embedding (2048) to embedding_dim
            model = torch.nn.Sequential(*list(model.children())[:-2],
                                        nn.Conv2d(2048, self.embedding_dim, 1))
        elif self.embedding_type == "flat":
            # remove classifier head and add linear layer to map original embedding (2048) to embedding_dim
            model.fc = nn.Linear(2048, self.embedding_dim)

        return model


class TextEncoder(nn.Module):
    """
    Text encoder
    """
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder")
        self.embedding_type = self.args.get("embedding_type")
        self.input_dim = self.args.get("input_dim")
        self.embedding_dim = self.args.get("embedding_dim")
        self.hidden_dim = self.args.get("hidden_dim")
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim,
                                      padding_idx=0)

        if self.text_encoder == "lstm":
            self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True)
        
    def forward(self, x, x_len):
        if self.text_encoder == "embedding":
            if self.embedding_type == "flat":
                # flat embedding for embedding only model
                output = self.embedding(x)  # (B, L, E)

                # calculate mean embedding per utterance
                output = torch.sum(output, dim=1)  # first sum over length dim, (B, E)
                output = torch.div(output, x_len.unsqueeze(1))  # then divide by utterance length, (B, E)
                
                return output
            elif self.embedding_type == "spatial":
                # spatial embedding for embedding only model
                output = self.embedding(x)  # (B, L, E)
                return output
        elif self.text_encoder == "lstm":
            if self.embedding_type == "flat":
                # flat embedding for biLSTM using final hidden states
                # initialize hidden state
                batch_size = x.size(0)
                hidden = self.init_hidden(batch_size)
     
                # embed padded sequence and pack
                embedding = self.embedding(x)  # (B, L, E)
                embedding = embedding.view(-1, batch_size, self.embedding_dim)  # (L, B, E)
                # need to move x_len to cpu for this line to work
                embedding = pack_padded_sequence(embedding, x_len.cpu(), enforce_sorted=False)
     
                # pass through lstm
                output, (hidden, cell) = self.lstm(embedding, hidden)

                # get final hidden state by averaging forward and backward passes
                hidden_fwd = hidden[0, :, :]  # (B, E)
                hidden_bwd = hidden[1, :, :]  # (B, E)
                hidden = torch.mean(torch.stack([hidden_fwd, hidden_bwd]), dim=0)  # (B, E)
                return hidden
            elif self.embedding_type == "spatial":
                # spatial embedding for biLSTM using all hidden states
                # initialize hidden state
                batch_size = x.size(0)
                hidden = self.init_hidden(batch_size)
     
                # embed padded sequence and pack
                embedding = self.embedding(x)  # (B, L, E)
                embedding = embedding.view(-1, batch_size, self.embedding_dim)  # (L, B, E)
                # need to move x_len to cpu for this line to work
                embedding = pack_padded_sequence(embedding, x_len.cpu(), enforce_sorted=False)
     
                # pass through lstm
                output, _ = self.lstm(embedding, hidden)
     
                # unpack and reshape
                output, _  = pad_packed_sequence(output)  # (L, B, 2*E)
                output_fwd = output[:, :, :self.embedding_dim]  # get hidden states from forward pass
                output_bwd = output[:, :, self.embedding_dim:]  # get hidden states from backward pass
                output_avg = torch.mean(torch.stack([output_fwd, output_bwd]), dim=0)  # take average
                output = output.view(batch_size, -1, self.embedding_dim)  # (B, L, E)
                return output

    def _forward_unbatched(self, x, x_len):
        if self.text_encoder == "embedding":
            outputs = []
            for i in x:
                output = self.embedding(i)
                outputs.append(output)
            outputs = torch.stack(outputs)
            return outputs
        
    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim, device="cuda"),
                torch.zeros(2, batch_size, self.hidden_dim, device="cuda"))


class MultiModalModel(nn.Module):
    """
    Model description
    """
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder", TEXT_ENCODER)
        self.sim = self.args.get("sim", SIM)
        self.embedding_type = self.args.get("embedding_type", EMBEDDING_TYPE)
        self.normalize_features = self.args.get("normalize_features", NORMALIZE_FEATURES)

        self.image_embed = VisionEncoder(args=args)
        self.text_embed = TextEncoder(args=args)

        # contrastive temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self-distillation temperature parameter
        self.kl_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def encode_image(self, image):
        return self.image_embed(image)

    def encode_text(self, text, text_length):
        return self.text_embed(text, text_length)
        
    def forward(self, image, text, text_length, self_distillation=False, teacher=False):
        if self.embedding_type == "flat":
            # encode image and text as flat embeddings
            image_features = self.encode_image(image)  # (B, E)
            text_features = self.encode_text(text, text_length)  # (B, E)

            if self.normalize_features:
                image_features = F.normalize(image_features, p=2, dim=1)  # normalize image features
                text_features = F.normalize(text_features, p=2, dim=1)  # normalize text features

            # calculate match similarity
            match = image_features @ text_features.T
            print(match.size())
        elif self.embedding_type == "spatial":
            # encode image and text as spatial embeddings
            image_features = self.encode_image(image)  # (B, E, H, W,)
            text_features = self.encode_text(text, text_length)  # (B, L, E)
     
            if self.normalize_features:
                image_features = F.normalize(image_features, p=2, dim=1)  # normalize image features
                text_features = F.normalize(text_features, p=2, dim=2)  # normalize text features
     
            # calculate batched similarity
            if self.sim == "mean":
                # mean similarity takes the dot product (sum) of all spatial image and word
                # embeddings, and then takes the average across all words and spatial locations
                match_sum = torch.einsum('iehw,tle->it', [image_features, text_features])  # calculate matchmap
                match = match_sum / (7 * 7 * text_length)  # divide by h, w, l
            elif self.sim == "max":
                # max similarity takes the maximum dot product for all spatial embeddings
                # for a given word, and then averages across words
                match_max = torch.einsum('iehw,tle->itlhw', [image_features, text_features])  # calculate matchmap
                match_max = torch.amax(match_max, dim=(3, 4))  # amax to apply over multiple dims
                match = torch.sum(match_max, dim=2) / text_length  # divide by h, w, l

        # transform to logits and scale with temperature param (either infonce or kl temp)
        if self_distillation:
            if teacher:
                logit_scale = 1  # don't scale logits for teacher model
            else:
                logit_scale = self.kl_logit_scale.exp()
        else:
            logit_scale = self.logit_scale.exp()
            
        logits_per_image = match * logit_scale
        logits_per_text = match.t() * logit_scale

        return logits_per_image, logits_per_text

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--text_encoder", type=str, default=TEXT_ENCODER, choices=["embedding", "lstm"],
                            help="type of text encoder to use (embedding only or LSTM)")
        parser.add_argument("--embedding_type", type=str, default=EMBEDDING_TYPE, choices=["spatial", "flat"],
                            help="type of embeddings to use (spatial or flat embedding)")
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
        parser.add_argument("--normalize_features", action="store_true",
                            help="normalize feature embeddings after encoding")
        parser.add_argument("--sim", type=str, default=SIM, choices=["mean", "max"],
                            help="type of similarity to use (mean or max over image patches per word)")
