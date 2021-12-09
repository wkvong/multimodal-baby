import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from locked_dropout import LockedDropout
from multimodal.multimodal_data_module import PAD_TOKEN_ID

TEXT_ENCODER = "embedding"
EMBEDDING_TYPE = "spatial"
EMBEDDING_DIM = 128
CRANGE = 1
DROPOUT_I = 0.0
DROPOUT_O = 0.0
PRETRAINED_CNN = True
FINETUNE_CNN = False
NORMALIZE_FEATURES = False
SIM = "max"
TEMPERATURE = 0.07
KL_TEMPERATURE = 0.07
FIX_TEMPERATURE = False


def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout, dim=1):
        if not (self.training and dropout):
            return x
        return x.new_empty(x.shape[:dim] + (1,) + x.shape[dim+1:]).bernoulli_(1 - dropout) / (1 - dropout) * x


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

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained_cnn", action="store_true",
                            help="use pretrained CNN")
        parser.add_argument("--finetune_cnn", action="store_true",
                            help="finetune CNN (frozen by default)")

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
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=2765, bias=True)

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

    def __init__(self, vocab, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder")
        self.bidirectional = self.args.get("bidirectional")
        self.embedding_type = self.args.get("embedding_type")
        self.embedding_dim = self.args.get("embedding_dim")
        # always match embedding and hidden dim for consistency
        self.hidden_dim = self.embedding_dim
        self.crange = self.args.get("crange")
        self.dropout_i = self.args.get("dropout_i")
        self.dropout_o = self.args.get("dropout_o")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # load vocab and create dict to map indices back to words
        self.vocab = vocab
        self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}

        # build embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim,
                                      padding_idx=0)

        # build model
        if self.text_encoder == "lstm":
            self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                                bidirectional=self.bidirectional)

        self.lockdrop = LockedDropout()
        self.output_dropout = nn.Dropout(self.dropout_o)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--text_encoder", type=str, default=TEXT_ENCODER, choices=["embedding", "cbow", "lstm"],
                            help="type of text encoder to use (embedding only or LSTM)")
        parser.add_argument("--bidirectional", action="store_true",
                            help="set LSTM text encoder to bidirectional")
        parser.add_argument("--crange", type=int, default=CRANGE,
                            help="context range for cbow")
        parser.add_argument("--dropout_i", type=float, default=DROPOUT_I,
                            help="input dropout rate; not applicable for embedding text encoder")
        parser.add_argument("--dropout_o", type=float, default=DROPOUT_O,
                            help="output dropout rate")

    def forward(self, x, x_len):
        embedding = self.embedding(x)  # (B, L, E)

        if self.text_encoder == "embedding":
            raw_output = embedding  # (B, L, E)

            if self.embedding_type == "flat":  # flat embedding for embedding only model
                # calculate mean embedding per utterance
                ret = torch.sum(raw_output, dim=1) / x_len.unsqueeze(1)

        elif self.text_encoder == "cbow":
            assert self.embedding_type != "flat", "cbow with flat embedding is nonsense"
            presum = F.pad(
                embedding, (0, 0, self.crange + 1, self.crange)).cumsum(1)
            raw_output = (presum[:, 2 * self.crange + 1:] - presum[:,
                                                                   : - (2 * self.crange + 1)] - embedding) / (2 * self.crange)
            # raw_output = torch.stack([torch.cat([embedding[:, max(j - self.crange, 0) : j], embedding[:, j + 1 : j + self.crange + 1]], dim=1).sum(1) for j in range(embedding.size(1))], dim=1) / (2 * self.crange) # alternative way (brute force by definition)

        elif self.text_encoder == "lstm":
            # initialize hidden state
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)

            # embed padded sequence and pack
            embedding = self.lockdrop(embedding, self.dropout_i)

            # need to move x_len to cpu for this line to work
            embedding = pack_padded_sequence(
                embedding, x_len.cpu(), batch_first=True, enforce_sorted=False)

            # pass through lstm
            raw_output, (hidden, cell) = self.lstm(embedding, hidden)

            # unpack and reshape
            # (B, L, 2*E) for bilstm, (B, L, E) for unilstm
            raw_output, _ = pad_packed_sequence(raw_output, batch_first=True)

            # average hidden states from forward and backward passes for bilstm
            if self.bidirectional:
                raw_output_fwd = raw_output[:, :,
                                            :self.embedding_dim]  # (B, L, E)
                raw_output_bwd = raw_output[:, :,
                                            self.embedding_dim:]  # (B, L, E)
                raw_output = torch.mean(torch.stack(
                    [raw_output_fwd, raw_output_bwd]), dim=0)  # (B, L, E)

            if self.embedding_type == "flat":
                # flat embedding for biLSTM using final hidden states
                # get final hidden state by averaging forward and backward passes
                ret = hidden.mean(dim=0)  # (B, E)

        output = self.lockdrop(raw_output, self.dropout_o)

        if self.embedding_type == "flat":
            ret = self.output_dropout(ret)
        elif self.embedding_type == "spatial":
            ret = output

        return ret, output

    def _forward_unbatched(self, x, x_len):
        if self.text_encoder == "embedding":
            outputs = []
            for i, i_len in zip(x, x_len):
                if self.embedding_type == "flat":
                    output = self.embedding(i)  # embed each word individually
                    output = torch.sum(output, dim=0)  # sum over length dim
                    # divide by utterance len
                    output = torch.div(output, i_len)
                elif self.embedding_type == "spatial":
                    output = self.embedding(i)  # embed each word individually

                outputs.append(output)
            outputs = torch.stack(outputs)
            return outputs
        elif self.text_encoder == "lstm":
            outputs = []
            max_seq_len = torch.max(x_len)  # get max length for padding
            for i, i_len in zip(x, x_len):
                batch_size = 1
                i = i.unsqueeze(0)  # add fake batch dimension for lstm
                i_len = i_len.unsqueeze(0)  # do the same for length
                hidden = self.init_hidden(batch_size)

                # embed, and single sequence (need to do this otherwise model uses padding)
                # alternative would be to remove padding before passing into lstm
                embedding = self.embedding(i)  # (1, L, E)
                embedding = embedding.transpose(0, 1)  # (L, 1, E)
                embedding = pack_padded_sequence(
                    embedding, i_len.cpu(), enforce_sorted=False)

                # pass through lstm
                output, (hidden, cell) = self.lstm(embedding, hidden)

                if self.embedding_type == "flat":
                    # flat embedding for biLSTM using final hidden states
                    # get final hidden state by averaging forward and backward passes
                    padded_output = hidden.mean(dim=0).squeeze()  # (B, E)
                elif self.embedding_type == "spatial":
                    # (L, B, 2*E) for bilstm, (L, B, E) for unilstm
                    output, _ = pad_packed_sequence(output)

                    # average hidden states from forward and backward passes for bilstm
                    if self.bidirectional:
                        output_fwd = output[:, :,
                                            :self.embedding_dim]  # (L, B, E)
                        output_bwd = output[:, :,
                                            self.embedding_dim:]  # (L, B, E)
                        output = torch.mean(torch.stack(
                            [output_fwd, output_bwd]), dim=0)  # (L, B, E)

                    # transpose, and remove batch dim
                    output = output.transpose(0, 1).squeeze()

                    # pad output to max_seq_len of current batch
                    padded_output = torch.zeros((max_seq_len, self.embedding_dim)).to(self.device)
                    padded_output[:i_len[0], :] = output
                    
                outputs.append(padded_output)

            outputs = torch.stack(outputs)
            return outputs

    @property
    def vocab_size(self):
        return len(self.vocab)

    def init_hidden(self, batch_size):
        d = 2 if self.bidirectional else 1
        return (torch.zeros(d * self.lstm.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(d * self.lstm.num_layers, batch_size, self.hidden_dim).to(self.device))


class MultiModalModel(nn.Module):
    """
    Model description
    """
    def __init__(self, vision_encoder, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.sim = self.args.get("sim", SIM)
        self.embedding_type = self.args.get("embedding_type", EMBEDDING_TYPE)
        self.normalize_features = self.args.get("normalize_features", NORMALIZE_FEATURES)
        self.initial_temperature = self.args.get("temperature", TEMPERATURE)
        self.initial_kl_temperature = self.args.get("kl_temperature", KL_TEMPERATURE)
        self.fix_temperature = self.args.get("fix_temperature", FIX_TEMPERATURE)

        self.image_embed = vision_encoder
        self.text_embed = text_encoder

        # contrastive temperature parameter
        self.logit_neg_log_temperature = nn.Parameter(torch.ones([]) * - np.log(self.initial_temperature))

        # self-distillation temperature parameter
        self.kl_logit_neg_log_temperature = nn.Parameter(torch.ones([]) * - np.log(self.initial_kl_temperature))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--embedding_type", type=str, default=EMBEDDING_TYPE, choices=["spatial", "flat"],
                            help="type of embeddings to use (spatial or flat embedding)")
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        parser.add_argument("--normalize_features", action="store_true",
                            help="normalize feature embeddings after encoding")
        parser.add_argument("--sim", type=str, default=SIM, choices=["mean", "max"],
                            help="type of similarity to use (mean or max over image patches per word)")
        parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                            help="initial temperature")
        parser.add_argument("--kl_temperature", type=float, default=KL_TEMPERATURE,
                            help="initial kl temperature")
        parser.add_argument("--fix_temperature", action="store_true",
                            help="fix the temperature so it is not trained (TODO: but still under weight decay)")

    def encode_image(self, image):
        return self.image_embed(image)

    def encode_text(self, text, text_length):
        text_features, outputs = self.text_embed(text, text_length)
        return text_features

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
                match = match_sum / (image_features.size(-2) * image_features.size(-1) * text_length)  # divide by h, w, l
            elif self.sim == "max":
                # max similarity takes the maximum dot product for all spatial embeddings
                # for a given word, and then averages across words
                match_max = torch.einsum('iehw,tle->itlhw', [image_features, text_features])  # calculate matchmap
                match_max = torch.amax(match_max, dim=(3, 4))  # amax to apply over multiple dims
                match = torch.sum(match_max, dim=2) / text_length  # divide by h, w, l

        # transform to logits and scale with temperature param (either infonce or kl temp)
        if self_distillation:
            if teacher:
                logit_log_scale = torch.zeros_like(self.kl_logit_neg_log_temperature)  # don't scale logits for teacher model
            else:
                logit_log_scale = self.kl_logit_neg_log_temperature
        else:
            logit_log_scale = self.logit_neg_log_temperature
        logit_scale = logit_log_scale.exp()

        if self.fix_temperature: # do not train the logit_scale, i.e., do not backprop through the temperature
            logit_scale = logit_scale.detach()

        logits_per_image = match * logit_scale
        logits_per_text = match.t() * logit_scale

        return logits_per_image, logits_per_text


class LanguageModel(nn.Module):
    def __init__(self, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = text_encoder

        # build output layer
        self.output_layer = nn.Linear(self.text_encoder.hidden_dim, self.text_encoder.vocab_size, bias=self.args.get("bias", True))
        if self.args.get("tie", True):
            self.output_layer.weight = self.text_encoder.embedding.weight

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tie", type=lambda s: bool(eval(s)), default=True, help="whether to tie the input embedding and output layer weight")
        parser.add_argument("--bias", type=lambda s: bool(eval(s)), default=True, help="whether to use bias for output layer")

    def forward(self, y, y_len):
        text_features, outputs = self.text_encoder(y, y_len)
        logits = self.output_layer(outputs)
        return outputs, logits

    def calculate_ce_loss(self, y, y_len, tokenwise=False):
        outputs, logits = self(y, y_len)

        if self.text_encoder.text_encoder in ['cbow', 'bert']:
            labels = y
        else:
            logits = logits[:, :-1]
            labels = y[:, 1:1+logits.size(1)]
        loss = F.cross_entropy(logits.transpose(-2, -1), labels, ignore_index=PAD_TOKEN_ID, reduction="none" if tokenwise else "mean")
        perplexity = loss.exp()

        return loss, perplexity, outputs, logits, labels
