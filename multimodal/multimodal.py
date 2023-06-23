import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
    PackedSequence
from multimodal.multimodal_data_module import \
    PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, MAX_LEN_UTTERANCE
from multimodal.beam_search import beam_search
from multimodal.utils import get_entropy, map_structure, apply_permutation
from multimodal.attention_maps import Hook
from multimodal.utils import load_model

TEXT_ENCODER = "embedding"
ATTENTION_ACTIVATION = "relu"
EMBEDDING_TYPE = "flat"
EMBEDDING_DIM = 128
CRANGE = 1
DROPOUT_I = 0.0
DROPOUT_O = 0.0
PRETRAINED_CNN = True
FINETUNE_CNN = False
NORMALIZE_FEATURES = False
SIM = "max"
TEMPERATURE = 0.07
FIX_TEMPERATURE = False

# vision encoder arguments
CNN_MODEL = "models/TC-S-resnext.tar"  # link to TC resnext model
CNN_DINO = False  # boolean flag to use DINO resnext model
VIT_DINO = False  # boolean flag to use DINO vision transformer model

# text encoder arguments
POS_EMBED_TYPE = "no_pos_embed"

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
        self.cnn_model = self.args.get("cnn_model", CNN_MODEL)
        self.cnn_dino = self.args.get("cnn_dino", CNN_DINO)
        self.vit_dino = self.args.get("vit_dino", VIT_DINO)
        self.finetune_cnn = self.args.get("finetune_cnn")
        self.model = self._load_pretrained_cnn()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained_cnn", action="store_true",
                            help="use pretrained CNN")
        parser.add_argument("--cnn_model", type=str, default=CNN_MODEL,
                            help="name in torchvision.models or "
                                 "the path to the CNN model checkpoint")
        parser.add_argument("--cnn_dino", action="store_true", default=CNN_DINO,
                            help="use DINO resnext model")
        parser.add_argument("--vit_dino", action="store_true", default=VIT_DINO,
                            help="use DINO vision transformer model")
        parser.add_argument("--finetune_cnn", action="store_true",
                            help="finetune CNN (frozen by default)")

    def forward(self, x):
        if self.vit_dino:
            x = self.model(x)
            features = self.model.head(x)
            feature_map = None
        else:
            if self.embedding_type == "spatial":
                layer = self.model[-2]
            else:
                layer = self.model.layer4
            with Hook(layer, requires_grad=False) as hook:
                features = self.model(x)
                feature_map = hook.activation
                
        return features, feature_map

    def _forward_unbatched(self, x):
        outputs = []
        for i in x:
            i = i.unsqueeze(0)  # add fake batch dim
            output = self.model(i)  # pass through model
            output = output.squeeze()  # remove batch dim
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs

    @property
    def last_cnn_out_dim(self):
        """The dimension of the output of the last CNN block. This is
        in_features of the last projection layer and encoder_dim in attention.
        This may vary with different models, so it may be changed if other
        models are used.
        """
        if self.vit_dino:
            return 768 
        else:
            return 2048

    def _load_pretrained_cnn(self):
        if self.cnn_dino:
            print("Loading DINO resnext model!")
            model_name = "dino_sfp_resnext50"
            model = load_model(model_name, self.pretrained_cnn)
        elif self.vit_dino:
            print("Loading DINO vision transformer model!")
            model_name = "dino_sfp_vitb14"
            model = load_model(model_name, self.pretrained_cnn)
        else:
            # get the model name and checkpoint path
            model_name = self.cnn_model
            checkpoint_path = None
            if not hasattr(torchvision.models, model_name):
                checkpoint_path = self.cnn_model
                name_to_model_name = {
                    'resnext': 'resnext50_32x4d',
                }
                for name, model_name_ in name_to_model_name.items():
                    if name in model_name:
                        model_name = model_name_
                        break
                else:
                    assert False, \
                        f"Unable to recognize the model name of {model_name}"
     
            # initialize model and replace fc layer to match pretrained model
            model = getattr(torchvision.models, model_name)(
                pretrained=self.pretrained_cnn and not checkpoint_path)
            model.fc = torch.nn.Linear(
                in_features=self.last_cnn_out_dim, out_features=2765, bias=True)
     
            # load checkpoint
            if self.pretrained_cnn and checkpoint_path:
                # rename checkpoint keys since we are not using DataParallel
                print('Loading pretrained CNN!')
     
                checkpoint = torch.load(checkpoint_path,
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
            # remove classifier head and add 1x1 convolution to map original embedding to embedding_dim
            model = torch.nn.Sequential(
                *list(model.children())[:-2],
                nn.Conv2d(self.last_cnn_out_dim, self.embedding_dim, 1))
        elif self.embedding_type == "flat":
            # remove classifier head and add linear layer to map original embedding to embedding_dim
            print("Adding linear layer to vision encoder!")
            if self.vit_dino:
                model.head = nn.Linear(self.last_cnn_out_dim, self.embedding_dim)
            else:
                model.fc = nn.Linear(self.last_cnn_out_dim, self.embedding_dim)

        return model


class Attention(nn.Module):
    """Attention
    """

    def __init__(self, encoder_dim, decoder_dim, attn_dim,
                 activation=ATTENTION_ACTIVATION):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attn_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attn_dim)
        self.attn_layer = nn.Linear(attn_dim, 1)
        activation_mapping = {
            'relu': 'ReLU',
            'tanh': 'Tanh',
        }
        self.activation_fn = getattr(nn, activation_mapping[activation])()

    def permute(self, t):
        """Permute the tensor to move the feature dim from the second to the
        last.
        """
        perm = tuple(range(t.dim()))
        perm = perm[:1] + perm[2:] + perm[1:2]
        return t.permute(*perm)

    def unpermute(self, t):
        """Unpermute the tensor to move the feature dim from the last to the
        second.
        """
        perm = tuple(range(t.dim()))
        perm = perm[:1] + perm[-1:] + perm[1:-1]
        return t.permute(*perm)

    def project_encoder_features(self, encoder_features):
        """Project the encoder features.
        Inputs:
            encoder_features: Tensor of shape [N, encoder_dim, ...]
        Returns:
            Tensor of shape [N, attn_dim, ...]
        """
        projected_encoder_features = self.unpermute(self.encoder_projection(
            self.permute(encoder_features)))
        return projected_encoder_features

    def forward(self, encoder_features, projected_encoder_features,
                decoder_features):
        """
        Inputs:
            encoder_features: Tensor of shape [N, encoder_dim, ...]
            projected_encoder_features: Tensor of shape [N, attn_dim, ...]
            decoder_features: Tensor of shape [N, decoder_dim]
        Returns: Tuple of
            features: aggregated features, Tensor of shape [N, encoder_dim]
            attns: normalized attention weights, Tensor of shape [N, ...]
        """
        # [N, attn_dim]
        projected_decoder_features = self.decoder_projection(decoder_features)
        # [N, encoder_dim, -1]
        encoder_features_ = encoder_features.reshape(
            *(encoder_features.shape[:2] + (-1,)))
        # [N, attn_dim, -1]
        projected_encoder_features_ = projected_encoder_features.reshape(
            *(projected_encoder_features.shape[:2] + (-1,)))
        # [N, attn_dim, 1]
        projected_decoder_features_ = projected_decoder_features.unsqueeze(-1)
        # [N, -1]
        attn_logits_ = self.attn_layer(self.permute(
            self.activation_fn(
                # [N, attn_dim, -1]
                projected_encoder_features_ + projected_decoder_features_
            )
        )).squeeze(-1)
        # [N, -1]
        attns_ = F.softmax(attn_logits_, -1)
        # [N, ...]
        attns = attns_.reshape(attns_.size(0), *encoder_features.shape[2:])
        # [N, encoder_dim]
        features = torch.matmul(encoder_features_, attns_.unsqueeze(-1))\
            .squeeze(-1)
        return features, attns


class TextEncoder(nn.Module):
    """
    Text encoder
    """

    def __init__(self, vocab, image_feature_map_dim, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = self.args.get("text_encoder")
        self._captioning = self.args.get("captioning", False)
        self._attention = self.args.get("attention", False)
        self._attention_gate = self.args.get("attention_gate", False)
        self.embedding_type = self.args.get("embedding_type")
        self.embedding_dim = self.args.get("embedding_dim")
        # always match embedding and hidden dim for consistency
        self.hidden_dim = self.embedding_dim
        self.input_dim = self.embedding_dim
        if self._attention:
            self.input_dim += image_feature_map_dim
        self.crange = self.args.get("crange")
        self.dropout_i = self.args.get("dropout_i")
        self.dropout_o = self.args.get("dropout_o")
        self.pos_embed_type = self.args.get("pos_embed_type", POS_EMBED_TYPE)
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
        if self.text_encoder in ["lstm", "bilstm"]:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,
                                bidirectional= self.text_encoder == "bilstm")
        elif self.text_encoder == "transformer":
            # add transformer components
            print("Building transformer text encoder!")
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
            
            # set-up positional embeddings
            if self.pos_embed_type == "sinusoidal":
                # create sinusoidal positional embeddings
                # adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
                print("Initializing sinusoidal positional embeddings!")
                pos_embed = torch.zeros(MAX_LEN_UTTERANCE, self.embedding_dim)
                position = torch.arange(0, MAX_LEN_UTTERANCE).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) *
                    -(math.log(10000.0) / self.embedding_dim))
                pos_embed[:, 0::2] = torch.sin(position * div_term)
                pos_embed[:, 1::2] = torch.cos(position * div_term)
                pos_embed = pos_embed.unsqueeze(0).permute(1, 0, 2)
                self.register_buffer('pos_embed', pos_embed)
            elif self.pos_embed_type == "learned":
                # set up learned positional embeddings
                print("Initializing learned positional embeddings!")
                self.pos_embed = nn.Parameter(torch.zeros(MAX_LEN_UTTERANCE, 1, self.embedding_dim))
            else:
                # don't use any positional embeddings
                print("Initializing no positional embeddings!")
                pass

        # build captioning related parts
        if self.captioning:
            assert self.regressional, \
                "only regressional text encoder supports captioning"
            self.connector = nn.Linear(
                self.args.get("embedding_dim"),  # input image feature dim
                2 * self.lstm.num_layers * self.hidden_dim,
            )

        # build attention module
        if self._attention:
            self.attention = Attention(
                image_feature_map_dim, self.hidden_dim, self.hidden_dim)
            if self.has_attention_gate:
                self.attention_gate_projection = nn.Linear(
                    self.hidden_dim, image_feature_map_dim)

        self.lockdrop = LockedDropout()
        self.output_dropout = nn.Dropout(self.dropout_o)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--text_encoder", type=str, default=TEXT_ENCODER,
                            choices=["embedding", "cbow", "lstm", "bilstm", "transformer"],
                            help="text encoder architecture")
        parser.add_argument("--captioning", action="store_true",
                            help="whether to initialize the hidden states with the image features")
        parser.add_argument("--attention", action="store_true",
                            help="whether to attend to the image feature map")
        parser.add_argument("--attention_activation", type=str,
                            default=ATTENTION_ACTIVATION,
                            choices=["relu", "tanh"],
                            help="activation in attention")
        parser.add_argument("--attention_gate", action="store_true",
                            help="whether to use attention gate")
        parser.add_argument("--crange", type=int, default=CRANGE,
                            help="context range for cbow")
        parser.add_argument("--dropout_i", type=float, default=DROPOUT_I,
                            help="input dropout rate; not applicable for embedding text encoder")
        parser.add_argument("--dropout_o", type=float, default=DROPOUT_O,
                            help="output dropout rate")
        parser.add_argument("--pos_embed_type", type=str, default=POS_EMBED_TYPE,
                            choices=["no_pos_embed", "sinusoidal", "learned"],
                            help="type of positional embedding to use")

    def inputs_to_outputs(self, inputs, states,
                          image_feature_map=None,
                          projected_image_feature_map=None):
        """Perform one step from embedded inputs to outputs and states.
        Inputs:
            inputs: Tensor of shape [batch_size, embedding_dim].
            states: hidden states.
            image_feature_map: the image_feature_map.
            projected_image_feature_map: the projected image_feature_map.
        Returns:
            (outputs, states, attns)
        """
        if image_feature_map is not None:
            h = states[0][-1]
            attn_feature, attns = self.attention(
                image_feature_map, projected_image_feature_map, h)
            if self.has_attention_gate:
                gate = F.sigmoid(self.attention_gate_projection(h))
                attn_feature = gate * attn_feature
            # concatenate attention features to inputs
            inputs = torch.cat([inputs, attn_feature], dim=1)
        else:
            attns = None
        inputs = inputs.unsqueeze(0)
        outputs, states = self.lstm(inputs, states)
        outputs = outputs.squeeze(0)
        return outputs, states, attns

    def ids_to_outputs(self, ids, states,
                       image_feature_map=None,
                       projected_image_feature_map=None):
        """Perform one step from input ids to outputs and states.
        Inputs:
            ids: Tensor of shape [batch_size].
            states: hidden states.
            image_feature_map: image_feature_map.
            projected_image_feature_map: the projected image_feature_map.
        Returns:
            (outputs, states, attns)
        """
        inputs = self.embedding(ids)
        return self.inputs_to_outputs(
            inputs, states,
            image_feature_map=image_feature_map,
            projected_image_feature_map=projected_image_feature_map)

    def train_greedy(self, input, hidden, image_feature_map=None):
        """Teacher-forcing training.
        Inputs:
            input: PackedSequence containing packed embedded inputs.
            hidden: initial hidden states.
            image_feature_map: the image_feature_map to attend to.
        Returns:
            (outputs, hidden, attns)
            outputs, hidden are of the same format of LSTM outputs.
            attns: PackedSequence containing attention weights.
        """
        input, batch_sizes, sorted_indices, unsorted_indices = input
        hidden = self.lstm.permute_hidden(hidden, sorted_indices)
        if image_feature_map is not None:
            image_feature_map = apply_permutation(
                image_feature_map, sorted_indices, 0)
            projected_image_feature_map = \
                self.attention.project_encoder_features(image_feature_map)
            attns = []
        outputs = []
        p = 0
        for batch_size in batch_sizes:
            p_ = p + batch_size
            input_batch = input[p:p_]
            hidden_batch = map_structure(lambda t: t[:, :batch_size], hidden)
            if image_feature_map is not None:
                image_feature_map_batch = image_feature_map[:batch_size]
                projected_image_feature_map_batch = \
                    projected_image_feature_map[:batch_size]
            else:
                image_feature_map_batch = None
                projected_image_feature_map_batch = None
            outputs_batch, hidden_batch, attn_batch = self.inputs_to_outputs(
                input_batch, hidden_batch,
                image_feature_map=image_feature_map_batch,
                projected_image_feature_map=projected_image_feature_map_batch)
            hidden = map_structure(
                lambda h, h_batch: torch.cat(
                    (h_batch, h[:, batch_size:]), dim=1),
                hidden, hidden_batch)
            outputs.append(outputs_batch)
            if image_feature_map is not None:
                attns.append(attn_batch)
            p = p_
        outputs = torch.cat(outputs)
        outputs_packed = PackedSequence(
            outputs, batch_sizes, sorted_indices, unsorted_indices)
        if image_feature_map is not None:
            attns = torch.cat(attns)
            attns_packed = PackedSequence(
                attns, batch_sizes, sorted_indices, unsorted_indices)
        else:
            attns_packed = None
        return outputs_packed, \
            self.lstm.permute_hidden(hidden, unsorted_indices), attns_packed

    def forward(self, x, x_len, image_features=None, image_feature_map=None):
        attns = None

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

        elif self.text_encoder in ["lstm", "bilstm"]:
            # initialize hidden state
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size, image_features=image_features)

            # embed padded sequence and pack
            embedding = self.lockdrop(embedding, self.dropout_i)

            # need to move x_len to cpu for this line to work
            embedding = pack_padded_sequence(
                embedding, x_len.cpu(), batch_first=True, enforce_sorted=False)

            # pass through lstm
            if self.has_attention:
                # teacher-forcing
                raw_output, (hidden, cell), attns = self.train_greedy(
                    embedding, hidden, image_feature_map=image_feature_map)
                if attns is not None:
                    attns, _ = pad_packed_sequence(attns, batch_first=True)
            else:
                # use PyTorch LSTM directly
                raw_output, (hidden, cell) = self.lstm(embedding, hidden)

            # unpack and reshape
            # (B, L, 2*E) for bilstm, (B, L, E) for unilstm
            raw_output, _ = pad_packed_sequence(raw_output, batch_first=True)

            # average hidden states from forward and backward passes for bilstm
            if self.text_encoder == "bilstm":
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
        elif self.text_encoder == "transformer":
            # first, calculate src padding mask based on input
            src_key_padding_mask = (x == 0).bool()

            # then, transpose to (L, B, E) for transformer encoder and pass through
            embedding = embedding.permute(1, 0, 2)

            # add positional embedding if specified
            if self.pos_embed_type == "sinusoidal" or self.pos_embed_type == "learned":
                pos_embed = self.pos_embed[:embedding.size(0), :, :]
                embedding = embedding + pos_embed

            raw_output = self.transformer_encoder(embedding, src_key_padding_mask=src_key_padding_mask)

            # transpose back to (B, L, E)
            raw_output = raw_output.permute(1, 0, 2)

            # perform average pooling over inputs
            if self.embedding_type == "flat":  # flat embedding for transformer 
                # calculate mean embedding per utterance
                ret = torch.sum(raw_output, dim=1) / x_len.unsqueeze(1)

        output = self.lockdrop(raw_output, self.dropout_o)

        if self.embedding_type == "flat":
            ret = self.output_dropout(ret)
        elif self.embedding_type == "spatial":
            ret = output

        # print(f'output shape: {output.shape}')

        return ret, output, attns

    def _forward_unbatched(self, x, x_len, image_features=None):
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
        elif self.text_encoder in ["lstm", "bilstm"]:
            outputs = []
            max_seq_len = torch.max(x_len)  # get max length for padding
            for i, i_len in zip(x, x_len):
                batch_size = 1
                i = i.unsqueeze(0)  # add fake batch dimension for lstm
                i_len = i_len.unsqueeze(0)  # do the same for length
                hidden = self.init_hidden(batch_size, image_features=image_features)

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
                    if self.text_encoder == "bilstm":
                        output_fwd = output[:, :,
                                            :self.embedding_dim]  # (L, B, E)
                        output_bwd = output[:, :,
                                            self.embedding_dim:]  # (L, B, E)
                        output = torch.mean(torch.stack(
                            [output_fwd, output_bwd]), dim=0)  # (L, B, E)

                    # transpose, and remove batch dim
                    output = output.transpose(0, 1).squeeze()

                    # pad output to max_seq_len of current batch
                    padded_output = torch.zeros(
                        (max_seq_len, self.embedding_dim)).to(self.device)
                    padded_output[:i_len[0], :] = output

                outputs.append(padded_output)

            outputs = torch.stack(outputs)
            return outputs

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def regressional(self):
        return self.text_encoder == "lstm"

    # the following properties are for backward compatibility
    @property
    def captioning(self):
        return getattr(self, '_captioning', False)

    @property
    def has_attention(self):
        return getattr(self, '_attention', False)

    @property
    def has_attention_gate(self):
        return getattr(self, '_attention_gate', False)

    def init_hidden(self, batch_size, image_features=None):
        d = 2 if self.text_encoder == "bilstm" else 1

        # captioning: init by image_features
        if image_features is not None:
            if image_features.dim() > 2: # (B, E, H, W)
                assert image_features.dim() == 4
                # compress image_features into shape (B, E)
                image_features = image_features.mean(dim=(2, 3))
            else: # (B, E)
                assert image_features.dim() == 2
            return self.connector(image_features)\
                .reshape(image_features.size(0), 2, d * self.lstm.num_layers, self.hidden_dim)\
                .permute(1, 2, 0, 3)\
                .unbind()

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
        self.normalize_features = self.args.get(
            "normalize_features", NORMALIZE_FEATURES)
        self.initial_temperature = self.args.get("temperature", TEMPERATURE)
        self.fix_temperature = self.args.get(
            "fix_temperature", FIX_TEMPERATURE)

        self.image_embed = vision_encoder
        self.text_embed = text_encoder

        # contrastive temperature parameter
        self.logit_neg_log_temperature = torch.ones([]) * - np.log(self.initial_temperature)
        if not self.fix_temperature:
            self.logit_neg_log_temperature = nn.Parameter(
                self.logit_neg_log_temperature)

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
        parser.add_argument("--fix_temperature", action="store_true",
                            help="fix the temperature so it is not trained")

    def encode_image(self, image):
        image_features, image_feature_map = self.image_embed(image)
        if self.normalize_features:
            # normalize image features
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features, image_feature_map

    def encode_text(self, text, text_length):
        text_features, text_outputs, attns = self.text_embed(text, text_length)
        if self.normalize_features:
            # normalize text features
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features, text_outputs

    def forward(self, image, text, text_length, return_image_features=False, return_text_outputs=False):
        # encode image and text
        image_features, image_feature_map = self.encode_image(image)
        text_features, text_outputs = self.encode_text(text, text_length)

        if self.embedding_type == "flat":
            # image_features: (B, E), text_features: (B, E)

            # calculate match similarity
            match = image_features @ text_features.T

        elif self.embedding_type == "spatial":
            # image_features: (B, E, H, W), text_features: (B, L, E)

            # calculate batched similarity
            if self.sim == "mean":
                # mean similarity takes the dot product (sum) of all spatial image and word
                # embeddings, and then takes the average across all words and spatial locations
                # calculate matchmap
                match_sum = torch.einsum(
                    'iehw,tle->it', [image_features, text_features])
                # divide by h, w, l
                match = match_sum / (
                    image_features.size(-2) * image_features.size(-1)
                    * text_length)
            elif self.sim == "max":
                # max similarity takes the maximum dot product for all spatial embeddings
                # for a given word, and then averages across words
                # calculate matchmap
                match_max = torch.einsum(
                    'iehw,tle->itlhw', [image_features, text_features])
                # amax to apply over multiple dims
                match_max = torch.amax(match_max, dim=(3, 4))
                # divide by h, w, l
                match = torch.sum(match_max, dim=2) / text_length

        # transform to logits and scale with temperature param
        logit_log_scale = self.logit_neg_log_temperature
        logit_scale = logit_log_scale.exp()

        logits_per_image = match * logit_scale
        logits_per_text = match.t() * logit_scale

        ret = logits_per_image, logits_per_text
        if return_image_features:
            ret = ret + (image_features, image_feature_map)
        if return_text_outputs:
            ret = ret + (text_outputs,)
        return ret

    def calculate_contrastive_loss(self, x, y, y_len):
        logits_per_image, logits_per_text, image_features, image_feature_map,\
        text_outputs = self(
            x, y, y_len, return_image_features=True, return_text_outputs=True)

        # create ground truth labels
        batch_size = x.size(0)
        ground_truth = torch.tensor(
            np.arange(batch_size), dtype=torch.long,
            device=logits_per_image.device)

        # calculate infonce loss
        infonce_loss = (
            F.cross_entropy(logits_per_image, ground_truth) +
            F.cross_entropy(logits_per_text, ground_truth)).div(2)

        # calculate accuracy (image and text separately)
        image_pred = torch.argmax(logits_per_image, dim=-1)
        text_pred = torch.argmax(logits_per_text, dim=-1)
        image_accuracy = (image_pred == ground_truth).sum() / batch_size
        text_accuracy = (text_pred == ground_truth).sum() / batch_size
        image_entropy = get_entropy(logits_per_image, dim=-1).mean()
        text_entropy = get_entropy(logits_per_text, dim=-1).mean()

        return infonce_loss, image_accuracy, text_accuracy, \
            image_entropy, text_entropy, logits_per_image, logits_per_text, \
            image_features, image_feature_map, text_outputs


class LanguageModel(nn.Module):
    def __init__(self, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.text_encoder = text_encoder

        # build output layer
        self.output_layer = nn.Linear(
            self.text_encoder.hidden_dim, self.text_encoder.vocab_size, bias=self.args.get("bias", True))
        if self.args.get("tie", True):
            self.output_layer.weight = self.text_encoder.embedding.weight

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tie", type=lambda s: bool(eval(s)), default=True,
                            help="whether to tie the input embedding and output layer weight")
        parser.add_argument("--bias", type=lambda s: bool(eval(s)), default=True,
                            help="whether to use bias for output layer")

    def forward(
        self, y, y_len,
        outputs=None,
        image_features=None,
        image_feature_map=None,
    ):
        if outputs is None:
            text_features, outputs, attns = self.text_encoder(
                y, y_len,
                image_features=image_features,
                image_feature_map=image_feature_map,
            )
        else:
            # in this case the outputs is reused, so it mustn't be an attention
            # model.
            attns = None
        logits = self.output_layer(outputs)
        return outputs, logits, attns

    def calculate_ce_loss(
        self, y, y_len,
        outputs=None,
        image_features=None,
        image_feature_map=None,
        tokenwise=False,
        weight=None,
    ):
        outputs, logits, attns = self(
            y, y_len,
            outputs=outputs,
            image_features=image_features,
            image_feature_map=image_feature_map,
        )

        if not self.text_encoder.regressional:
            labels = y
        else:
            logits = logits[:, :-1]
            labels = y[:, 1:1+logits.size(1)]
        loss = F.cross_entropy(
            logits.transpose(-2, -1),
            labels,
            weight=weight,
            ignore_index=PAD_TOKEN_ID,
            reduction="none" if tokenwise else "mean")

        return loss, outputs, logits, attns, labels

    def beam_search_decode(
        self, batch_size, beam_width, decode_length, length_penalty_alpha,
        image_features=None, image_feature_map=None):
        """Beam search decoding.

        Args:
            batch_size: the batch size.
            beam_width: the beam width.
            decode_length: the maximum decode length.
            length_penalty_alpha: the length penalty alpha.
            image_features: if it is image captioning, use these image_features
                to get the initial states.
            image_feature_map: if it is image captioning with attention, attend
                to image_feature_map.
        Returns:
            Tuple of

            beam_seq: a [batch_size, beam_width, sequence_length] tensor
                containing the sequence ids.
            log_prob: a [batch_size, beam_width] tensor containing the log
                probabilities of the corresponding sequences.
        """

        assert self.text_encoder.regressional, \
            "only regressional text encoder supports beam search decoding"

        start_tokens = torch.full(
            (batch_size,), SOS_TOKEN_ID,
            dtype=torch.int, device=self.text_encoder.device)
        init_states = self.text_encoder.init_hidden(
            batch_size, image_features=image_features)
        init_states = map_structure(lambda t: t.transpose(0, 1), init_states)
        if self.text_encoder.has_attention:
            projected_image_feature_map = \
                self.text_encoder.attention.project_encoder_features(
                    image_feature_map)
            # pack image_feature_map into states
            init_states = init_states, \
                image_feature_map, projected_image_feature_map

        def _symbols_to_logits_fn(ids, states):
            if self.text_encoder.has_attention:
                # unpack states
                states, image_feature_map, projected_image_feature_map = states
            else:
                image_feature_map, projected_image_feature_map = None, None
            states = map_structure(lambda t: t.transpose(0, 1), states)
            outputs, states, attns = self.text_encoder.ids_to_outputs(
                ids[:, -1], states,
                image_feature_map=image_feature_map,
                projected_image_feature_map=projected_image_feature_map)
            states = map_structure(lambda t: t.transpose(0, 1), states)
            logits = self.output_layer(outputs)
            if self.text_encoder.has_attention:
                # pack states
                states = states, image_feature_map, projected_image_feature_map
            return logits, states

        return beam_search(
            _symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self.text_encoder.vocab_size,
            length_penalty_alpha,
            states=init_states,
            eos_id=EOS_TOKEN_ID,
        )


def calculate_attn_reg_loss(attns):
    return ((attns.sum(dim=1) - 1.) ** 2).mean()
