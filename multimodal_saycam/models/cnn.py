import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

EMBEDDING_DIM = 128
PRETRAINED_CNN = True
FINETUNE_CNN = False

class CNN(nn.Module):
    """
    ResNeXt CNN
    """

    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        self.pretrained_cnn = self.args.get("pretrained_cnn", PRETRAINED_CNN)
        self.finetune_cnn = self.args.get("finetune_cnn", FINETUNE_CNN)
        self.model = self._load_pretrained_cnn()

    def forward(self, x):
        return self.model(x)

    def _load_pretrained_cnn(self):
        model = torchvision.models.resnext50_32x4d(pretrained=False)
            
        model.fc = torch.nn.Linear(in_features=2048, out_features=2765, bias=True)

        # load in checkpoint
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
            print('Fine-tuning CNN layers!')

        # remove classifier head and add 1x1 convolution to map embedding to lower dim
        model = torch.nn.Sequential(*list(model.children())[:-2],
                                    nn.Conv2d(2048, self.embedding_dim, 1))
        return model
        

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        parser.add_argument("--pretrained_cnn", action="store_true",
                            help="use pretrained CNN")
        parser.add_argument("--finetune_cnn", action="store_true",
                            help="finetune CNN (frozen by default)")


def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
