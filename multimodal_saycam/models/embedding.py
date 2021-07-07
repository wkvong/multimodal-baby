import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 10000
EMBEDDING_DIM = 128

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.input_dim = self.args.get("input_dim", INPUT_DIM)
        self.embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim,
                                      padding_idx=0)

    def forward(self, x):
        return self.embedding(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM,
                            help="size of input embedding")
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        
        
