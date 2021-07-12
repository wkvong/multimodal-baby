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
        # self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.fc2 = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        # x = F.relu(self.fc1(x))
        # x = torch.max(x, dim=1, keepdim=True)[0]
        # x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM,
                            help="size of input embedding")
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM,
                            help="size of embedding representations")
        
        
