import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_saycam.data.base_data_module import BaseDataModule, load_and_print_info
from multimodal_saycam.data.multimodal_data_module import MultiModalSAYCamDataModule
from multimodal_saycam.models.multimodal_cnn_embedding import MultiModalCNNEmbedding
from training.train import _setup_parser

parser = _setup_parser()
args = parser.parse_args()
print(args)

dm = MultiModalSAYCamDataModule(args)
dm.prepare_data()
dm.setup()
train_dataloader = dm.train_dataloader()

print(f'size of dataloader: {len(train_dataloader)}')

batch = next(iter(train_dataloader))
w, x, y, z = batch



