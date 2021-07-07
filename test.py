import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_saycam.data.base_data_module import BaseDataModule, load_and_print_info
from multimodal_saycam.data.multimodal_data_module import MultiModalSAYCamDataModule
from multimodal_saycam.models.multimodal_cnn_embedding import MultiModalCNNEmbedding

mm = MultiModalSAYCamDataModule()
# load_and_print_info(MultiModalSAYCamDataModule)

mm.prepare_data()
mm.setup()
batch = iter(mm.train_dataloader()).next()
x, y, z = batch

parser = argparse.ArgumentParser()
args = parser.parse_args()
model = MultiModalCNNEmbedding(args)

logits_per_image, logits_per_text = model(x, y, z)

batch_size = x.size(0)
ground_truth = torch.LongTensor(np.arange(batch_size))

# calculate infonce loss
loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)).div(2)
print(loss)
