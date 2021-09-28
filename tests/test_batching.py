import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import seed_everything

from multimodal import multimodal_data_module
from multimodal import multimodal
from multimodal import multimodal_lit

def test_check():
    a = 2
    b = 2
    assert a == b

def test_cnn():
    # set default cnn args
    args = argparse.Namespace(
        embedding_dim=128,
        pretrained_cnn=True,
        finetune_cnn=False,
    )
    
    # initialize vision encoder
    model = multimodal.VisionEncoder(args)
    model.eval()

    # generate random tensor of correct shape
    x = torch.rand([4, 3, 224, 224])

    with torch.no_grad():
        # compute batched and unbatched outputs
        y_batched = model(x)
        y_unbatched = model._forward_unbatched(x)

        # compare
        assert torch.allclose(y_batched, y_unbatched, atol=1e-5)
    

def test_embedding():
    # set default embedding args
    args = argparse.Namespace(
        text_encoder="embedding",
        input_dim=10000,
        embedding_dim=128,
    )

    # initialize embedding text encoder
    model = multimodal.TextEncoder(args)
    model.eval()

    # random tensor of input shape
    x = torch.LongTensor([[1, 5, 2, 6], [20, 12, 4, 2]])
    x_len = torch.LongTensor([[4], [4]])

    with torch.no_grad():
        # compute batched and unbatched outputs
        y_batched = model(x, x_len)
        y_unbatched = model._forward_unbatched(x, x_len)

        # compare
        assert torch.allclose(y_batched, y_unbatched, atol=1e-5)

    # random padded tensor
    x = torch.LongTensor([[1, 5, 0, 0], [20, 12, 4, 2]])
    x_len = torch.LongTensor([[2], [4]])

    with torch.no_grad():
        # compute batched and unbatched outputs
        y_batched = model(x, x_len)
        y_unbatched = model._forward_unbatched(x, x_len)
     
        # compare
        assert torch.allclose(y_batched, y_unbatched, atol=1e-5)

        
def test_bilstm():
    pass

def test_mean_sim():
    pass

def test_max_sim():
    pass
