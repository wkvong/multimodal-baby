import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import seed_everything

from multimodal import multimodal_data_module
from multimodal import multimodal
from multimodal import multimodal_lit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

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
    

def random_padded_tensor():
    batch_size = 4  
    max_seq_len = 16  # same as dataset

    # set initial tensor to be all zeros, and generate random seq_lens
    x = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    x_len = torch.randint(low=1, high=16, size=(4,)).to(device)

    for i in range(len(x)):
        # randomly generate sequence of a certain length
        x[i, :x_len[i]] = torch.randint(low=1, high=16, size=(x_len[i], ))
    
    return x, x_len
    
def test_spatial_embedding():
    # set default embedding args
    args = argparse.Namespace(
        text_encoder="embedding",
        embedding_type="spatial",
        input_dim=10000,
        embedding_dim=128,
    )

    # initialize embedding text encoder
    model = multimodal.TextEncoder(args).to(device)
    model.eval()

    # random tensor of input shape
    x, x_len = random_padded_tensor()

    with torch.no_grad():
        # compute batched and unbatched outputs
        y_batched = model(x, x_len)
        y_unbatched = model._forward_unbatched(x, x_len)

        # compare
        assert torch.allclose(y_batched, y_unbatched, atol=1e-5)


def test_flat_embedding():
    # set default embedding args
    args = argparse.Namespace(
        text_encoder="embedding",
        embedding_type="flat",
        input_dim=10000,
        embedding_dim=128,
    )

    # initialize embedding text encoder
    model = multimodal.TextEncoder(args).to(device)
    model.eval()

    # random tensor of input shape
    x, x_len = random_padded_tensor()

    with torch.no_grad():
        # compute batched and unbatched outputs
        y_batched = model(x, x_len)
        y_unbatched = model._forward_unbatched(x, x_len)

        # compare
        assert torch.allclose(y_batched, y_unbatched, atol=1e-5)
        
        
def test_lstm():
    # create cartesian product of lstm configs
    embedding_types = ["flat", "spatial"]
    bidirectional_types = [True, False]
    product = list(itertools.product(embedding_types, bidirectional_types))

    # test each config
    for embedding_type, bidirectional in product:
        # set default embedding args
        args = argparse.Namespace(
            text_encoder="lstm",
            bidirectional=bidirectional,
            embedding_type=embedding_type,
            input_dim=10000,
            embedding_dim=128,
        )
     
        # initialize embedding text encoder
        model = multimodal.TextEncoder(args).to(device)
        model.eval()
     
        # random padded tensor
        x, x_len = random_padded_tensor()
        
        with torch.no_grad():
            # compute batched and unbatched outputs
            y_batched = model(x, x_len)
            y_unbatched = model._forward_unbatched(x, x_len)
         
            # compare
            assert torch.allclose(y_batched, y_unbatched, atol=1e-4)
