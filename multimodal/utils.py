from PIL import ImageFilter
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

def msplit(string, delimiters):
    """Split with multiple delimiters."""
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def convert_timestamps_to_seconds(timestamps):
    """Function to convert a variety of starting timestamps from SAYCam transcripts into seconds."""

    new_timestamps = []
    for timestamp in timestamps:
        timestamp = str(timestamp)  # convert to string
        if timestamp != 'nan':
            timestamp_one = msplit(timestamp, '-')[0]  # get starting timestamp

            if timestamp_one != '':
                splits = msplit(timestamp_one, (':', '.', ',', ';'))

                if splits[0] == '':
                    splits[0] = '0'

                if len(splits) == 1:
                    splits.append('0')
                else:
                    # sometimes only the tens of seconds are reported as single digits
                    # this converts these values to seconds
                    if splits[1] == '1':
                        splits[1] = '10'
                    elif splits[1] == '2':
                        splits[1] = '20'
                    elif splits[1] == '3':
                        splits[1] = '30'
                    elif splits[1] == '4':
                        splits[1] = '40'
                    elif splits[1] == '5':
                        splits[1] = '50'

                # trim whitespace
                splits[0] = splits[0].strip()
                splits[1] = splits[1].strip()

                if len(splits[1]) <= 2:
                    # handle proper timestamps
                    timestamp_one_secs = int(splits[0]) * 60 + int(splits[1])
                else:
                    # handle float-like timestamps
                    # TODO: figure out what the floats encode, otherwise
                    # for now just setting them to None
                    timestamp_one_secs = None
         
                new_timestamps.append(timestamp_one_secs)
            else:
                new_timestamps.append(None)  # handles non-empty string that is not a timestamp
        else:
            new_timestamps.append(None)  # handles non-strings like nans

    return new_timestamps


def split_dataset(base_dataset, fraction, seed):
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size]
    )
    
    # return torch.utils.data.random_split(  # type: ignore
    #     base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    # )


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x    


def get_entropy(logits, dim=-1):
    log_p = F.log_softmax(logits, dim=dim)
    return (F.softmax(log_p, dim=dim) * -log_p).sum(dim=dim) # E[- log p] = sum - p log p


def map_structure(fn, *obj):
    r"""Map a function over all elements in a (possibly nested) collection.

    Args:
        fn (callable): The function to call on elements.
        *obj: The collection to map function over.

    Returns:
        The collection in the same structure, with elements mapped.
    """
    if hasattr(obj[0], "--no-map--"):
        return fn(*obj)
    if isinstance(obj[0], list):
        return [map_structure(fn, *x) for x in zip(*obj)]
    if isinstance(obj[0], tuple):
        if isinstance(obj[0], torch.Size):
            return fn(*obj)
        if hasattr(obj[0], '_fields'):  # namedtuple
            return type(obj[0])(*[map_structure(fn, *x) for x in zip(*obj)])
        else:
            return tuple(map_structure(fn, *x) for x in zip(*obj))
    if isinstance(obj[0], dict):
        return {k: map_structure(fn, *[o[k] for o in obj])
                for k in obj[0].keys()}
    if isinstance(obj[0], set):
        assert len(obj) == 1, "map_structure can only accept one set"
        return {map_structure(fn, x) for x in obj[0]}
    return fn(*obj)


def apply_permutation(tensor: torch.Tensor, permutation, dim: int) -> torch.Tensor:
    return tensor.index_select(dim, permutation)

# adapted from https://github.com/eminorhan/silicon-menagerie/blob/master/utils.py
def load_model(model_name, pretrained=True):

    alg, data, model_spec = model_name.split("_")

    # checks
    assert alg in ["dino", "mugs", "mae"], "Unrecognized algorithm!"
    assert data in ["say", "s", "sfp", "a", "y", "imagenet100", "imagenet10", "imagenet3", "imagenet1"], "Unrecognized data!"
    assert model_spec in ["resnext50", "vitb14", "vitl16", "vitb16", "vits16"], "Unrecognized architecture!"

    if model_spec == "resnext50":
        arch, patch_size = "resnext50_32x4d", None
    elif model_spec == "vitb14":
        arch, patch_size = "vit_base", 14
    elif model_spec == "vitl16":
        arch, patch_size = "vit_large", 16
    elif model_spec == "vitb16":
        arch, patch_size = "vit_base", 16
    elif model_spec == "vits16":
        arch, patch_size = "vit_small", 16

    # download checkpoint from hf
    checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")

    if alg == "dino" or alg == "mugs":
        model = build_dino_mugs(arch, patch_size)
        if pretrained:
            print("Loading pretrained weights for DINO resnext model")
            load_dino_mugs(model, checkpoint, "teacher")
    elif alg == "mae":
        model = build_mae(arch, patch_size)
        if pretrained:
            load_mae(model, checkpoint)

    return model

def load_dino_mugs(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix induced by MAE
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

def build_dino_mugs(arch, patch_size):
    import multimodal.vision_transformer_dino_mugs as vits
    from torchvision import models as torchvision_models

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if arch in vits.__dict__.keys():
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    # otherwise, we check if the architecture is in torchvision models
    elif arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[arch]()
        model.fc = torch.nn.Identity()
    else:
        print(f"Unknown architecture: {arch}")
        sys.exit(1)

    return model
