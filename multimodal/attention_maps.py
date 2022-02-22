# Code to create Grad-CAM attention maps
# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb

import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from multimodal.multimodal_data_module import read_vocab, LabeledSEvalDataset, multiModalDataset_collate_fn
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule, MultiModalSAYCamDataset
# from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
        (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, attn_map_filename, blur=True):
    plt.figure()
    plt.imshow(getAttMap(img, attn_map, blur))
    plt.xticks([])
    plt.yticks([])

    # save attention map
    plt.savefig(attn_map_filename, bbox_inches='tight')
    plt.close()


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

# Reference: https://arxiv.org/abs/1610.02391


def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


if __name__ == "__main__":
    # load pretrained model
    model_checkpoint_name = "multimodal_text_encoder_lstm_lr_5e-05_weight_decay_0.1_fix_temperature_True_batch_size_8"
    model_checkpoint = glob.glob(
        f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{model_checkpoint_name}/*.ckpt")[0]
    print(model_checkpoint)
    model = MultiModalLitModel.load_from_checkpoint(
        model_checkpoint, map_location=device)
    model.eval()

    # get data
# directories and filenames
    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_dev.json"
    VOCAB_FILENAME = DATA_DIR / "vocab.json"

    with open(EVAL_DEV_METADATA_FILENAME) as f:
        eval_dev_data = json.load(f)
        eval_dev_data = eval_dev_data["data"]

    # read vocab
    def read_vocab(vocab_filename=VOCAB_FILENAME):
        with open(vocab_filename) as f:
            return json.load(f)

    # get vocab and reverse
    vocab = read_vocab()
    vocab_idx2word = dict((v, k) for k, v in vocab.items())

    def label_to_category(i):
        label_idx = i.item()
        return vocab_idx2word[label_idx]

    # create eval datasets
    eval_dev_dataset = LabeledSEvalDataset(eval_dev_data, vocab)

    # create dataloaders
    eval_dev_dataloader = torch.utils.data.DataLoader(
        eval_dev_dataset, batch_size=1, shuffle=False)

    imgs, label, label_len, _ = eval_dev_dataset.__getitem__(29)
    # inverse normalization step
    from torchvision import transforms
    n_inv = transforms.Normalize(
        [-0.485/0.229, -0.546/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

    # display images
    inv_imgs = imgs.squeeze(0)
    inv_imgs = n_inv(inv_imgs)

    # determine saliency layer to use
    saliency_layer = "layer4"

    # create attention map
    attn_map = gradCAM(
        model.vision_encoder.model,
        imgs[0].unsqueeze(0).to(device),
        model.model.encode_text(label.unsqueeze(0).to(
            device), torch.Tensor([1]).to(device)),
        getattr(model.vision_encoder.model, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    np_img = inv_imgs[0].permute((1, 2, 0)).cpu().numpy()
    blur = True

    print("saving attention map!")
    viz_attn(np_img, attn_map, blur, filename)
