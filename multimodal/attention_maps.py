# Code to create Grad-CAM attention maps
# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb

import glob

import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
# from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel

# inverse normalization step
n_inv = transforms.Normalize(
    [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    print("normalizing:")
    print("min:", x.min())
    print("max:", x.max())
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


def viz_attn(img, attn_map, blur=True, with_img=False, attn_map_filename=None):
    attn_map = getAttMap(img, attn_map, blur)
    if with_img:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[1].imshow(attn_map)
        for ax in axes:
            ax.axis("off")
    else:
        plt.figure()
        plt.imshow(attn_map)
        plt.xticks([])
        plt.yticks([])

    if attn_map_filename is not None:
        # save attention map
        print("saving attention map!")
        plt.savefig(attn_map_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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
    layer: nn.Module,
    normalize_features: bool = False,
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
        if normalize_features:
            output = F.normalize(output, p=2, dim=1)
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
    # parse empty args
    parser = _setup_parser()
    data_args = parser.parse_args("")
    # set args
    for key, value in model.args.items():
        setattr(data_args, key, value)
    # make the train dataloader deterministic
    data_args.augment_frames = False

    # build data module
    dataset_name = getattr(data_args, "dataset", "saycam")
    DataModuleClass = {
        "saycam": MultiModalSAYCamDataModule,
        "coco": COCOCaptionsDataModule,
    }[dataset_name]
    data = DataModuleClass(data_args)
    data.prepare_data()
    data.setup()

    vocab = data.read_vocab()

    # dataset
    eval_dataset = data.eval_datasets["val"]

    # dataloader
    eval_dataloader = {
        "dev": data.val_dataloader,
        "test": data.test_dataloader,
    }["dev"]()[1]

    imgs, label, label_len, raw_label = eval_dataset[29]
    label_len = torch.tensor(label_len)

    # display images
    inv_imgs = imgs.squeeze(0)
    inv_imgs = n_inv(inv_imgs)

    # determine saliency layer to use
    saliency_layer = "layer4"

    # get text features
    text_features = model.model.encode_text(
        label.unsqueeze(0).to(device), label_len.unsqueeze(0).to(device))[0]
    if model.model.normalize_features:
        text_features = F.normalize(text_features, p=2, dim=1)

    # create attention map
    attn_map = gradCAM(
        model.vision_encoder.model,
        imgs[0].unsqueeze(0).to(device),
        text_features,
        getattr(model.vision_encoder.model, saliency_layer),
        normalize_features=model.model.normalize_features,
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    np_img = inv_imgs[0].permute((1, 2, 0)).cpu().numpy()
    blur = True

    viz_attn(np_img, attn_map, blur, attn_map_filename=filename)
