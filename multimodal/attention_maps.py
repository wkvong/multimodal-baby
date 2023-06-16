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

# inverse normalization step
n_inv = transforms.Normalize(
    [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])


def normalize(x: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Normalize to [0, 1].
    """
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    print(f"normalizing: [vmin, vmax] = [{vmin:.6f}, {vmax:.6f}] to [0, 1]")
    x = x - vmin
    vmax = vmax - vmin
    if vmax > 0:
        x = x / vmax
    return x


def preprocess_attn_map(attn_map, shape, interpolation='cubic', blur=False,
                        vmin=None, vmax=None, cmap=None, **kwargs):
    if attn_map.shape != shape:
        import cv2
        attn_map = cv2.resize(
            attn_map, shape[::-1],
            interpolation=getattr(cv2, "INTER_" + interpolation.upper()))
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(shape))
    attn_map = normalize(attn_map, vmin=vmin, vmax=vmax)
    if cmap is not None:
        cmap = plt.get_cmap(cmap)
        attn_map_c = np.delete(cmap(attn_map), 3, 2)
    else:
        attn_map_c = None
    return attn_map, attn_map_c


def getAttMap(img, attn_map, blur=True, cmap='viridis', **kwargs):
    attn_map, attn_map_c = preprocess_attn_map(
        attn_map, img.shape[:2], blur=blur, cmap=cmap, **kwargs)
    attn_map_weights = (attn_map ** 0.7).reshape(attn_map.shape + (1,))
    attn_map = (1 - attn_map_weights) * img + attn_map_weights * attn_map_c
    return attn_map


def imshow(ax, img: np.ndarray):
    ax.imshow(img)
    ax.axis("off")


def plot_image(ax, img, attn_map=None, text=None, overlying=True,
               alpha=0.8, cmap='Greys_r', **kwargs):
    if overlying:
        imshow(ax, img)
        if attn_map is not None:
            attn_map, _ = preprocess_attn_map(
                attn_map, img.shape[:2], cmap=None, **kwargs)
            ax.imshow(attn_map, alpha=alpha, cmap=cmap)
    else:
        if attn_map is not None:
            img = getAttMap(img, attn_map, cmap=cmap, **kwargs)
        imshow(ax, img)

    if text is not None:
        ax.text(0, 1, text, color='black', backgroundcolor='white')


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module, requires_grad : bool = True):
        self.data = None
        self.requires_grad = requires_grad
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        if self.requires_grad:
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


def gradCAM_with_act_and_grad(act, grad):
    # Global average pool gradient across spatial dimension
    # to obtain importance weights.
    alpha = grad.mean(dim=(2, 3), keepdim=True)
    # Weighted combination of activation maps over channel dimension.
    gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
    # We only want neurons with positive influence
    # so we clamp any negative ones.
    gradcam = torch.clamp(gradcam, min=0)

    return gradcam


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module,
    normalize_features: bool = False,
    resize: bool = True,
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

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    gradcam = gradCAM_with_act_and_grad(act, grad)

    # Resize gradcam to input resolution.
    if resize:
        gradcam = F.interpolate(
            gradcam,
            input.shape[2:],
            mode='bicubic',
            align_corners=False)

    return gradcam


if __name__ == "__main__":
    from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
    from multimodal.coco_captions_data_module import COCOCaptionsDataModule
    # from multimodal.multimodal import MultiModalModel
    from multimodal.multimodal_lit import MultiModalLitModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    fig, ax = plt.subplots()
    imshow(ax, getAttMap(np_img, attn_map))
    # save attention map
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
