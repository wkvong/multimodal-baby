import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from multimodal.attention_maps import *
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from multimodal.multimodal_lit import MultiModalLitModel

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# selection type for images to generate attention maps for
selection_type = 'random' # 'random' or 'manual'

# load pretrained model
model_checkpoint_name = "multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
model_checkpoint = glob.glob(
    f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{model_checkpoint_name}/epoch*.ckpt")[0]
print(model_checkpoint)
model = MultiModalLitModel.load_from_checkpoint(
    model_checkpoint, map_location=device)
model.eval()

# create imagefolder dataloader
def get_random_indices_per_class(dataset, num_samples_per_class):
    class_indices = {cls: [] for cls in range(len(dataset.classes))}
    
    # Collect indices for each class
    for idx, (_, class_label) in enumerate(dataset.samples):
        class_indices[class_label].append(idx)

    # Shuffle indices within each class
    for cls in class_indices:
        random.shuffle(class_indices[cls])

    # Select the desired number of samples per class
    indices = []
    for cls in class_indices:
        indices.extend(class_indices[cls][:num_samples_per_class])

    return indices

if selection_type == "manual":
    data_dir = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval_manual_filtered_attention_maps"
    num_samples_per_class = 8
elif selection_type == "random":
    data_dir = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/eval_manual_filtered/test"
    num_samples_per_class = 4
    
normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalizer])

clean_dataset = datasets.ImageFolder(data_dir, transform=preprocess)
subset_indices = get_random_indices_per_class(clean_dataset, num_samples_per_class)
clean_subset_dataset = Subset(clean_dataset, subset_indices)
clean_dataloader = DataLoader(clean_subset_dataset, shuffle=False, batch_size=num_samples_per_class)
clean_iterator = iter(clean_dataloader)

# get images/labels/vocab
for idx, (imgs, labels) in enumerate(clean_dataloader):
    inv_imgs = n_inv(imgs)
    grid_img = torchvision.utils.make_grid(inv_imgs, nrow=4).permute((1, 2, 0)).cpu().numpy()

    class_names = clean_dataset.classes
    print(class_names[labels[0].item()])

    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    VOCAB_FILENAME = DATA_DIR / "vocab.json"
    with open(VOCAB_FILENAME) as f:
        vocab = json.load(f)

    if labels[0].item() == "cat":
        labels = torch.LongTensor([vocab["kitty"] for label in labels])
    else:
        labels = torch.LongTensor([vocab[class_names[label.item()]] for label in labels])

    text_features = model.model.encode_text(
        labels.unsqueeze(0).to(device), torch.tensor([1]*len(labels)).to(device))[0]
    text_features = F.normalize(text_features, p=2, dim=-1)

    # determine saliency layer to use
    saliency_layer = "layer4"

    # create attention maps
    attn_maps = []
    all_images = []
    for i in range(len(imgs)):
        attn_map = gradCAM(
            model.vision_encoder.model.to(device),
            imgs[i].unsqueeze(0).to(device),
            text_features[i].unsqueeze(0).to(device),
            getattr(model.vision_encoder.model, saliency_layer),
            normalize_features=model.model.normalize_features,
        )
        attn_maps.append(attn_map.squeeze().detach().cpu().numpy())

    # calculate max attention across all attention maps
    # note: not using this, and normalizing each attention map separately
    max_attn = np.max(attn_maps)

    for i in range(len(imgs)):
        np_img = inv_imgs[i].permute((1, 2, 0)).cpu().numpy()
        attn_map = attn_maps[i]
        attn_map = getAttMap(np_img, attn_map)

        #imshow(ax, getAttMap(np_img, attn_map))
        all_images.extend([np_img, attn_map])

    all_images = torch.tensor(all_images).permute((0, 3, 1, 2))
    reordered_images = []
    for i in range(len(all_images)):
        if i % 2 == 0:
            reordered_images.extend([all_images[i]])
    for i in range(len(all_images)):
        if i % 2 == 1:
            reordered_images.extend([all_images[i]])        

    print(f"Saving attention map to ../figures/attention_maps_random_selection/{class_names[idx]}.jpg")
    grid_img = torchvision.utils.make_grid(reordered_images, nrow=4).permute((1, 2, 0)).cpu().numpy()
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(grid_img)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"../figures/attention_maps_random_selection/{class_names[idx]}.jpg", bbox_inches="tight")
