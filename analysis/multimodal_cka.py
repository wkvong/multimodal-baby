import glob
import os
import pickle
import json
import shutil

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch_cka import CKA

from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def copy_validation_frames():
    """Create a separate folder of a subset of validation images"""
    with open("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/val.json") as f:
        val_data = json.load(f)
     
    source_dir = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/train_5fps"
    dest_dir = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/val_subset"
     
    os.makedirs(dest_dir, exist_ok=True)
     
    val_data = val_data["data"]
    count = 0
    for utterance in val_data:
        frames = utterance["frame_filenames"]
        first_frame = frames[0]
        shutil.copyfile(os.path.join(source_dir, first_frame), os.path.join(dest_dir, first_frame))

def get_validation_dataloader():
    """Create ImageFolder dataloader for validation frames"""
    val_dir = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/val_subset"
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)
    # val_dataset_subset = torch.utils.data.Subset(val_dataset, range(8))

    dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False)
    return dataloader

def run_cka():
    # specify checkpoints
    # model one: contrastive model with frozen vision encoder
    # model two: contrastive model with randomly initialized vision encoder
    checkpoint_one_name = "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
    checkpoint_two_name = "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_1"
    
    # checkpoint_two_name = "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_True_finetune_cnn_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"

    checkpoint_one = glob.glob(
        f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_one_name}/epoch*.ckpt")[0]
    checkpoint_two = glob.glob(
        f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_two_name}/epoch*.ckpt")[0]
    
    # load models
    model_one = MultiModalLitModel.load_from_checkpoint(checkpoint_one, map_location=device)
    model_one = model_one.vision_encoder.model
    model_one.eval()
    
    model_two = MultiModalLitModel.load_from_checkpoint(checkpoint_two, map_location=device)
    model_two = model_two.vision_encoder.model
    model_two.eval()

    cka = CKA(model_one, model_two,
              model1_name="self-supervised",
              model2_name="contrastive-fine-tuned",   
              device=device)

    dataloader = get_validation_dataloader()
    cka.compare(dataloader) # secondary dataloader is optional
    results = cka.export()

    # save results as a pickle file
    with open("../results/cka/self-supervised-vs-self-supervised-cka-0-1.pkl", "wb") as f:
        pickle.dump(results, f)

    cka.plot_results(save_path="../results/cka/self-supervised-vs-self-supervised-cka-0-1.png")

if __name__ == "__main__":
    # copy_validation_frames()
    run_cka()
    

