import argparse
import glob
import json
import os
import random
from pathlib import Path
import numpy as np
import scipy

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torchvision import transforms
from multimodal.multimodal_lit import MultiModalLitModel

from PIL import Image

import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set random seed in python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# get paths and categories
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
OBJECTS_EVALUATION_FRAMES_DIR = DATA_DIR / "object_categories"

def load_model():
    # load model
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalizer])

    # load cvc embedding checkpoint 
    checkpoint_name = f"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
    model = MultiModalLitModel.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()

    return model, preprocess

def load_vocab():
    # load vocab
    VOCAB_FILENAME = DATA_DIR / "vocab.json"
    with open(VOCAB_FILENAME) as f:
        vocab = json.load(f)

    return vocab

def load_data():
    """Load multimodal saycam data"""
    with open(DATA_DIR / 'train.json') as f:
        train_data = json.load(f)
        train_data = train_data["data"]
        train_df = pd.DataFrame(train_data)
        train_df["split"] = "train"

    with open(DATA_DIR / 'val.json') as f:
        val_data = json.load(f)
        val_data = val_data["data"]
        val_df = pd.DataFrame(val_data)
        val_df["split"] = "val"

    with open(DATA_DIR / 'test.json') as f:
        test_data = json.load(f)
        test_data = test_data["data"]
        test_df = pd.DataFrame(test_data)
        test_df["split"] = "test"

    # concatenate train, val and test data
    multimodal_saycam_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return multimodal_saycam_df


def compare_eval_image_vs_text_embeddings(model, preprocess, vocab):     
    # get eval categories sorted by performance
    # note to self: copied directly from R script
    eval_categories = ["apple", "crib", "butterfly", "juice", "broom", "hat", "tree", "dog", "chair", "ring", "cake", "sandwich", "rock", "bell", "clock", "turtle", "ball", "bucket", "train", "cat", "necklace", "leaves", "microwave", "socks", "scissors", "tv", "backpack", "cheese", "bed", "button", "jacket", "pants", "bench", "balloon", "tricycle", "bowl", "stool", "toothpaste", "bird", "bottle", "cookie", "airplane", "shoe", "desk", "sofa", "phone", "watch", "coin", "key", "umbrella", "spoon", "pen", "pipe", "kayak", "guitar", "pizza", "hairbrush", "basket", "camera", "bagel", "stamp", "bike", "fan", "knife"]
    
    # get average image embedding for each category
    avg_image_embeddings = []
    for eval_category in eval_categories:
        image_embeddings = []
        frames = sorted(glob.glob(os.path.join(OBJECTS_EVALUATION_FRAMES_DIR, eval_category, "*.jpg")))
        print(f"Processing {eval_category} with {len(frames)} frames")
        for frame in frames:
            I = preprocess(Image.open(frame).convert(
                'RGB')).unsqueeze(0).to(device)
            image_features, _ = model.model.encode_image(I)
            image_embeddings.append(
                image_features.squeeze().detach().cpu().numpy())

        avg_image_embeddings.append(np.mean(image_embeddings, axis=0))

    avg_image_embeddings = np.array(avg_image_embeddings)
        
    # get text embeddings for each category
    text_embeddings = []
    for eval_category in eval_categories:
        text = torch.tensor(
            [vocab[eval_category]]).unsqueeze(0).to(device)
        text_len = torch.tensor(
            [len(text)], dtype=torch.long).to(device)
        text_features, _ = model.model.encode_text(text, text_len)
                
        text_embeddings.append(
                text_features.squeeze().detach().cpu().numpy())
    text_embeddings = np.array(text_embeddings)

    # print size of image and text embeddings
    print("image embeddings shape: ", avg_image_embeddings.shape)
    print("text embeddings shape: ", text_embeddings.shape)

    # calculate cosine similarity between image and text embeddings
    sims = np.zeros((len(eval_categories), len(eval_categories)))
    for i in range(len(eval_categories)):
        for j in range(len(eval_categories)):
            x1 = F.normalize(torch.Tensor(avg_image_embeddings[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(text_embeddings[j]), p=2, dim=0)
            sims[i, j] = F.cosine_similarity(x1, x2, dim=0)
            
    # plot sims as a heatmap
    # matplotlib.rcParams['figure.dpi'] = 300
    # fig, ax = plt.subplots(figsize=(30, 30))
    # im = ax.imshow(sims)
    # ax.set_xticks(np.arange(len(eval_categories)))
    # ax.set_yticks(np.arange(len(eval_categories)))
    # ax.set_xticklabels(eval_categories)
    # ax.set_yticklabels(eval_categories)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #             rotation_mode="anchor")
    # for i in range(len(eval_categories)):
    #     for j in range(len(eval_categories)):
    #         text = ax.text(j, i, round(sims[i, j], 2),
    #                         ha="center", va="center", color="w")
    # ax.set_title("Cosine Similarity Between Image and Text Embeddings")
    # fig.tight_layout()
    # plt.savefig("../figures/konkle-objects-image-text-cosine-similarity.png")

    # calculate cosine similarity between image mean and text embeddings
    combined_embeddings = np.concatenate((avg_image_embeddings, text_embeddings), axis=0)
    print("combined embeddings shape: ", combined_embeddings.shape)
        
    all_sims = np.zeros((len(eval_categories)*2, len(eval_categories)*2))
    for i in range(len(eval_categories)*2):
        for j in range(len(eval_categories)*2):
            x1 = F.normalize(torch.Tensor(combined_embeddings[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(combined_embeddings[j]), p=2, dim=0)
            all_sims[i, j] = F.cosine_similarity(x1, x2, dim=0)

    # set-up t-sne model
    tsne_model = TSNE(random_state=1, metric="precomputed", perplexity=7.5)

    # normalize similarity scores to be between 0 and 1
    normalized_sims = (all_sims - all_sims.min()) / (all_sims.max() - all_sims.min())

    # invert similarity scores since t-sne takes in a distance matrix
    tsne_df = tsne_model.fit_transform(1 - normalized_sims)

    # convert to pandas dataframe and add additional columns
    tsne_df = pd.DataFrame(tsne_df, columns=["x", "y"])
    tsne_df["eval_category"] = np.tile(eval_categories, 2)
    tsne_df["embedding_type"] = np.concatenate((np.repeat("image_mean", len(eval_categories)), np.repeat("text", len(eval_categories))))
            
    # save as csv
    tsne_df.to_csv(
        "../results/alignment/konkle-objects-avg-image-text-embeddings_seed_0.csv", index=False)

            
def compare_train_image_vs_eval_image_embeddings(model, preprocess, vocab, data):
    # get eval categories sorted by performance
    # note to self: copied directly from R script
    eval_categories = ["apple", "crib", "butterfly", "juice", "broom", "hat", "tree", "dog", "chair", "ring", "cake", "sandwich", "rock", "bell", "clock", "turtle", "ball", "bucket", "train", "cat", "necklace", "leaves", "microwave", "socks", "scissors", "tv", "backpack", "cheese", "bed", "button", "jacket", "pants", "bench", "balloon", "tricycle", "bowl", "stool", "toothpaste", "bird", "bottle", "cookie", "airplane", "shoe", "desk", "sofa", "phone", "watch", "coin", "key", "umbrella", "spoon", "pen", "pipe", "kayak", "guitar", "pizza", "hairbrush", "basket", "camera", "bagel", "stamp", "bike", "fan", "knife"]

    # get average train image embedding for each category
    train_data = data[data["split"] == "train"]
    avg_train_image_embeddings = []

    for eval_category in eval_categories:
        print("Processing training frames for", eval_category)
        image_embeddings = []
        for i, utterance in enumerate(train_data['utterance'].tolist()):
            words = utterance.split()
            if eval_category in words:
                # get frame filename from i
                frame = train_data.iloc[i][
                    "frame_filenames"][0]
                I = preprocess(Image.open(DATA_DIR / "train_5fps" / frame).convert(
                    'RGB')).unsqueeze(0).to(device)
                image_features, _ = model.model.encode_image(I)
                image_embeddings.append(
                    image_features.squeeze().detach().cpu().numpy())

        avg_train_image_embeddings.append(np.mean(image_embeddings, axis=0))
    
    # get average evaluation image embedding for each category
    avg_eval_image_embeddings = []
    for eval_category in eval_categories:
        image_embeddings = []
        frames = sorted(glob.glob(os.path.join(OBJECTS_EVALUATION_FRAMES_DIR, eval_category, "*.jpg")))
        print(f"Processing {eval_category} with {len(frames)} frames")
        for frame in frames:
            I = preprocess(Image.open(frame).convert(
                'RGB')).unsqueeze(0).to(device)
            image_features, _ = model.model.encode_image(I)
            image_embeddings.append(
                image_features.squeeze().detach().cpu().numpy())

        avg_eval_image_embeddings.append(np.mean(image_embeddings, axis=0))

    avg_eval_image_embeddings = np.array(avg_eval_image_embeddings)

    # calculate cosine similarity between train and eval image embeddings
    sims = np.zeros((len(eval_categories)))
    for i in range(len(eval_categories)):
        x1 = F.normalize(torch.Tensor(avg_train_image_embeddings[i]), p=2, dim=0)
        x2 = F.normalize(torch.Tensor(avg_eval_image_embeddings[i]), p=2, dim=0)
        sims[i] = F.cosine_similarity(x1, x2, dim=0)

    print(sims)
        
    # calculate rank correlation between eval_categories ordering and sims
    correlation = scipy.stats.spearmanr(eval_categories, sims)
    print(correlation)

    # plot sims as a scatterplot
    matplotlib.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(eval_categories, sims)
    ax.set_title("Cosine Similarity Between Train and Eval Image Embeddings")
    ax.set_xlabel("Eval Category")
    ax.set_ylabel("Cosine Similarity")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.savefig("../figures/konkle-objects-train-eval-image-cosine-similarity.png")
    
def main():
    model, preprocess = load_model()
    vocab = load_vocab()
    data = load_data()
    compare_eval_image_vs_text_embeddings(model, preprocess, vocab)
    compare_train_image_vs_eval_image_embeddings(model, preprocess, vocab, data)

if __name__ == "__main__":
    main()
