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

# set random seed in python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main():
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     
    preprocess = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalizer])
     
    # load embedding checkpoint
    # TODO: redo this procedure for diff seeds?
    checkpoint_name = f"multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
    model = MultiModalLitModel.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()

    # get paths and categories
    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    EVALUATION_FRAMES_DIR = DATA_DIR / "eval" / "test"
    eval_categories = sorted(os.listdir(EVALUATION_FRAMES_DIR))
     
    # get image embeddings
    # first check if the embeddings have been computed
    if os.path.exists("../results/alignment/all_image_features.npy"):
        all_image_features = np.load(
            "../results/alignment/all_image_features.npy")
        mean_image_features = np.load(
            "../results/alignment/mean_image_features.npy")
        all_text_features = np.load(
            "../results/alignment/all_text_features.npy")
    else:
        # otherwise, compute the embeddings
        all_image_features = []
        all_eval_categories = []
        all_image_filenames = []
        
        # get image embeddings
        for eval_category in eval_categories:
            frames = sorted(glob.glob(os.path.join(EVALUATION_FRAMES_DIR, eval_category, "*.jpeg")))
            frames = np.random.choice(frames, size=min(len(frames), 200))
            
            for frame in frames:
                I = preprocess(
                    Image.open(frame).convert('RGB')).unsqueeze(0).to(device)
                image_features, _ = model.model.encode_image(I)
                
                all_image_features.append(
                    image_features.squeeze().detach().cpu().numpy())
                all_eval_categories.append(eval_category)
                all_image_filenames.append(frame)
     
        # get mean image embeddings
        all_image_features = np.array(all_image_features)
        mean_image_features = []
         
        for i in range(len(eval_categories)):
            idxs = [j for j in range(len(all_eval_categories)) if all_eval_categories[j] == eval_categories[i]]
            curr_image_features = all_image_features[idxs]
            curr_mean_image_features = np.mean(curr_image_features, axis=0)
            mean_image_features.append(curr_mean_image_features)
     
        mean_image_features = np.array(mean_image_features)
     
        # load vocab
        VOCAB_FILENAME = DATA_DIR / "vocab.json"
        with open(VOCAB_FILENAME) as f:
            vocab = json.load(f)
     
        eval_categories[3] = "kitty"  # match eval set-up
     
        # get text embeddings
        all_text_features = []
        for eval_category in eval_categories:
            text = torch.tensor([vocab[eval_category]]).unsqueeze(0).to(device)
            text_len = torch.tensor([len(text)], dtype=torch.long).to(device)
            text_features, _ = model.model.encode_text(text, text_len)
            all_text_features.append(
                text_features.squeeze().detach().cpu().numpy())
        all_text_features = np.array(all_text_features)
     
        # save embeddings
        np.save(f"../results/alignment/all_image_features.npy", all_image_features)
        np.save(f"../results/alignment/mean_image_features.npy", mean_image_features)
        np.save(f"../results/alignment/all_text_features.npy", all_text_features)
    
    # combine features to get joint similarity/distance matrix
    combined_features = []
    for i in range(22):
        combined_features.append(mean_image_features[i])
        combined_features.append(all_text_features[i])
    combined_features = np.array(combined_features)
            
    combined_sims = np.zeros((len(combined_features), len(combined_features)))
    for i in range(len(combined_features)):
        for j in range(len(combined_features)):
            x1 = F.normalize(torch.Tensor(combined_features[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(combined_features[j]), p=2, dim=0)
            combined_sims[i, j] = F.cosine_similarity(x1, x2, dim=0)

    # get image-text similarity matrix
    image_text_sims = np.zeros((len(mean_image_features), len(all_text_features)))
    for i in range(len(mean_image_features)):
        for j in range(len(all_text_features)):
            x1 = F.normalize(torch.Tensor(mean_image_features[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(all_text_features[j]), p=2, dim=0)
            image_text_sims[i, j] = F.cosine_similarity(x1, x2, dim=0)
            
    # set-up t-sne model
    tsne_model = TSNE(random_state=1, metric="precomputed", perplexity=5)
     
    # normalize similarity scores to be between 0 and 1
    normalized_sims = (combined_sims-np.min(combined_sims))/(np.max(combined_sims)-np.min(combined_sims))
     
    # invert similarity scores since t-sne takes in a distance matrix
    tsne_df = tsne_model.fit_transform(1 - normalized_sims)

    # convert to pandas dataframe and add additional columns
    tsne_df = pd.DataFrame(tsne_df, columns=["x", "y"])
    tsne_df["eval_category"] = np.repeat(eval_categories, 2)
    tsne_df["modality"] = np.tile(["image", "text"], 22)

    # save as csv
    tsne_df.to_csv(f"../results/alignment/joint_embeddings_tsne.csv", index=False)
    
    # calculate pearson similarity
    # get image and text sims separately
    image_sims = np.zeros((len(mean_image_features), len(mean_image_features)))
    text_sims = np.zeros((len(all_text_features), len(all_text_features)))
     
    for i in range(len(mean_image_features)):
        for j in range(len(mean_image_features)):
            x1 = F.normalize(torch.Tensor(mean_image_features[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(mean_image_features[j]), p=2, dim=0)
            image_sims[i, j] = F.cosine_similarity(x1, x2, dim=0) 
            
    for i in range(len(all_text_features)):
        for j in range(len(all_text_features)):
            x1 = F.normalize(torch.Tensor(all_text_features[i]), p=2, dim=0)
            x2 = F.normalize(torch.Tensor(all_text_features[j]), p=2, dim=0)
            text_sims[i, j] = F.cosine_similarity(x1, x2, dim=0)

    # convert to long form using eval_categories
    image_sims_long = []
    text_sims_long = []
    eval_categories_x = []
    eval_categories_y = []
    for i in range(len(image_sims)):
        for j in range(len(image_sims)):
            image_sims_long.append(image_sims[i, j])
            text_sims_long.append(text_sims[i, j])
            eval_categories_x.append(eval_categories[i])
            eval_categories_y.append(eval_categories[j])

    # combine into pandas dataframe and save as csv
    sims_df = pd.DataFrame({"image_sims": image_sims_long, "text_sims": text_sims_long, "eval_category_x": eval_categories_x, "eval_category_y": eval_categories_y})
    sims_df.to_csv(f"../results/alignment/joint_embeddings_sims.csv", index=False)

    # do the same thing for image_text_sims
    image_text_sims_long = []
    eval_categories_x = []
    eval_categories_y = []

    for i in range(len(image_text_sims)):
        for j in range(len(image_text_sims)):
            image_text_sims_long.append(image_text_sims[i, j])
            eval_categories_x.append(eval_categories[i])
            eval_categories_y.append(eval_categories[j])

    # combine into pandas dataframe and save as csv
    sims_df = pd.DataFrame({"image_text_sims": image_text_sims_long, "eval_category_x": eval_categories_x, "eval_category_y": eval_categories_y})
    sims_df.to_csv(f"../results/alignment/image_text_embeddings_sims.csv", index=False)
     
    # calculate alignment via pearson correlation
    # e.g. alignment between within-category similarities for image and text categories, along the upper triangular of the similarity matrix
    image_sims_upper = image_sims[np.triu_indices_from(image_sims, k=1)]
    text_sims_upper = text_sims[np.triu_indices_from(text_sims, k=1)]
    print(scipy.stats.pearsonr(image_sims_upper, text_sims_upper))

if __name__ == "__main__":
    main()

    
