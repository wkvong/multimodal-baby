import glob
import json
import os
import random
from pathlib import Path
import numpy as np
import scipy
import pandas as pd
from siuba import group_by, summarize, arrange, filter, mutate, if_else, _
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import torch
from torchvision import transforms
import torch.nn.functional as F
from multimodal.multimodal_lit import MultiModalLitModel
import clip
from sklearn.manifold import TSNE
from PIL import Image

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# visualize embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalizer,
        ])

# list of models
model_names = ['cvcl', 'cvcl (shuffled)', 'cvcl (random features)']

for model_name in model_names:
    # load embedding checkpoint
    seed = 0
    if model_name == 'cvcl':
        checkpoint_name = f"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{seed}"
    elif model_name == 'cvcl (shuffled)':
        checkpoint_name = f"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_shuffle_utterances_True_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{seed}"
    elif model_name == 'cvcl (random_features)':
        checkpoint_name = f"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_pretrained_cnn_False_finetune_cnn_False_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{seed}"
    
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
    model = MultiModalLitModel.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()

    # get image embeddings
    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    EVALUATION_FRAMES_DIR = DATA_DIR / "eval" / "test"
    eval_categories = sorted(os.listdir(EVALUATION_FRAMES_DIR))

    all_image_features = []
    mean_image_features = []
    all_eval_categories = []
    all_image_filenames = []

    n_samples = 100

    for eval_category in eval_categories:
        frames = sorted(glob.glob(os.path.join(EVALUATION_FRAMES_DIR, eval_category, "*.jpeg")))
        print(eval_category, len(frames))
        frames = np.random.choice(frames, size=min(len(frames), n_samples), replace=False)
        curr_category_features = []

        # get individual frame features
        for frame in frames:
            I = preprocess(Image.open(frame).convert('RGB')).unsqueeze(0).to(device)
            image_features, _ = model.model.encode_image(I)
            all_image_features.append(image_features.squeeze().detach().cpu().numpy())
            curr_category_features.append(image_features.squeeze().detach().cpu().numpy())
            all_eval_categories.append(eval_category)
        
            # rename frame
            frame_path = frame.split(os.path.sep)
            frame = os.path.join(*frame_path[-4:])
            all_image_filenames.append(frame)

        # calculate mean frame features from all frames from current eval_category
        curr_category_features = np.array(curr_category_features)
        mean_image_features.append(np.mean(curr_category_features, axis=0))

    VOCAB_FILENAME = DATA_DIR / "vocab.json"
    with open(VOCAB_FILENAME) as f:
        vocab = json.load(f)

    # calculate text features
    all_text_features = []
    eval_categories[3] = "kitty"  # match eval set-up

    for eval_category in eval_categories:
        text = torch.tensor([vocab[eval_category]]).unsqueeze(0).to(device)
        text_len = torch.tensor([len(text)], dtype=torch.long).to(device)
        text_features, _ = model.model.encode_text(text, text_len)
        all_text_features.append(text_features.squeeze().detach().cpu().numpy())

    # combine image and text embeddings
    # all_features = np.concatenate([all_image_features, mean_image_features, all_text_features], axis=0)
    # print(all_features.shape)

    # # compute similarity matrix
    # all_sims = np.zeros((len(all_features), len(all_features)))
    # for i in range(len(all_features)):
    #     for j in range(len(all_features)):
    #         x1 = F.normalize(torch.tensor(all_features[i]), p=2, dim=0)
    #         x2 = F.normalize(torch.tensor(all_features[j]), p=2, dim=0)
    #         all_sims[i, j] = F.cosine_similarity(x1, x2, dim=0)

    # # normalize similarity matrix
    # all_sims = (all_sims - np.min(all_sims)) / (np.max(all_sims) - np.min(all_sims))
    
    # # get t-SNE of image and text embeddings and put into a dataframe
    # n_components = 2
    # tsne = TSNE(n_components, random_state=1, metric="precomputed", perplexity=7.5)
    # tsne_result = tsne.fit_transform(1 - all_sims)
    # tsne_result_df = pd.DataFrame(tsne_result, columns=["x", "y"])

    # # append additional information to dataframe
    # all_eval_categories = all_eval_categories + eval_categories + eval_categories
    # all_image_filenames = all_image_filenames + [None] * len(eval_categories) + [None] * len(eval_categories)
    # embedding_type = ["image"] * len(all_image_features) + ["image_mean"] * len(mean_image_features) + ["text"] * len(all_text_features)

    # tsne_result_df["eval_category"] = all_eval_categories
    # tsne_result_df["image_filename"] = all_image_filenames
    # tsne_result_df["embedding_type"] = embedding_type

    # # compute cosine similarities for each eval category
    # for i, curr_category in enumerate(eval_categories):
    #     category_idx = eval_categories.index(curr_category)
    #     sims = []
    #     for image_feature in all_image_features:
    #         sim = np.dot(image_feature, all_text_features[category_idx])
    #         sims.append(sim)

    #     # add extra zeros for mean image and text embeddings
    #     for j in range(len(eval_categories)):
    #         sims.append(0)
        
    #     for j in range(len(eval_categories)):
    #         sims.append(0)
        
    #     sims = np.array(sims)

    #     tsne_result_df[curr_category] = sims
 
    # # save to CSV
    # if model_name == 'cvcl':
    #     embeddings_filename = f"../results/alignment/cvcl_joint_embeddings_with_eval_sims_seed_{seed}.csv"
    #     print(f'Saving embeddings to {embeddings_filename}')
    #     tsne_result_df.to_csv(embeddings_filename, index=False)
    # elif model_name == 'cvcl (shuffled)':
    #     embeddings_filename = f"../results/alignment/cvcl_shuffled_joint_embeddings_with_eval_sims_seed_{seed}.csv"
    #     print(f'Saving embeddings to {embeddings_filename}')
    #     tsne_result_df.to_csv(embeddings_filename, index=False)
    # elif model_name == 'cvcl (random features)':
    #     embeddings_filename = f"../results/alignment/cvcl_random_features_joint_embeddings_with_eval_sims_seed_{seed}.csv"
    #     print(f'Saving embeddings to {embeddings_filename}')
    #     tsne_result_df.to_csv(embeddings_filename, index=False)

    # calculate pearson correlation of mean image-text sims
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

    # calculate alignment via pearson correlation
    # e.g. alignment between within-category similarities for image and text categories, along the upper triangular of the similarity matrix
    image_sims_upper = image_sims[np.triu_indices_from(image_sims, k=1)]
    text_sims_upper = text_sims[np.triu_indices_from(text_sims, k=1)]
    print(f"Mean image-text correlation for {model_name}: ", scipy.stats.pearsonr(image_sims_upper, text_sims_upper))
