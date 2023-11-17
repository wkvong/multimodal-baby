import glob
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import entropy
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

from multimodal.multimodal_lit import MultiModalLitModel

def main():
    # create data frame columns

    # get eval categories
    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    LABELED_S_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5")
    EVALUATION_FRAMES_DIR = DATA_DIR / "eval" / "test"
    eval_categories = sorted(os.listdir(EVALUATION_FRAMES_DIR))
    
    # get training data
    with open(DATA_DIR / 'train.json') as f:
        train_data = json.load(f)
        train_data = train_data["data"]

    # load vocab
    with open(DATA_DIR / 'vocab.json') as f:
        vocab = json.load(f)
     
    # get word frequency of eval_categories in train_data
    word_freq = {}
    for category in eval_categories:
        word_freq[category] = 0
        for data in train_data:
            if category in data["utterance"].split():
                word_freq[category] += 1

    print(word_freq)

    # image frequency
    labeled_s_categories = os.listdir(LABELED_S_DIR)
    labeled_s_categories.remove("carseat")
    labeled_s_categories.remove("couch")
    labeled_s_categories.remove("greenery")
    labeled_s_categories.remove("plushanimal")
    total_images = 0
    img_freq = {}

    for category in labeled_s_categories:
        category_dir = LABELED_S_DIR / category
        num_images = len(os.listdir(category_dir))
        total_images += num_images
        img_freq[category] = num_images

    print(img_freq)
    
    # contextual diversity
    contextual_diversity = {}
    for eval_category in eval_categories:
        # get all utterances from train with eval_category
        utterances = [utterance['utterance'] for utterance in train_data if eval_category in utterance['utterance']]
        eval_category_counts = np.zeros(len(vocab))
     
        # loop over utterances and get index of eval_category in each utterance
        for utterance in utterances:
            words = utterance.split()
            # get index of eval_category in utterance, otherwise skip if not found
            try:
                eval_category_index = words.index(eval_category)
                # get adjacent indices to eval_category_index
                context_utterance_idxs = [eval_category_index - 1,
                    eval_category_index + 1]
         
                # get vocab idxs of context words
                # return UNK if word not in vocab
                context_idxs = []
                for i, idx in enumerate(context_utterance_idxs):
                    if idx >= 0 and idx < len(words):
                        word = words[idx]
                        if word in vocab:
                            context_idxs.append(vocab[words[idx]])
                        else:
                            context_idxs.append(vocab['<unk>'])
                
                # increment eval_category_counts by one for each context_idx
                for context_idx in context_idxs:
                    eval_category_counts[context_idx] += 1
            except ValueError:
                continue
     
        # calculate entropy of eval_category_counts
        eval_category_entropy = entropy(eval_category_counts)
        # get non-zero entries of eval_category_counts
        eval_category_nonzero_counts = eval_category_counts[eval_category_counts > 0]
        
        contextual_diversity[eval_category] = eval_category_entropy

    print(contextual_diversity)
    
    # visual diversity
    # load model
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     
    preprocess = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalizer])
     
    # load embedding checkpoint
    checkpoint_name = f"multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
    model = MultiModalLitModel.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()

    visual_diversity = {}
    for eval_category in eval_categories:
        # get all utterances from train with eval_category
        utterances = [utterance for utterance in train_data if eval_category in utterance['utterance'].split()]
         
        # get all frame_filenames from utterances and flatten
        frame_filenames = [utterance['frame_filenames'][0] for utterance in utterances]
         
        # read in images, convert to PIL and turn into a batch of tensors
        images = [Image.open(os.path.join(DATA_DIR, "train_5fps", frame_filename)).convert('RGB') for frame_filename in frame_filenames]
        images = [preprocess(image) for image in images]
        images = torch.stack(images).to(device)
         
        # pass through model and get eval_category_features
        with torch.no_grad():
            eval_category_features, _ = model.model.encode_image(images)
            eval_category_features = eval_category_features.view(eval_category_features.size(0), -1)
            eval_category_features = eval_category_features.cpu().numpy()
         
        # calculate standard deviation of eval_category_features
        eval_category_feature_std = np.mean(np.std(eval_category_features, axis=0))
        visual_diversity[eval_category] = eval_category_feature_std

    print(visual_diversity)

    # convert word_freq, img_freq, contextual_diversity, visual_diversity to 
    regression_df = pd.DataFrame([word_freq, img_freq, contextual_diversity, visual_diversity]).T
    regression_df.columns = ["word_freq", "img_freq", "contextual_diversity", "visual_diversity"]
    regression_df["eval_category"] = eval_categories

    # save regression_df to csv
    regression_df.to_csv("../results/summary/regression_df.csv", index=False)

if __name__ == "__main__":
    main()
    
