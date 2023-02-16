import glob
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from siuba import group_by, summarize, arrange, filter, mutate, if_else, _
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import torch
from torchvision import transforms
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

# load embedding checkpoint
seed = 0
checkpoint_name = f"multimodal_cnn_dino_True_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_fix_temperature_True_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{seed}"
checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
model = MultiModalLitModel.load_from_checkpoint(checkpoint, map_location=device)
model.eval()

# get image embeddings
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
EVALUATION_FRAMES_DIR = DATA_DIR / "eval" / "test"
eval_categories = sorted(os.listdir(EVALUATION_FRAMES_DIR))

all_image_features = []
all_eval_categories = []
all_image_filenames = []

for eval_category in eval_categories:
    frames = sorted(glob.glob(os.path.join(EVALUATION_FRAMES_DIR, eval_category, "*.jpeg")))
    print(eval_category, len(frames))
    frames = np.random.choice(frames, size=min(len(frames), 200), replace=False)
    
    for frame in frames:
        I = preprocess(Image.open(frame).convert('RGB')).unsqueeze(0).to(device)
        image_features, _ = model.model.encode_image(I)
        all_image_features.append(image_features.squeeze().detach().cpu().numpy())
        all_eval_categories.append(eval_category)
        
        # rename frame
        frame_path = frame.split(os.path.sep)
        frame = os.path.join(*frame_path[-4:])
        all_image_filenames.append(frame)

VOCAB_FILENAME = DATA_DIR / "vocab.json"
with open(VOCAB_FILENAME) as f:
    vocab = json.load(f)
    
all_text_features = []
eval_categories[3] = "kitty"  # match eval set-up

for eval_category in eval_categories:
    text = torch.tensor([vocab[eval_category]]).unsqueeze(0).to(device)
    text_len = torch.tensor([len(text)], dtype=torch.long).to(device)
    text_features, _ = model.model.encode_text(text, text_len)
    all_text_features.append(text_features.squeeze().detach().cpu().numpy())

# plot image embeddings
n_components = 2
tsne = TSNE(n_components, perplexity=7.5)
tsne_result = tsne.fit_transform(all_image_features)
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': all_eval_categories, 'filename': all_image_filenames})
fig = plt.figure(figsize=(20, 20))

ax = plt.gca()
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, s=100, legend="auto", palette="Paired", linewidth=0, alpha=0.8)
lim = (tsne_result.min()-5, tsne_result.max()+5)

# compute cosine similarities for each eval category
fig = plt.figure(figsize=(50, 40))
for i, curr_category in enumerate(eval_categories):
    category_idx = eval_categories.index(curr_category)
    sims = []
    for image_feature in all_image_features:
        sim = np.power(np.dot(image_feature, all_text_features[category_idx]) / (np.linalg.norm(image_feature)*np.linalg.norm(all_text_features[category_idx])), 2)
        sims.append(sim)

    sims = np.array(sims)
    sims = (sims - np.min(sims))/np.ptp(sims)

    tsne_result_df[curr_category] = sims

    ax = fig.add_subplot(5, 5, i+1)
    ax.set_title(curr_category)
    # ax = plt.gca()
    sns.scatterplot(x='tsne_1', y='tsne_2', data=tsne_result_df, s=25, color='blue', legend='full', alpha=sims)
    lim = (tsne_result.min()-5, tsne_result.max()+5)

# save to CSV
tsne_result_df.to_csv(f"../results/alignment/joint_embeddings_with_eval_sims_seed_{seed}.csv", index=False)
