import os
import pickle
import imageio as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dataset import SAYCamValDataset
from train import ImageUtteranceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model from checkpoint
model = ImageUtteranceModel.load_from_checkpoint('models/pretrained-epoch=86.ckpt')
model = model.to(device)

# generate examples from validation dataset
n_evaluations = 1
n_foils = 3
val_dataset = SAYCamValDataset(n_evaluations, n_foils)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# read vocab pickle file
with open('data/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)
    word2index = vocab['word2index']
    index2word = {v: k for k, v in word2index.items()}


for idx, (images, label, filenames, all_labels) in enumerate(val_dataloader):
    print(idx)
    
    # move to gpu
    images = images.to(device)
    label = label.to(device)

    # get all labels
    all_labels = list(all_labels.squeeze().numpy())
    all_labels = [index2word[label] for label in all_labels]
     
    # compute matchmap for all examples
    images = images.squeeze()  # (target + foils, channel, height, width)
    image_embeddings = model.image_model(images)  # (target + foils, embedding_size, height, width)
    label_embedding = model.utterance_model(label)  # (1, embedding_size)
    matchmap_sims = []
     
    # compute matchmap similarity for each image (target + all foils)
    fig = plt.figure(figsize=(6, 10))
    
    for i in range(len(image_embeddings)):
        image_embedding = image_embeddings[i]
        matchmap = model.compute_matchmap(image_embedding, label_embedding)
        matchmap_sim = model.compute_matchmap_sim(matchmap).item()
        matchmap_sims.append(matchmap_sim)

        ax = fig.add_subplot(4, 2, 2*i+1)
        print(filenames[i][0])
        img = io.imread(filenames[i][0])
        plt.imshow(img)
        plt.title(f'category: {all_labels[i]}, sim: {matchmap_sim:4f}')
        plt.axis('off')
        
        curr_heatmap = resize(matchmap[0].cpu().detach().numpy(), [img.shape[0], img.shape[1]])
        ax = fig.add_subplot(4, 2, 2*i+2)
        plt.imshow(img)
        plt.imshow(curr_heatmap, alpha=0.75)
        plt.axis('off')


    plt.legend([])
    plt.savefig(f'viz/eval/eval_{idx}.png')
    plt.close()
