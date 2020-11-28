import os
import glob
import pickle
import imageio as io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage.transform import resize
import cv2

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dataset import SAYCamEvalDataset
from train import ImageUtteranceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_classification_task():
    # load model from checkpoint
    model = ImageUtteranceModel.load_from_checkpoint('models/multimodal-epoch=99.ckpt')
    model = model.to(device)

    # get image categories
    categories = os.listdir('/misc/vlgscratch4/LakeGroup/shared_data/S_labeled_data/S_labeled_data_1fps_4')
    categories.remove('greenery')
    categories.remove('plushanimal')

    # read vocab pickle file
    with open('data/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
        word2index = vocab['word2index']
        index2word = {v: k for k, v in word2index.items()}

    # get categories indices
    labels = torch.LongTensor([word2index[category] for category in categories])
    labels = labels.unsqueeze(0).to(device)

    # get validation dataset
    val_dataset = SAYCamEvalDataset(eval_type='val')
    val_dataloader = iter(DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1))
    val_imgs = pd.read_csv('data/validation.csv')

    fig = plt.figure(figsize=(15, 10))
    
    # get examples
    for i in range(1):
        images, _ = val_dataloader.next()
        target_image = images.squeeze()[0]  # get target image only
     
        # compute matchmap
        target_image = target_image.unsqueeze(0).to(device)
        target_image_embedding = model.image_model(target_image).squeeze()
        label_embeddings = model.utterance_model(labels).squeeze()
     
        # matchmap = model.compute_matchmap(target_image_embedding, label_embeddings)


        img = io.imread(val_imgs['target_img_filename'].iloc[0])

        for j, category in enumerate(categories):
            label = torch.LongTensor([word2index[category]]).unsqueeze(0).to(device)
            label_embedding = model.utterance_model(label).squeeze(0)
            matchmap = model.compute_matchmap(target_image_embedding, label_embedding)
            curr_heatmap = cv2.resize(matchmap[0].cpu().detach().numpy(), (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_CUBIC)
            
            ax = fig.add_subplot(4, 6, j+1)
            plt.imshow(img)
            plt.imshow(curr_heatmap, alpha=0.75)
            plt.title(f'{category}: {torch.max(matchmap[0]).item():2f}')
            plt.axis('off')
     
        # class_probs = F.log_softmax(F.max_pool2d(matchmap, kernel_size=(7, 7)).squeeze(), dim=0)
        # class_probs = F.log_softmax(torch.mean(matchmap, dim=(1, 2)).squeeze(), dim=0)
        # class_probs = class_probs.cpu().detach().numpy()
        # max_class = np.argmax(class_probs)
        # print(f'trial {i}: {categories[max_class]}: {np.exp(class_probs[max_class])}')

    plt.legend([])
    plt.tight_layout()
    plt.savefig(f'viz/image_classification.png')
    plt.close()

        
def cross_situational_task():
    # load model from checkpoint
    model = ImageUtteranceModel.load_from_checkpoint('models/multimodal-epoch=99.ckpt')
    model = model.to(device)
     
    # generate examples from validation dataset
    n_evaluations = 1
    n_foils = 3
    val_dataset = SAYCamEvalDataset(eval_type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_imgs = pd.read_csv('data/validation.csv')
     
    # read vocab pickle file
    with open('data/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
        word2index = vocab['word2index']
        index2word = {v: k for k, v in word2index.items()}
     
    for idx, (images, label) in enumerate(val_dataloader):
        # move to gpu
        images = images.to(device)
        label = label.to(device)
     
        # compute matchmap for all examples
        images = images.squeeze(0)  # (target + foils, channel, height, width)
        label = label.squeeze(0)
        image_embeddings = model.image_model(images)  # (target + foils, embedding_size, height, width)
        label_embedding = model.utterance_model(label)  # (1, embedding_size)
        matchmaps = []
        matchmap_sims = []
         
        # compute matchmap similarity for each image (target + all foils)
        fig = plt.figure(figsize=(6, 10))
        max_sim = 1
     
        # calculate matchmap and max sim
        for i in range(len(image_embeddings)):
            image_embedding = image_embeddings[i]
            matchmap = model.compute_matchmap(image_embedding, label_embedding)
            matchmaps.append(matchmap[0])
     
            if torch.max(matchmap).item() > max_sim:
                max_sim = torch.max(matchmap).item()
                
            matchmap_sim = model.compute_matchmap_sim(matchmap).item()
            matchmap_sims.append(matchmap_sim)
     
        for i in range(len(image_embeddings)):
            matchmap = matchmaps[i] 
            matchmap_sim = matchmap_sims[i]
            ax = fig.add_subplot(4, 2, 2*i+1)
     
            if i == 0:
                img = io.imread(val_imgs['target_img_filename'].iloc[idx])
            elif i == 1:
                img = io.imread(val_imgs['foil_one_img_filename'].iloc[idx])
            elif i == 2:
                img = io.imread(val_imgs['foil_two_img_filename'].iloc[idx])
            elif i == 3:
                img = io.imread(val_imgs['foil_three_img_filename'].iloc[idx])
                
            plt.imshow(img)
            plt.title(f'sim: {matchmap_sim:4f}')
            plt.axis('off')
            
            # curr_heatmap = resize(matchmap[0].cpu().detach().numpy(), [img.shape[0], img.shape[1]])
            curr_heatmap = cv2.resize(matchmap.cpu().detach().numpy(), (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_CUBIC)
            ax = fig.add_subplot(4, 2, 2*i+2)
            plt.imshow(img)
            plt.imshow(curr_heatmap, alpha=0.75, vmin=np.min(matchmap.cpu().detach().numpy()), vmax=max_sim)
            plt.axis('off')
          
        plt.legend([])
        plt.savefig(f'viz/eval/eval_{idx}.png')
        plt.close()

if __name__ == "__main__":
    # cross_situational_task()
    image_classification_task()
    
