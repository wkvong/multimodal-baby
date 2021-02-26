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

from dataset import SAYCamEvalDataset, WordDictionary
from train import ImageUtteranceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_classification_task():
    # load model from checkpoint
    model = ImageUtteranceModel.load_from_checkpoint('models/multimodal-epoch=99.ckpt')
    model = model.to(device)

    # get image categories
    categories = os.listdir('/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5/')
    categories.remove('greenery')
    categories.remove('plushanimal')
    categories.remove('carseat')
    categories.remove('couch')

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

        
def cross_situational_task(exp_name, eval_type):
    print(f'running cross situational evaluation with {exp_name} model')
    
    # load model from checkpoint
    if exp_name == 'multimodal_word_embed_random_init':
        model = ImageUtteranceModel.load_from_checkpoint(
            'models/multimodal_word_embed_random_init_epoch=89.ckpt')
    elif exp_name == 'multimodal_rnn_random_init':
        model = ImageUtteranceModel.load_from_checkpoint(
            'models/multimodal_rnn_random_init_epoch=92.ckpt')
    elif exp_name == 'multimodal_rnn_pretrained_init':
        model = ImageUtteranceModel.load_from_checkpoint(
            'models/multimodal_rnn_pretrained_init_epoch=93.ckpt')
        
    model = model.to(device)
     
    # grab validation dataset
    dataset = SAYCamEvalDataset(eval_type=eval_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    if eval_type == 'val':
        df = pd.read_csv('data/validation.csv')
    elif eval_type == 'test':
        df = pd.read_csv('data/test.csv')
        
    results_columns = df.columns.tolist() + ['target_sim', 'foil_one_sim',
                                             'foil_two_sim', 'foil_three_sim', 'correct']
    results = []
     
    # read vocab pickle file
    vocab = WordDictionary()
     
    for idx, (images, label) in enumerate(dataloader):
        # move to gpu
        images = images.to(device)
        label = label.to(device)
     
        # compute matchmap for all examples
        images = images.squeeze(0)  # (target + foils, channel, height, width)
        label = label.squeeze(0)

        image_embeddings = model.image_model(images)  # (target + foils, embedding_size, height, width)

        # modify lang encoder forward pass depending on word embed or rnn
        if 'word_embed' in exp_name:
            label_embedding = model.utterance_model(label)  # (1, embedding_size)
        elif 'rnn' in exp_name:
            label_embedding = model.utterance_model(label, torch.LongTensor([1]))  # (1, embedding_size)
            label_embedding = label_embedding.squeeze(0)  # rnn adds batck dim in, so remove it here

        matchmaps = []
        matchmap_sims = []
         
        # compute matchmap similarity for each image (target + all foils)
        # fig = plt.figure(figsize=(6, 10))
        min_sim = None
        max_sim = None
     
        # calculate matchmap and max sim
        for i in range(len(image_embeddings)):
            image_embedding = image_embeddings[i]
            matchmap = model.compute_matchmap(image_embedding, label_embedding)
            matchmaps.append(matchmap[0])

            # update min and max for plotting
            if min_sim is None or torch.max(matchmap).item() < min_sim:
                min_sim = torch.max(matchmap).item()
            if max_sim is None or torch.max(matchmap).item() > max_sim:
                max_sim = torch.max(matchmap).item()
                
            matchmap_sim = model.compute_matchmap_sim(matchmap).item()
            matchmap_sims.append(matchmap_sim)

        # calculate whether trial is correct
        if matchmap_sims[0] == max_sim:
            correct = True
        else:
            correct = False

        # create heatmaps
        for i in range(len(image_embeddings)):
            matchmap = matchmaps[i]
            
            fig = plt.figure(figsize=(8, 8))
            ax = plt.axes()

            if i == 0:
                img = io.imread(df['target_img_filename'].iloc[idx])
                filename = f'figures/{exp_name}/trial_{idx}_target_heatmap.png'
            elif i == 1:
                img = io.imread(df['foil_one_img_filename'].iloc[idx])
                filename = f'figures/{exp_name}/trial_{idx}_foil_one_heatmap.png'
            elif i == 2:
                img = io.imread(df['foil_two_img_filename'].iloc[idx])
                filename = f'figures/{exp_name}/trial_{idx}_foil_two_heatmap.png'
            elif i == 3:
                img = io.imread(df['foil_three_img_filename'].iloc[idx])
                filename = f'figures/{exp_name}/trial_{idx}_foil_three_heatmap.png'

            plt.imshow(img)
            curr_heatmap = cv2.resize(matchmap.cpu().detach().numpy(), (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_CUBIC)
            plt.imshow(img)
            plt.imshow(curr_heatmap, alpha=0.75, vmin=torch.min(matchmap).item(), vmax=torch.max(matchmap).item())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.tight_layout()

            plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

        print('finished saving heatmaps for eval trial', idx)
            
        # append results
        curr_results = df.iloc[idx].tolist() + matchmap_sims + [correct]
        results.append(curr_results)

    print('saving evaluation results to file!')
    results_df = pd.DataFrame(results, columns=results_columns)
    results_df.to_csv(f'results/{exp_name}_{eval_type}_results.csv', index=False)

if __name__ == "__main__":
    cross_situational_task(exp_name='multimodal_word_embed_random_init', eval_type='val')
    # cross_situational_task(exp_name='multimodal_rnn_random_init', eval_type='val')
    cross_situational_task(exp_name='multimodal_rnn_pretrained_init', eval_type='val')
