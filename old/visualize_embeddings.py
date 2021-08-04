import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import gensim.downloader

from dataset import SAYCamEvalDataset, WordDictionary
from train import ImageUtteranceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load models
def visualize_word_embeddings():
    word_embed_model = ImageUtteranceModel.load_from_checkpoint(
        'models/multimodal_word_embed_random_init_epoch=89.ckpt').to(device)
    rnn_random_init_model = ImageUtteranceModel.load_from_checkpoint(
        'models/multimodal_rnn_random_init_epoch=92.ckpt').to(device)
    rnn_pretrained_init_model = ImageUtteranceModel.load_from_checkpoint(
        'models/multimodal_rnn_pretrained_init_epoch=93.ckpt').to(device)
     
    # get evaluation categories
    categories = sorted(os.listdir('/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5'))
    categories.remove('greenery')
    categories.remove('plushanimal')
    categories.remove('carseat')
    categories.remove('couch')
     
    # create vocab
    vocab = WordDictionary()
        
    # get categories indices
    labels = torch.LongTensor([[vocab.word2index[category]] for category in categories])
    labels = labels.to(device)
     
    # get embeddings
    word_embed_model_embeddings = word_embed_model.utterance_model(labels).squeeze().detach().cpu().numpy()
     
    seq_len = torch.LongTensor([1] * len(categories)).to(device)
    rnn_random_init_model_embeddings = rnn_random_init_model.utterance_model(labels, seq_len).squeeze().detach().cpu().numpy()
     
    rnn_pretrained_init_model_embeddings = rnn_pretrained_init_model.utterance_model(labels, seq_len).squeeze().detach().cpu().numpy()
     
    pca = PCA(n_components=2)
    pca.fit(word_embed_model_embeddings)
    word_embed_model_pca_embeddings = pca.transform(word_embed_model_embeddings)
    print(word_embed_model_pca_embeddings)
     
    plt.figure(figsize=(10, 10))
    for i in range(len(word_embed_model_pca_embeddings)):
        x = word_embed_model_pca_embeddings[i][0]
        y = word_embed_model_pca_embeddings[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x + 0.05, y + 0.05 , categories[i], fontsize=12)
     
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.title('pca plot of evaluation category embeddings with the word embedding only model')
    plt.savefig('figures/word_embed_pca.png')
     
    pca = PCA(n_components=2)
    pca.fit(rnn_random_init_model_embeddings)
    rnn_random_init_model_pca_embeddings = pca.transform(rnn_random_init_model_embeddings)
    print(rnn_random_init_model_pca_embeddings)
     
    plt.figure(figsize=(10, 10))
    for i in range(len(rnn_random_init_model_pca_embeddings)):
        x = rnn_random_init_model_pca_embeddings[i][0]
        y = rnn_random_init_model_pca_embeddings[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x + 0.05, y + 0.05 , categories[i], fontsize=12)
     
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.title('pca plot of evaluation category embeddings with the rnn random init model')
    plt.savefig('figures/rnn_random_init_pca.png')
     
    pca = PCA(n_components=2)
    pca.fit(rnn_pretrained_init_model_embeddings)
    rnn_pretrained_init_model_pca_embeddings = pca.transform(rnn_pretrained_init_model_embeddings)
    print(rnn_pretrained_init_model_pca_embeddings)
     
    plt.figure(figsize=(10, 10))
    for i in range(len(rnn_pretrained_init_model_pca_embeddings)):
        x = rnn_pretrained_init_model_pca_embeddings[i][0]
        y = rnn_pretrained_init_model_pca_embeddings[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x + 0.05, y + 0.05 , categories[i], fontsize=12)
     
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.title('pca plot of evaluation category embeddings with the rnn pretrained init model')
    plt.savefig('figures/rnn_pretrained_init_pca.png')
     
    # word2vec
    word_vectors = gensim.downloader.load('word2vec-google-news-300')
    word2vec_embeddings = []
    for category in categories:
        word2vec_embeddings.append(word_vectors[category])
     
    word2vec_embeddings = np.array(word2vec_embeddings)
    print(word2vec_embeddings)
     
    pca = PCA(n_components=2)
    pca.fit(word2vec_embeddings)
    word2vec_pca_embeddings = pca.transform(word2vec_embeddings)
    print(word2vec_pca_embeddings)
     
    plt.figure(figsize=(10, 10))
    for i in range(len(word2vec_pca_embeddings)):
        x = word2vec_pca_embeddings[i][0]
        y = word2vec_pca_embeddings[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x + 0.05, y + 0.05 , categories[i], fontsize=12)
     
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.title('pca plot of evaluation category embeddings with word2vec')
    plt.savefig('figures/word2vec_pca.png')


def visualize_image_embeddings():
    # load pretrained model
    model = torchvision.models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=2765, bias=True)

    # load in checkpoint
    checkpoint = torch.load('models/resnext50_32x4d_augmentation_True_S_5_288.tar')

    # rename checkpoint keys since we are not using DataParallel
    prefix = 'module.'
    n_clip = len(prefix)
    renamed_checkpoint = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()}

    # load state dict 
    model.load_state_dict(renamed_checkpoint)

    # get evaluation categories
    categories = sorted(os.listdir('/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5'))

    # set transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # create ImageFolder
    image_dir = '/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5'
    dataset = ImageFolder(root=image_dir, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    
    images, labels = iter(dataloader).next()

    outputs = model(images)

    
if __name__ == "__main__":
    visualize_image_embeddings()
