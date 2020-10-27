"""
Multimodal cross-situational word learning model for SAYCam
"""
import os
import glob
import json
import pickle
import numpy as np
from imageio import imread
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import SAYCamTrainDataset, SAYCamEvalDataset
from dataset import pad_collate_fn

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageModel(nn.Module):
    """
    Image model
    """
    def __init__(self, embedding_size):
        super(ImageModel, self).__init__()
        self.embedding_size = embedding_size

        print('using pretrained vision model (resnext)')
        self.image_embed = self.load_pretrained_model()

    def forward(self, x):
        return self.image_embed(x)

    def load_pretrained_model(self):
        # create mobilenetv2 architecture
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        set_parameter_requires_grad(model)  # freeze cnn layers
        model.fc = torch.nn.Linear(in_features=2048, out_features=2765, bias=True)

        # load in checkpoint
        checkpoint = torch.load('models/resnext50_32x4d_augmentation_True_S_5_288.tar')

        # rename checkpoint keys since we are not using DataParallel
        prefix = 'module.'
        n_clip = len(prefix)
        renamed_checkpoint = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()}

        # load state dict 
        model.load_state_dict(renamed_checkpoint)

        # remove classifier head and add 1x1 convolution to map embedding to lower dim
        model = torch.nn.Sequential(*list(model.children())[:-2],
                                    nn.Conv2d(2048, self.embedding_size, 1))
        return model
    

class UtteranceModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(UtteranceModel, self).__init__()
        self.word_embed = nn.Embedding(input_size, embedding_size)

    def forward(self, x):
        return self.word_embed(x)

    def get_embedding(self, x):
        return self.forward(x)
    

class ImageUtteranceModel(pl.LightningModule):
    def __init__(self, hparams):
        super(ImageUtteranceModel, self).__init__()
        self.hparams = hparams
        self.image_model = ImageModel(embedding_size=self.hparams.embedding_size)
        self.utterance_model = UtteranceModel(input_size=7500, embedding_size=self.hparams.embedding_size)

    def forward(self, images, utterances):
        image_embeddings = self.image_model(images)  # (batch_size, embedding_size, height, width)
        utterance_embeddings = self.utterance_model(utterances)  # (batch_size, utterance_length, embedding_size)

        return image_embeddings, utterance_embeddings

    def compute_matchmap(self, image_embedding, utterance_embedding):
        # computes the dot product between all image embeddings for each cell
        # and for all utterances
        matchmap = torch.einsum('ehw,ke->khw', [image_embedding, utterance_embedding])
        return matchmap

    def compute_matchmap_sim(self, matchmap):
        matchmap = matchmap.view(matchmap.size(0), -1)  # flatten image height/width
        matchmap_max, _ = matchmap.max(dim=1)
        matchmap_sim = matchmap_max.mean()
        return matchmap_sim

    def compute_triplet_loss(self, image_embeddings, utterance_embeddings, utterance_lengths):
        batch_size = image_embeddings.size(0)
        loss = 0
        
        for i in range(batch_size):
            # sample mismatches for images and utterances
            # ensures both mismatches are from the same image/utterance pair
            imp_idx = i
            while imp_idx == i:
                imp_idx = np.random.randint(0, batch_size)

            image_imp_idx = imp_idx
            utterance_imp_idx = imp_idx

            # get utterance lengths for slicing
            utterance_len = utterance_lengths[i]
            utterance_imp_len = utterance_lengths[utterance_imp_idx]

            # compute matchmap for matching image-utterance pair
            matchmap = self.compute_matchmap(image_embeddings[i], utterance_embeddings[i][:utterance_len])
            matchmap_sim = self.compute_matchmap_sim(matchmap)

            # compute matchmap for mismatching image, matching utterance
            matchmap_image_imp = self.compute_matchmap(image_embeddings[image_imp_idx], utterance_embeddings[i][:utterance_len])
            matchmap_image_imp_sim = self.compute_matchmap_sim(matchmap_image_imp)

            # compute matchmap for matching image, mismatching utterance
            matchmap_utterance_imp = self.compute_matchmap(image_embeddings[i], utterance_embeddings[utterance_imp_idx][:utterance_imp_len])
            matchmap_utterance_imp_sim = self.compute_matchmap_sim(matchmap_utterance_imp)
            
            # calculate triplet loss
            loss += F.relu(self.hparams.margin - matchmap_sim + matchmap_image_imp_sim) + \
                F.relu(self.hparams.margin - matchmap_sim + matchmap_utterance_imp_sim)

        loss = loss / batch_size
        return loss
    
    def training_step(self, batch, batch_idx):
        images, utterances, utterance_lengths = batch
        image_embeddings, utterance_embeddings = self.forward(images, utterances)
        loss = self.compute_triplet_loss(image_embeddings, utterance_embeddings, utterance_lengths)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        losses = torch.stack([x['loss'] for x in outputs])
        self.log('train_loss', losses.mean())

    def validation_step(self, batch, batch_idx):
        # pass in images and compute matchmap
        images, labels = batch

        # remove batch dim
        images = images.squeeze(0)  # (target + foils, channel, height, width)
        labels = labels.squeeze(0)

        # get image embeddings
        image_embeddings = self.image_model(images)  # (target + foils, embedding_size, height, width)

        # get label embedding for target word only
        label_embedding = self.utterance_model(labels)  # (1, embedding_size)

        # compute matchmap similarity for each image (target + all foils)
        matchmap_sims = []
        for i in range(len(image_embeddings)):
            image_embedding = image_embeddings[i]
            matchmap = self.compute_matchmap(image_embedding, label_embedding)
            matchmap_sim = self.compute_matchmap_sim(matchmap).item()
            matchmap_sims.append(matchmap_sim)

        # convert to torch tensor
        matchmap_tensor = torch.FloatTensor(matchmap_sims)

        return {'matchmap': matchmap_tensor, 'category_label': labels}

    def validation_epoch_end(self, outputs):
        # collect matchmap scores
        matchmap = torch.stack([x['matchmap'] for x in outputs])
        labels = torch.stack([x['category_label'] for x in outputs]).squeeze()
        categories = torch.unique(labels)
        matchmap_max_idx = torch.argmax(matchmap, dim=1)

        # calculate overall accuracy
        val_accuracy = (matchmap_max_idx == 0).sum().item() / len(matchmap)
        val_loss = 1 - val_accuracy
        self.log('val_loss', val_loss)
        self.log('epoch', self.trainer.current_epoch)

        # get vocab
        with open('data/vocab.pickle', 'rb') as f:
            vocab = pickle.load(f)
            word2index = vocab['word2index']
            index2word = {v: k for k, v in word2index.items()}
        
        # calculate accuracy per category
        print('\n')  # because of pytorch lightning
        for category in categories:
            # get max indices for the current category
            category_label = index2word[category.item()]
            category_max_idx = matchmap_max_idx[labels == category]
            category_accuracy = (category_max_idx == 0).sum().item() / len(category_max_idx)
            print(f'category: {category_label}, accuracy: {category_accuracy}, n_evals: {len(category_max_idx)}')

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_dataset = SAYCamTrainDataset()
        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = SAYCamEvalDataset(eval_type='val')
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        return val_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--embedding_size', default=512, type=int)
        parser.add_argument('--margin', default=1.0, type=float)

        # training specific (for this model)
        parser.add_argument('--exp_name', type=str, required=True)
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--seed', default=0, type=int)

        return parser

def main(hparams):
    # init module
    seed_everything(hparams.seed)

    # instantiate model with hparams
    model = ImageUtteranceModel(hparams)

    # set-up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'models/multimodal-{epoch}'),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    # set-up logger
    logger = CSVLogger("logs", name=hparams.exp_name)
    
    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        num_nodes=hparams.nodes,
        checkpoint_callback=checkpoint_callback,
        logger=logger
    )
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = ImageUtteranceModel.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
