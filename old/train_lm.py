"""
Code to pre-train a language model on utterances from SAYCam
"""
import os
import glob
import json
import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import SAYCamTrainDataset, SAYCamEvalDataset

def generate_lm_dataset():
    # takes in the cleaned utterances and creates train and val datasets
    # consisting of a 90%/10% random split

    transcripts = pd.read_csv('data/frame_utterance_pairs_clean.csv')
    train_len = int(np.floor(0.9 * len(transcripts)))

    utterances = transcripts['utterance']
    np.random.shuffle(utterances.values)

    train_dataset = utterances[:train_len]
    val_dataset = utterances[train_len:]

    train_dataset.to_csv('data/train_lm.csv', index=False)
    val_dataset.to_csv('data/val_lm.csv', index=False)
    

class WordDictionary(object):
    def __init__(self):
        self.path = 'data/frame_utterance_pairs_clean.csv'
        self.transcripts = pd.read_csv(self.path)
        self.utterances = self.transcripts['utterance']

        # create vocab
        self.word2index = {}
        self.index2word = []

        # add tokens
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.add_word('<sos>')
        self.add_word('<eos>')

        # build rest of vocab
        self.build_vocab()

    def build_vocab(self):
        # add words to the dictionary
        for utterance in self.utterances:
            words = utterance.split(' ')
            for word in words:
                self.add_word(word)
        
    def add_word(self, word):
        if word not in self.word2index:
            self.index2word.append(word)
            self.word2index[word] = len(self.index2word) - 1
        return self.word2index[word]

    def __len__(self):
        return len(self.index2word)

    
class WordLevelUtteranceDataset(Dataset):
    # word-level dataset class for training a language model
    def __init__(self, split, max_len):
        if split == 'train':
            self.transcripts = pd.read_csv('data/train_lm.csv')
        elif split == 'val':
            self.transcripts = pd.read_csv('data/val_lm.csv')

        self.vocab = WordDictionary()
        self.split = split
        self.max_len = max_len

    def __getitem__(self, i):
        utterance = self.transcripts['utterance'].iloc[i]  # get utterance
        words = ['<sos>'] + utterance.split(' ') + ['<eos>']  # get words from utterance

        seq_len = len(words) - 1  # subtract one because either remove start/end
        utterance_ids = []
        for word in words:
            try:
                utterance_ids.append(self.vocab.word2index[word])
            except KeyError:
                utterance_ids.append(self.vocab.word2index['<unk>'])

        # create input and output tensors from utterance ids
        if seq_len >= self.max_len:
            x = torch.LongTensor(utterance_ids[:-1])[:self.max_len]
            y = torch.LongTensor(utterance_ids[1:])[:self.max_len]
            seq_len = torch.LongTensor([self.max_len])
        else:
            x = torch.LongTensor(utterance_ids[:-1])
            y = torch.LongTensor(utterance_ids[1:])
            seq_len = torch.LongTensor([seq_len])
            
        return x, y, seq_len

    def __len__(self):
        # hack to skip last batch
        return len(self.transcripts) - (len(self.transcripts) % 32)


class LanguageModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LanguageModel, self).__init__()
        self.hparams = hparams
        self.input_size = self.hparams.input_size
        self.hidden_size = self.hparams.hidden_size
        self.output_size = self.hparams.output_size

        self.train_dataset = WordLevelUtteranceDataset(split='train',
                                                       max_len=self.hparams.max_len)
        self.val_dataset = WordLevelUtteranceDataset(split='val',
                                                     max_len=self.hparams.max_len)
        
        self.dropout = nn.Dropout(self.hparams.dropout_p)
        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, batch_size, self.hidden_size, device=self.device))
        
    def forward(self, x, seq_len, hidden):
        batch_size = x.size(0)
        
        # embed padded sequence and pack 
        embedding = self.dropout(self.embedding(x))
        embedding = embedding.view(-1, batch_size, self.hparams.hidden_size)  # reshape for lstm
        embedding = pack_padded_sequence(embedding, seq_len, enforce_sorted=False)

        # pass through lstm
        output, hidden = self.lstm(embedding, hidden)

        # unpack and reshape to calculate softmax over words
        output, _ = pad_packed_sequence(output)
        output = output.view(-1, self.hidden_size)
        output = self.output(self.dropout(output))
        output = F.log_softmax(output, dim=1)
        return output, hidden

    def training_step(self, batch, batch_idx):
        x, y, seq_len = batch
        batch_size = x.size(0)
        y = y.view(-1)  # flatten outputs
        hidden = self.init_hidden(batch_size)  # initialize hidden embedding
        y_hat, hidden = self.forward(x, seq_len, hidden)
        return {'loss': F.cross_entropy(y_hat, y, ignore_index=0)}  # mask padding index for loss

    def validation_step(self, batch, batch_idx):
        x, y, seq_len = batch
        batch_size = x.size(0)
        y = y.view(-1)  # flatten outputs
        hidden = self.init_hidden(batch_size)  # initialize hidden embedding
        y_hat, hidden = self.forward(x, seq_len, hidden)
        return {'val_loss': F.cross_entropy(y_hat, y, ignore_index=0)}  # mask padding index for loss

    def validation_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss)
        self.log('epoch', self.trainer.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def pad_collate_fn(self, batch):
        x, y, seq_len = zip(*batch)

        # convert to torch tensor
        x = torch.LongTensor(pad_sequence(x, batch_first=True))
        y = torch.LongTensor(pad_sequence(y, batch_first=True))
        seq_len = torch.LongTensor(seq_len)
        
        return x, y, seq_len
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, collate_fn=self.pad_collate_fn, num_workers=8)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size, collate_fn=self.pad_collate_fn, num_workers=8)
        return val_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=1024, type=int)
        parser.add_argument('--input_size', default=7500, type=int)
        parser.add_argument('--hidden_size', default=128, type=int)
        parser.add_argument('--output_size', default=7500, type=int)
        parser.add_argument('--max_len', default=50, type=int)
        parser.add_argument('--dropout_p', default=0.1, type=float)

        # training specific (for this model)
        parser.add_argument('--exp_name', default="lstm_language_model", type=str)
        parser.add_argument('--max_epochs', default=1000, type=int)
        parser.add_argument('--seed', default=0, type=int)

        return parser

def main(hparams):
    # init module
    model = LanguageModel(hparams)

    # set-up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), f'models/{hparams.exp_name}-' + '{epoch:02d}'),
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
        logger=logger,
        limit_val_batches=1.0,
    )

    # fit model
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = LanguageModel.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
