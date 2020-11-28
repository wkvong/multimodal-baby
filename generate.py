import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl
from train_lm import LanguageModel, WordDictionary

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# load model and set to eval
model = LanguageModel.load_from_checkpoint('models/lstm_lm_embedding_size_128_dropout-epoch=999.ckpt')
model.eval()

# set up dictionary
vocab = WordDictionary()

# decoder settings
temperature = 1.0
batch_size = 1
max_len = 50

with torch.no_grad():
    for i in range(20):
        utterance = []
        input = torch.LongTensor([[vocab.word2index['<sos>']]])
        seq_len = torch.LongTensor([1])
        hidden = model.init_hidden(batch_size)
         
        while len(utterance) <= max_len:
            output, hidden = model(input, seq_len, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]

            if word_idx == vocab.word2index['<eos>']:
                break
            else:
                input.fill_(word_idx)
                word = vocab.index2word[word_idx]
                utterance.append(word)
         
        print(' '.join(utterance))
