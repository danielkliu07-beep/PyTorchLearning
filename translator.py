import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator #pre-processing

import numpy as np
import spacy #tokenizer
import random

from torch.utils.tensorboard import SummaryWriter


#Setting up the tokenizer: (can use other tokenizers that aren't spacy)

spacy_ger = spacy.load('de') #german tokenizer
spacy_eng = spacy.load('en') #english tokenizer

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

    #'Hello my name is' -> ['Hello', 'my', 'name', 'is']

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize = tokenizer_ger, lower = True,
               init_token = '<sos>', eos_token = '<eos>')

english = Field(tokenize = tokenizer_eng, lower = True,
                init_token = '<sos>', eos_token = '<eos')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields = (german, english)) #training data

german.build_vocab(train_data, max_size = 10000, min_freq = 2)
english.build_vocab(train_data, max_size = 10000, min_freq = 2)

#Model:

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p) #Prevents overfitting during training by randomly zeroing out elements of an input tensor
        self.embedding = nn.Embedding(num_embeddings = input_size, embedding_dim = embedding_size)
        self.rnn = nn.LSTM(input_size = embedding_size, hidden_layers = hidden_size, num_layers = num_layers, dropout = p)

    def forward(self, x):
        pass



class Decoder(nn.Module):
    pass

class Seq2Seq(nn.Module): #Combines encoder with decoder
    pass 

