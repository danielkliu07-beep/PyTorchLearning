from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class EncoderRnn(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout_p = 0.1):
        
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embeddign(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True) #Applies a multi-layer gated recurent unit RNN to an input sentence

        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, input):
        
        embedded = self.dropout(self.embedding(input)) #Applies dropout to a word embedding of the input
        output, hidden = self.gru(embedded) #Puts the embedding into the RNN and returns its outputs

        return output, hidden
    



