from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import device, optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from translator.dataloader import SOS_token

max_length = 10


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

        #The last output of the encoder, or the context vector
        return output, hidden

class DecoderRnn(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(DecoderRnn, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.Gru(hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):

        batch_size = encoder_outputs.size(0)
        
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(SOS_token) #Creates a empty tensor of batch_size x 1 filled with the SOS token
        decoder_hidden = encoder_hidden #Sets the first hidden state to the encoder's outputs (the encoder's last hidden state)
        decoder_outputs = []

        for i in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output) #Adds the output to the list of outputs

            if target_tensor is not None:
                #Implements teacher forcing -> feeding the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1) #Returns the single largest element along with its index location
                decoder_input = topi.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)

        return decoder_outputs, decoder_hidden, None #None is returned for consistency in the training loop
    

    def forward_step(self, input, hidden):
        
        output = self.embedding(input) #Applies an embedding to the input
        output = F.relu(output) #Puts it through the Relu activation function
        output, hidden = self.gru(output, hidden) #Puts it into a RNN
        output = self.out(output) #Puts it into a fully connected layer

        return output

class BahdanauAttention(nn.Module):
    
    def __init__(self, hidden_size):

        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim = -1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class AttnDecoderRnn(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p = 0.1):

        super(AttnDecoderRnn, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    
    def forward_step(self, input, hidden, encoder_outputs):

        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim = 2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights



    



