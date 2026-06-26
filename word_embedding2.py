import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.input_to_hidden = nn.Linear(in_features = 4, out_features = 2, bias = False) #4 inputs and 2 outputs, makes 4 weights for each of the 2 nodes in the hidden layer
        self.hidden_to_output = nn.Linear(in_features = 2, out_features = 4, bias = False) #2 inputs and 4 outputs, makes 2 weights for each of the 4 nodes in the hidden layer

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input):

        hidden = self.input_to_hidden(input) #By passing an input variable to the torch nn.Lineaer object, the Linear object does all of the multiplication and addition for us

        output_values = self.hidden_to_output(hidden) #Calculates the output sums of the activation functions

        return(output_values)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)
    
    def training_step(self, batch, batch_idx):
        
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)

        return loss


