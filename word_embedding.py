import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform #Initialize the weights in the network uniformly
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbeddingFromScratch(L.LightningModule):

    def __init__(self):
        super().__init__()

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample()) #Initializes w1 to a random number between -0.5 and 0.5
        
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample()) 
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss() #Does the softmax for us


    def forward(self, input):
        
        input = input[0]

        inputs_to_top_hidden = ((input[0] * self.input1_w1) +
                                (input[1] * self.input2_w1) +
                                (input[2] * self.input3_w1) +
                                (input[3] * self.input4_w1))
        
        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) +
                                (input[1] * self.input2_w2) +
                                (input[2] * self.input3_w2) +
                                (input[3] * self.input4_w2))
        
        #Identity function (x = y), so no activation function is used

        output1 = ((inputs_to_top_hidden * self.output1_w1) + (inputs_to_bottom_hidden * self.output1_w2))
        output2 = ((inputs_to_top_hidden * self.output2_w1) + (inputs_to_bottom_hidden * self.output2_w2))
        output3 = ((inputs_to_top_hidden * self.output3_w1) + (inputs_to_bottom_hidden * self.output3_w2))
        output4 = ((inputs_to_top_hidden * self.output4_w1) + (inputs_to_bottom_hidden * self.output4_w2))

        output_presoftmax = torch.stack([output1, output2, output3, output4]) #Stacks the 4 outputs into 1 variable

        return(output_presoftmax)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)

    def training_step(self, batch, batch_idx):
        
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])

        return loss



inputs = torch.tensor([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

labels = torch.tensor([[0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.],
                       [0., 1., 0., 0.]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

modelFromScratch = WordEmbeddingFromScratch()

print("Before optimization, the parameters are...")
# for name, param in modelFromScratch.named_parameters(): #Prints all of the named parameters in the class
#     print(name, param.data)

#Makes a dictionary of all of the named parameters in the class
data = {
    "w1": [modelFromScratch.input1_w1.item(), #The item() function returns the tensor values as python numbers
           modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(),
           modelFromScratch.input4_w1.item()],
    "w2": [modelFromScratch.input1_w2.item(), 
           modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(),
           modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data) #Sorts the data into a pandas data structure
print(df) #Prints the dictionary

trainer = L.Trainer(max_epochs = 100)
trainer.fit(modelFromScratch, train_dataloaders = dataloader)

data = {
    "w1": [modelFromScratch.input1_w1.item(), 
           modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(),
           modelFromScratch.input4_w1.item()],
    "w2": [modelFromScratch.input1_w2.item(), 
           modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(),
           modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)


softmax = nn.Softmax(dim = 0) #dim=0 -> apply to rows of output values

#Testing the model's predicted outputs
print(torch.round(softmax(modelFromScratch(torch.tensor([[1.,0.,0.,0.]]))),
                  decimals = 2))