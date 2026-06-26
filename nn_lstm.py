import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam #Fits the neural network to the data (Alternative to SGD)

import lightning as L #Makes training easier to code
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule): #Inherits from Lightning module

    def __init__(self): #Creates and initializes all the weight and bias tensors needed to implement an LSTM unit

        super().__init__() #Call the initalization method for the parent class Lightning Module

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True) #Initializes the weight to a randomly generated number from a normal distribution with mean = 0 and standard deviation = 1
        self.wlr2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad = True) #Set biases to 0 when starting out

        self.wpr1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wpr2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad = True) 

        self.wp1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wp2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad = True) 

        self.wo1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wo2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad = True) 

    def lstm_unit(self, input_value, long_memory, short_memory): #Does the LSTM math

        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)

        updated_long_memory = ((long_memory * long_remember_percent) + (potential_remember_percent * potential_memory))

        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return([updated_long_memory, updated_short_memory])


    def forward(self, input): #Forward pass through the unrolled LSTM unit

        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory) #Updates long and short term memory using the LSTM function

        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory


    def configure_optimizers(self): #Configures the optimizer we want to use (SGD, ADAM, etc)
       
       return Adam(self.parameters())
    
    def training_step(self, batch, batch_idx): #Calculates the loss and logs the progress we're making during training
        
        input_i, label_i = batch
        output_i = self.forward(input_i[0]) #Uses the forward function to make a prediction with the batch training data
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss) #From the LightningModule - creates a new file called lightning_logs that stores the data

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss

model = LSTMbyHand()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.0])).detach())
print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.0])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset) #Make it easy to access data in batches and shuffle the data each epoch

trainer = L.Trainer(max_epochs = 2000)
trainer.fit(model, train_dataloaders = dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.0])).detach())
print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.0])).detach())

