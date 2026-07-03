import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam 

import lightning as L 
from torch.utils.data import TensorDataset, DataLoader

class LightningLSTM(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(input_size = 1, hidden_size = 1) #hidden_size = number of outputs
    
    def forward(self, input):

        input_trans = input.view(len(input), 1) #Turns input into a row with 1 column

        lstm_out, temp = self.lstm(input_trans) #lstm_out contains the short-term memory values for each LSTM unit we unrolled

        prediction = lstm_out[-1] #Last element in the array
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)
    
    def training_step(self, batch, batch_idx): 
        
        input_i, label_i = batch
        output_i = self.forward(input_i[0]) 
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss) 

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss

model = LightningLSTM()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.0])).detach())
print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.0])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset) #Make it easy to access data in batches and shuffle the data each epoch

trainer = L.Trainer(max_epochs = 300, log_every_n_steps = 2)
trainer.fit(model, train_dataloaders = dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.0])).detach())
print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.0])).detach())


