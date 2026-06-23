import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import SGD 

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN_train(nn.Module): #A copy of the first neural network but with the back propogation function and a trainable model

    def __init__(self): 

        super().__init__() 

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False) 
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) #Setting requires_grad to True tells PyTorch to optimize this parameter

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00 
        top_relu_output = F.relu(input_to_top_relu) 
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    


input_doses = torch.linspace(start=0, end=1, steps=11) 
print(input_doses)

model = BasicNN_train() 
output_values = model(input_doses)
print(output_values)

sns.set_theme(style="whitegrid")
sns.lineplot(x = input_doses, y = output_values.detach(), color = 'green', linewidth = 2.5) #.detach() strips away the gradient values

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()



inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.,]) #What the output should be like

optimizer = SGD(model.parameters(), lr = 0.1) #Creates Stocastic Gradient Descent optimizer with learning rate of 0.1
#Optimizes every parameter in which requires_grad = True

print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

#Backpropogation step
for epoch in range(100): #epoch - each time our optimization code sees all of the training data

    total_loss = 0 #Stores the loss of the model

    for iteration in range(len(inputs)):

        input_i = inputs[iteration]
        label_i = labels[iteration]

        output_i = model(input_i)

        loss = (output_i - label_i)**2 #loss function (squared residual)
        
        loss.backward() #Calculates the derivative of the loss function with respect to the parameters we want to optimize
        #loss.backward() also adds the derivative calculated to the previous derivative, thus accumulating the derivatives each time we go through the nested loop

        total_loss += float(loss)
    
    if (total_loss < 0.0001): #Stops early if total_loss is small (final_bias is optimized good enough)
        print("Num steps: " + str(epoch))
        break

    #Changing the actual value of the model:
    optimizer.step() #Uses the derivatives stored in the model to step the model in the correct direction
    optimizer.zero_grad() #Zeros out the derivatives that we're storing in model

    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")


output_values = model(input_doses)
print(output_values)

sns.set_theme(style="whitegrid")
sns.lineplot(x = input_doses, y = output_values.detach(), color = 'green', linewidth = 2.5) #.detach() strips away the gradient values

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()



