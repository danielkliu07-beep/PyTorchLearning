import numpy as np

import torch #Used to create tensors that store all of the numerical data values
import torch.nn as nn #Used to make the weight and bias tensors part of the neural network
import torch.nn.functional as F #Gives activation functions
from torch.optim import SGD #Gives Stochastic Gradient Descent

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN(nn.Module): #Inherits from a PyTorch class called Module

    def __init__(self): #Creates neural network parameters for each weight and bias

        super().__init__() #Calls the initialization function for the parent class (nn.Module)

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False) #Creates a new weight as a neural network parameter
        #Making a variable a neural network parameters gives us the option to optimize that variable
        #requires_grad (requires gradient) is False since we don't need to optimize the weight, as we're initializing it

        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input): #Passes through the neural network with the input value and calculates an output value using the weights, biases, and activation functions

        input_to_top_relu = input * self.w00 + self.b00 #Get the x value of the Relu activation function    
        top_relu_output = F.relu(input_to_top_relu) #Passes the input to the Relu activation function to get the Y value
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output


input_doses = torch.linspace(start=0, end=1, steps=11) #Creates a tensor with a sequence of 11 values between, and including, 0 and 1
print(input_doses)

model = BasicNN() #Model is the standard variable name used for neural network models when coding with PyTorch
output_values = model(input_doses)
print(output_values)

sns.set_theme(style="whitegrid")
sns.lineplot(x = input_doses, y = output_values, color = 'green', linewidth = 2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.show()


