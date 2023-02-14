"""
Building a neural network using PyTorch

The number of hidden layers
The number of units in a hidden layer
Activation functions performed at the various layers
The loss function that we try to optimize for
The learning rate associated with the neural network
The batch size of data leveraged to build the neural network
The number of epochs of forward and back-propagation

"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Definition input x (as list) and desired output y
# Note that in the preceding input and output variable initialization, the input and output are a list of lists where
# the sum of values in the input list is the values in the output list.
x = [[1,2], [3,4], [5,6], [7,8]]
y = [[3], [7], [11], [15]]


# Convert the input lists into tensor objects:
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)

# Define the neural network architecture:
import torch.nn as nn

# We will create a class (MyNeuralNet) that can compose our neural network architecture. It is mandatory to inherit
# from nn.Module when creating a model architecture as it is the base class for all neural network modules

class MyNeuralNet(nn.Module):

    # Within the class, we initialize all the components of a neural network using the __init__ method. We should
    # call super().__init__() to ensure that the class inherits nn.Module:
    def __init__(self):
        super().__init__()

        self.input_to_hidden_layer = nn.Linear(2, 8)
        print(self.input_to_hidden_layer)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
    
    
# torch.manual_seed(0)
mynet = MyNeuralNet().to(device)

print(mynet.input_to_hidden_layer.weight)

mynet.parameters()

for par in mynet.parameters():
    print(par)
    
loss_func = nn.MSELoss()

_Y = mynet(X)
loss_value = loss_func(_Y,Y)
print(loss_value)

from torch.optim import SGD
opt = SGD(mynet.parameters(), lr = 0.001)

print(opt)

loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(torch.Tensor.int(loss_value))



plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()
