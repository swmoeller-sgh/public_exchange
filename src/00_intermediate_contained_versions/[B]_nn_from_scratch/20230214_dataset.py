
# Import the methods that help in loading data and dealing with datasets:
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# Import the data, convert the data into floating-point numbers, and register them to a device:
x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)

# Instantiate a class of the dataset – MyDataset:
class MyDataset(Dataset):
    
    # Define an __init__ method that takes input and output pairs and converts them into Torch float objects
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
    
    # Specify the length (__len__) of the input dataset:
    def __len__(self):
        return len(self.x)

    # Finally, the __getitem__ method is used to fetch a specific row:
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    

# Create an instance of the defined class:
ds = MyDataset(X, Y)

# Pass the dataset instance defined previously through DataLoader to fetch the batch_size number of data points from
# the original input and output tensor objects:
dl = DataLoader(ds, batch_size=2, shuffle=True)

# how to print the input and output batches of data
for x,y in dl:
    print(x,y)
    
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
    
mynet = MyNeuralNet().to(device)
loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(mynet.parameters(), lr = 0.001)

import time
loss_history = []
start = time.time()
for _ in range(50):
    for data in dl:
        x, y = data
        opt.zero_grad()
        loss_value = loss_func(mynet(x),y)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end = time.time()
print(end - start)

val_x = [[10,11]]

val_x = torch.tensor(val_x).float().to(device)

print(mynet(val_x))


print(ds)
