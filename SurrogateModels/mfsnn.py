import torch as t
import torch.nn as tnn
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data

t.manual_seed(17)

## Doing it the Pytorch way without using our custom feature extraction
DEVICE='cpu'
DTYPE=t.float32
import numpy as np
import torch
import torch.nn
import torch.optim
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import mfs_functions

torch.manual_seed(17)

# These methods will be reused in later works so I plan to do development in python files
# Classes will be called by notebooks to show numerical experiments

# TODO:
# Plan is for the dataset here to be replaced with analytical functions
# Found easy nparray -> pytorch tensor methods while searching for how to use tensors

# Getting the dataset, the Pytorch way
all_training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# TODO:
# Validation data won't be needed in my implementation
training_data, validation_data = torch.utils.data.random_split(all_training_data, [0.9, 0.1])
fxn_ind = 1
hf_fxn = mfs_functions.expensive_fxn(1)
lf_fxn = mfs_functions.cheap_fxn(1)

X_lf = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
y_lf = lf_fxn.evaluate(X_lf)

X_lf = torch.tensor(X_lf, dtype=torch.float64)
y_lf = torch.tensor(y_lf, dtype=torch.float64)

X_check = np.linspace(0,1.0,300,endpoint=True)
y_check = lf_fxn.evaluate(X_check)
X_check = torch.tensor(X_check, dtype=torch.float64)
y_check = torch.tensor(y_check, dtype=torch.float64)

# training_data = mfs_functions

# Hyper parameters
learning_rate = 1e-3 # controls how fast the 
batch_size = 11
epochs = 5
momentum = 0.9

training_dataloader = DataLoader(list(zip(X_lf,y_lf)), shuffle=True, batch_size=batch_size)
validation_dataloader = DataLoader(list(zip(X_check,y_check)),  batch_size=batch_size)
#test_dataloader = DataLoader(test_data,  batch_size=batch_size)

# TODO:
# Define own loss function for mfsm loss from paper
loss = torch.nn.MSELoss()

# TODO:
# Expand this network to use be able to take in number of layers and array of hidden layers sizes
# Add activation function type to input
# Generalize to allow for n-dimensional inputs
class MLPNetwork(torch.nn.Module):
    def __init__(self, hidden_size=10, nclasses=1, input_size=11):
        super().__init__()
        self._layers = torch.nn.ModuleList([torch.nn.Flatten(),
            tnn.Linear(input_size, hidden_size),
            tnn.ReLU(),
            tnn.Linear(hidden_size, nclasses)])
    def forward(self, x):
        for l in self._layers:
            xnext = l(x) # call the layers in sequence
            x = xnext
        return x
    
# TODO:    
# Move this to a main file
model = MLPNetwork()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# TODO
# Either embrace and figure out cuda or remove it from this method
# Current issues are due to CUDA not being enabled
def loss_and_accuracy(model, loss, validation_dataloader, device=DEVICE):
    # Validation loop
    validation_size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in validation_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(DTYPE).sum().item()

    test_loss /= num_batches
    correct /= validation_size
    return test_loss, correct
    
def train(model, loss, training_dataloader, validation_dataloader, device=DEVICE):
    model.to(device)
    train_losses = []
    valid_losses = []
    for t in range(epochs):
        # Train loop
        training_size = len(training_dataloader.dataset)
        for batch, (X, y) in enumerate(training_dataloader):
            X = X.to(device)
            y = y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss_t = loss(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss_t, current = loss_t.item(), (batch + 1) * len(X)
                print(f"loss: {loss_t:>7f}  [{current:>5d}/{training_size:>5d}]", end="\r")
                train_losses.append(loss_t)
                valid_loss, correct = loss_and_accuracy(model, loss, validation_dataloader, device=device)
                valid_losses.append(valid_loss)
                print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    return model, train_losses, valid_losses
        
trained_model, train_losses, valid_losses = train(model, loss, training_dataloader, validation_dataloader)

test_loss, correct = loss_and_accuracy(model, loss, validation_dataloader)
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def loss_and_accuracy(model, loss, validation_dataloader, device=DEVICE):
    # Validation loop
    validation_size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in validation_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(DTYPE).sum().item()

    test_loss /= num_batches
    correct /= validation_size
    return test_loss, correct
    
def train(model, loss, training_dataloader, validation_dataloader, device=DEVICE):
    model.to(device)
    train_losses = []
    valid_losses = []
    for t in range(epochs):
        # Train loop
        training_size = len(training_dataloader.dataset)
        for batch, (X, y) in enumerate(training_dataloader):
            X = X.to(device)
            y = y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss_t = loss(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss_t, current = loss_t.item(), (batch + 1) * len(X)
                print(f"loss: {loss_t:>7f}  [{current:>5d}/{training_size:>5d}]", end="\r")
                train_losses.append(loss_t)
                valid_loss, correct = loss_and_accuracy(model, loss, validation_dataloader, device=device)
                valid_losses.append(valid_loss)
                print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    return model, train_losses, valid_losses
        
trained_model, train_losses, valid_losses = train(model, loss, training_dataloader, validation_dataloader)

test_loss, correct = loss_and_accuracy(model, loss, validation_dataloader)
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

torch.nn.__file__

X, _ = next(iter(validation_dataloader))
X.shape

#plt.imshow(X[0, 0])

print("The value ", model(X.to(DEVICE)).argmax(dim=-1)[0].item())