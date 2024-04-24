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
import model

torch.manual_seed(17)

# These methods will be reused in later works so I plan to do development in python files
# Classes will be called by notebooks to show numerical experiments

# TODO:
# Move this stuff over to jupyter notebook
# This section sets up the multifidelity functions for mapping
# Samples are taken for the low fidelity function regression
# A comparison is generated with the function for an error measurement
fxn_ind = 1
hf_fxn = mfs_functions.expensive_fxn(1)
lf_fxn = mfs_functions.cheap_fxn(1)

X_lf = np.zeros([30,1])
X_lf[:,0] = np.linspace(0.0,1.0,30,endpoint=True)
y_lf = lf_fxn.evaluate(X_lf)
X_lf = torch.tensor(X_lf, dtype=torch.float32)
y_lf = torch.tensor(y_lf, dtype=torch.float32)

X_check = np.zeros([300,1])
X_check[:,0] = np.linspace(0,1.0,300,endpoint=True)
y_check = lf_fxn.evaluate(X_check)
X_check = torch.tensor(X_check, dtype=torch.float32)
y_check = torch.tensor(y_check, dtype=torch.float32)
# training_data = mfs_functions



#test_dataloader = DataLoader(test_data,  batch_size=batch_size)

# TODO:
# Expand this network to use be able to take in number of layers and array of hidden layers sizes
# Add activation function type to input
# Generalize to allow for n-dimensional inputs
class MLPNetwork(torch.nn.Module,model.Model):
    def __init__(self, X, y, X_valid, y_valid, hidden_layer_size=[10,8], activation=tnn.Tanh(), loss = torch.nn.MSELoss()):
        torch.nn.Module.__init__(self)
        model.Model.__init__(self,X,y)

        # Hyper parameters
        batch_size = 30
        self.epochs = 2000        

        # Set loss function
        self.loss = loss

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X_lf,y_lf)), shuffle=True,batch_size=batch_size)
        self.validation_dataloader = DataLoader(list(zip(X_check,y_check)), batch_size=batch_size)

        # Setup network shape
        __, input_size = X.shape
        __, output_size = y.shape
        self._layers = torch.nn.ModuleList([
            tnn.Linear(input_size,hidden_layer_size[0]),
            activation])
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            self.layers.append(
                tnn.Linear(prev_size,layer_size)
            )
            self._layers.append(activation)
            prev_size = layer_size
        self._layers.append(tnn.Linear(prev_size,output_size))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, x):
        for l in self._layers:
            xnext = l(x) # call the layers in sequence
            x = xnext
        return x
    
    def loss_and_accuracy(self, device=DEVICE):
        # Validation loop
        validation_size = len(self.validation_dataloader.dataset)
        error = 0

        with torch.no_grad():
            for X, y in self.validation_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                error = loss(pred, y)

        error /= validation_size
        return error
    
    def train(self, device=DEVICE):
        self.to(device)
        error_history = []
        for t in range(self.epochs):
            # Train loop
            training_size = len(self.training_dataloader.dataset)
            for batch, (X, y) in enumerate(self.training_dataloader):
                X = X.to(device)
                y = y.to(device)
                # Compute prediction and loss
                pred = self.evaluate(X)
                loss_t = self.loss(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss_t.backward()
                self.optimizer.step()

            error = self.loss_and_accuracy()
            print(f"Validation Error: {error} \n")
            error_history.append(error)

        return error_history

    def evaluate(self,x_in):
        return self(x_in)
    
# TODO:    
# Move this to a main file
loss = torch.nn.MSELoss()
lf_model = MLPNetwork(X_lf,y_lf,X_check,y_check,[10,8])
error_history = lf_model.train()

# Plot the results
lf_model.plot_data("LF Model", label = "Data")
lf_model.plot_model(X_check,"LF Model", color = "blue", label="Model")
plt.show()

error = lf_model.loss_and_accuracy()
print(f"Test Error: {error} \n")

