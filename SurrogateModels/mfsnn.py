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
from torch.utils.data import DataLoader

import mfs_functions
import model

# These methods will be reused in later works so I plan to do development in python files
# Classes will be called by notebooks to show numerical experiments


# Class for a linear network of variable shape for regression purposes
class MLPNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size = [10,8], activation = tnn.Tanh()):
        super().__init__()

        # Setup network shape
        self._layers = torch.nn.ModuleList([
            tnn.Linear(input_size, hidden_layer_size[0]),
            activation])
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            self._layers.append(
                tnn.Linear(prev_size,layer_size)
            )
            self._layers.append(activation)
            prev_size = layer_size
        self._layers.append(tnn.Linear(prev_size,output_size))

    def forward(self, x):
        for l in self._layers:
            xnext = l(x) # call the layers in sequence
            x = xnext
        return x

# Low Fidelity NN class is used in MFS setup as the initial regression solution.
# Outside of this context it is simply a multi-layer NN for regression.
# Inherites model.Model within this project to access nice plotting features
class LowFidelityNN(model.Model):
    def __init__(self, X, y, X_valid, y_valid, hidden_layer_size=[10,8], activation=tnn.Tanh(), loss = torch.nn.MSELoss()):
        model.Model.__init__(self,X,y)

        # Hyper parameters
        batch_size = 30
        self.epochs = 500        

        # Report rate
        self.report_steps = 50

        # Set loss function
        self.loss = loss

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X,y)), shuffle=True,batch_size=batch_size)
        self.validation_dataloader = DataLoader(list(zip(X_valid,y_valid)), batch_size=batch_size)

        # Generate the neural network
        __, input_size = X.shape
        __, output_size = y.shape
        self.mlp_network = MLPNetwork(input_size, output_size, hidden_layer_size, activation)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.mlp_network.parameters(), lr=0.05)

        self.training_error, self.validation_error = self.train()        
   
    # Calculates the training and validation errors using the given loss function
    def loss_and_accuracy(self, device=DEVICE):
        # Validation loop
        validation_size = len(self.validation_dataloader.dataset)
        training_size = len(self.training_dataloader.dataset)
        train_error = 0
        validation_error = 0

        with torch.no_grad():
            for X, y in self.validation_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                validation_error = self.loss(pred, y)

            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                train_error = self.loss(pred, y)


        validation_error /= validation_size
        train_error /= training_size

        return train_error, validation_error
    
    # Train uses backpropogation to solve for the model parameters
    def train(self, device=DEVICE):
        self.mlp_network.to(device)
        train_error_history = []
        validation_error_history = []
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

            train_error, validation_error = self.loss_and_accuracy()
            if t%self.report_steps==0:
                print(f"----Progress Update----")
                print(f"Epoch: {t} \n")
                print(f"Training Error: {train_error} \n")
                print(f"Validation Error: {validation_error} \n")
            train_error_history.append(train_error)
            validation_error_history.append(validation_error)

        return train_error_history, validation_error_history

    # Evaluate function defines method for use in generic model class
    def evaluate(self,x_in):
        return self.mlp_network(x_in)
    
    # Plots errors output from training to show path to parameters
    def plot_error(self):
        epoch_number = np.arange(0,self.epochs,1)
        plt.figure("Model Error")
        plt.title("Model Error")
        plt.plot(epoch_number, self.training_error, color = "blue", label = "Training", linestyle = "-")        
        plt.plot(epoch_number, self.validation_error, color = "black", label = "Validation", linestyle = "--")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

# Function used for testing LowFidelity NN capabilities
# further showcase of class can be found in MFSNN.ipynb
def TestFunction():
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
    loss = torch.nn.MSELoss()
    lf_model = LowFidelityNN(X_lf,y_lf,X_check,y_check,[10,10,10])
    #error_history = lf_model.train()

    # Plot the results
    lf_model.plot_data("LF Model", label = "Data")
    lf_model.plot_model(X_check,"LF Model", color = "blue", label="Model")
    plt.show()

    lf_model.plot_error()
