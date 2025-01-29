import torch as t
import torch.nn as tnn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchview import draw_graph
import os
print(t.__version__)
#t.manual_seed(17)

## Doing it the Pytorch way without using our custom feature extraction
DEVICE='cpu'
DTYPE=t.float32
import numpy as np

import mfs_functions
import mf_model
import model
import data_handler

#t.set_default_tensor_type(t.DoubleTensor)

# Class for a linear network of variable shape for regression purposes
class MLPNetwork(t.nn.Module, model.Model):
    def __init__(self, X, y, hidden_layer_size = [10,8], activation = tnn.ReLU(), loss = t.nn.MSELoss(), 
                    learning_rate=0.001, weight_decay=0.001,batch_size=30, weight_file="network_weights", load_weights=False, write_weights=False, train=True):
        t.nn.Module.__init__(self)
        model.Model.__init__(self,X,y)

        # Hyper parameters
        self.epochs = 50        

        # Report rate
        self.report_steps = 1

        # Set loss function
        self.loss = loss

        # Set batch size
        self.batch_size = batch_size

        # Normalize the y data
        self.datahandler = data_handler.DataHandler(y)
        y = self.datahandler.scale(y)

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=self.batch_size)

        # Generate the neural network
        __, input_size = X.shape
        __, output_size = y.shape
        #self.mlp_network = MLPNetwork(input_size, output_size, hidden_layer_size, activation)

        # Setup network shape
        nn_layers = []
        nn_layers.append(tnn.Linear(input_size, hidden_layer_size[0]))
        if activation != "none":
            nn_layers.append(activation)
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            nn_layers.append(tnn.Linear(prev_size,layer_size))
            if activation != "none":
                nn_layers.append(activation)
            prev_size = layer_size
        nn_layers.append(tnn.Linear(prev_size,output_size))

        self._layers = tnn.Sequential(*nn_layers)
      
        # Apply initial weights
        #self.apply(self.init_weights)
        # Set the optimizer
        self.optimizer = t.optim.LBFGS(self.parameters(), lr = learning_rate, max_iter=400)#, weight_decay = weight_decay)

        # Try loading weights from file
        if load_weights:
            self.load_state_dict(t.load(weight_file))
        else:
            self.training_error = self.train()
        
        if write_weights:
            t.save(self.state_dict(), weight_file)

    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, tnn.Linear):
            tnn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
   
   # Define a forward method here
    def forward(self, x):
        x = self._layers(x)
        return x

    # Evaluate function defines method for use in generic model class
    def evaluate(self,x_in):
        with t.no_grad():
            return self.datahandler.unscale(self.forward(x_in))

    # Calculates the training error using the given loss function
    def training_accuracy(self, device=DEVICE):
        # Validation loop
        training_size = len(self.training_dataloader.dataset)
        train_error = 0

        with t.no_grad():
            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.forward(X)
                train_error = self.loss(pred, y)

        train_error /= training_size

        return train_error   

    # Train uses backpropogation to solve for the model parameters
    def train(self, device=DEVICE):
        self.to(device)
        train_error_history = []
        for t in range(self.epochs):
            # Train loop
            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                def closure():
                    #Compute prediction and loss
                    pred = self.forward(X)
                    loss_t = self.loss(pred, y)

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_t.backward()
                    return loss_t
                
                self.optimizer.step(closure)

            train_error = self.training_accuracy()
            if t%self.report_steps==0:
                print(f"----Progress Update----")
                print(f"Epoch: {t} \n")
                print(f"Training Error: {train_error} \n")
            train_error_history.append(train_error)

        return train_error_history

    def validation_accuracy(self, X_valid, y_valid, device=DEVICE):        
        
        valid_dataloader = DataLoader(list(zip(X_valid,y_valid)), shuffle=True)
        validation_size = len(valid_dataloader.dataset)

        # Validation Loop
        with t.no_grad():
            for X, y in self.valid_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                validation_error = self.loss(pred, y)

        validation_error /=validation_size
        return validation_error
    
    # Plots errors output from training to show path to parameters
    def plot_training_error(self, name):
        epoch_number = np.arange(0,self.epochs,1)
        plt.figure(name)
        plt.title(name)
        plt.plot(epoch_number, self.training_error, color = "blue", label = "Training", linestyle = "-")        
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")

    def visualize_shape(self):
        input_size, __ = self.x.shape
        model_graph = draw_graph(self(),input_size=(input_size,self.batch_size),device="meta")
        model_graph.visual_graph


# Composite Neural Network 
# Explicitly seperates the network into a linear and non-linear network.
# Fully connected NN architecture in both cases is concatenated at the final layer
class CompositeNN(tnn.Module, mf_model.MFModel):
    def __init__(self, X, y, layers_nl = [10,8], layers_linear = [10,10], activation_nl = tnn.Tanh(), loss = t.nn.MSELoss(), learning_rate=0.001, weight_decay=0.001, batch_size=10, train=True):
        t.nn.Module.__init__(self)
        model.Model.__init__(self,X,y)

        # Hyper parameters
        self.epochs = 50        

        # Report rate
        self.report_steps = 1

        # Set loss function
        self.loss = loss

        # Set batch size
        self.batch_size = batch_size

        # Normalize the y data
        self.datahandler = data_handler.DataHandler(y)
        y = self.datahandler.scale(y)

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=self.batch_size)

        # Generate the neural network
        __, input_size = X.shape
        __, output_size = y.shape

        # Setup non-linear network shape
        self._nl_layers = []
        self._nl_layers.append(tnn.Linear(input_size,layers_nl[0]))
        self._nl_layers.append(activation_nl)
        prev_size = layers_nl[0]

        for layer_size in layers_nl[1:-1]:
            self._nl_layers.append(
                tnn.Linear(prev_size,layer_size)
            )
            self._nl_layers.append(activation_nl)
            prev_size = layer_size
        self._nl_layers.append(tnn.Linear(prev_size,output_size))
        self._nl_layers = tnn.Sequential(*self._nl_layers)
      
        # Setup linear network shape
        self._layers_linear = []
        if layers_linear == []:
            self._layers_linear.append(tnn.Linear(prev_size,output_size))
        else:
            self._layers_linear.append(tnn.Linear(input_size, layers_linear[0]))
            prev_size = layers_linear[0]

            for layer_size in layers_linear[1:-1]:
                self._layers_linear.append(
                    tnn.Linear(prev_size,layer_size)
                )
                prev_size = layer_size
            self._layers_linear.append(tnn.Linear(prev_size,output_size))
        self._layers_linear = tnn.Sequential(*self._layers_linear)

        # Apply initial weights
        self.apply(self.init_weights)

        # Set weight for composite network combination
        #self.alpha = tnn.parameter.Parameter(t.tensor([0.99]))
        self.alpha = 0.999

        # Check whether the network is just being setup or if traing is specified
        if train:
            # Set the optimizer and parameters
            self.optimizer = t.optim.LBFGS(self.parameters(), lr = learning_rate, max_iter=400)

            # Run training on network
            self.training_error = self.train()        
    
    def forward(self, x):
        return self.alpha * self._layers_linear(x) + (1.0-self.alpha) * self._nl_layers(x)

    # Evaluate function defines method for use in generic model class
    def evaluate(self,x_in):
        with t.no_grad():
            return self.datahandler.unscale(self.forward(x_in))

    # Calculates the training error using the given loss function
    def training_accuracy(self, device=DEVICE):
        # Validation loop
        training_size = len(self.training_dataloader.dataset)
        train_error = 0

        for X, y in self.training_dataloader:
            X = X.to(device)
            y = y.to(device)
            with t.no_grad():
                pred = self.forward(X)
                train_error = self.loss(pred, y)

        train_error /= training_size

        return train_error   

    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, tnn.Linear):
            tnn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def validation_accuracy(self, X_valid, y_valid, device=DEVICE):        
        
        valid_dataloader = DataLoader(list(zip(X_valid,y_valid)), shuffle=True)
        validation_size = len(valid_dataloader.dataset)

        # Validation Loop
        with t.no_grad():
            for X, y in self.valid_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                validation_error = self.loss(pred, y)

        validation_error /=validation_size
        return validation_error

    # Train uses backpropogation to solve for the model parameters
    def train(self, device=DEVICE):
        self.to(device)
        train_error_history = []
        for t in range(self.epochs):
            # Train loop
            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                def closure():
                    #Compute prediction and loss
                    pred = self.forward(X)
                    loss_t = self.loss(pred, y)

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_t.backward()
                    return loss_t
                
                self.optimizer.step(closure)

            train_error = self.training_accuracy()
            if t%self.report_steps==0:
                print(f"----Progress Update----")
                print(f"Epoch: {t} \n")
                print(f"Training Error: {train_error} \n")
            train_error_history.append(train_error)

        return train_error_history       
    
    
    # Plots errors output from training to show path to parameters
    def plot_training_error(self, name):
        epoch_number = np.arange(0,self.epochs,1)
        plt.figure(name)
        plt.title(name)
        plt.plot(epoch_number, self.training_error, color = "blue", label = "Training", linestyle = "-")        
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")

    # def visual_graph(self):
    #     __, input_size = self.x.shape
    #     model_graph = draw_graph(self(), input_size=(2,input_size))

# Multifidelity Neural Network
# Setup based on Meng and Karniadakis 2020 paper
# Uses 3 neural networks combined
# A low fidelity NN and a linear and non-linear HF NN
# All nn dimensions are given as the size of the hidden layers.
# Input and output size are given by the x and y data
class MultiFidelityNN(tnn.Module,mf_model.MFModel):
    def __init__(self, x_lf, y_lf, x_hf, y_hf, lf_nn_dim, hf_linear_nn_dim, hf_nl_nn_dim, load_lf_from_file = False, lr_lf = 0.005, wd_lf = 0.000001, lr_mf = 0.001, wd_mf = 0.001):
        # Populate the multi-fidelity model data
        tnn.Module.__init__(self)
        mf_model.MFModel.__init__(self,x_lf, y_lf, x_hf, y_hf)

        # Generate a NN model of the lf model data
        self.low_fidelity_nn = MLPNetwork(x_lf, y_lf,hidden_layer_size=lf_nn_dim,activation=tnn.Tanh(), train=True)

        # Create lf approximations of the high fidelity data using the lf model
        yhat_lf = self.low_fidelity_nn.evaluate(x_hf)
        x_hf_prime = t.cat((yhat_lf,x_hf),dim=1)
        
        # Train composite NN
        self.high_fidelity_nn = CompositeNN(x_hf_prime, y_hf, layers_nl=hf_nl_nn_dim, layers_linear=hf_linear_nn_dim, activation_nl=tnn.Tanh(), train=True)

    def evaluate(self,x):
        return self.forward(x)
        
    def forward(self,x):
        yhat_lf = self.low_fidelity_nn.evaluate(x)
        x_hf = t.cat((yhat_lf,x),dim=1)
        yhat_hf = self.high_fidelity_nn.evaluate(x_hf)
        return yhat_hf

    # Private function that outputs both mf and lf approx of x
    def _forward_loud(self,x):        
        yhat_lf = self.low_fidelity_nn.evaluate(x)
        x_hf = t.cat((yhat_lf,x),dim=1)
        yhat_hf = self.high_fidelity_nn.evaluate(x_hf)
        return yhat_hf, yhat_lf

    # def loss(self, yl_hat, yl, y_hat, y):
    #     t.nn.MSELoss(yl_hat, yl)
    #     return 0

    # Train uses backpropogation to solve for the model parameters
    # def train(self, device=DEVICE):
    #     self.to(device)
    #     train_error_history = []
    #     for t in range(self.epochs):
    #         # Train loop
    #         for X, y in self.training_dataloader:
    #             X = X.to(device)
    #             y = y.to(device)

    #             #Compute prediction and loss
    #             pred = self.forward(X)
    #             loss_t = self.loss(pred, y)

    #             # Backpropagation
    #             self.optimizer.zero_grad()
    #             loss_t.backward()
                
    #             self.optimizer.step()

    #         train_error = self.training_accuracy()
    #         if t%self.report_steps==0:
    #             print(f"----Progress Update----")
    #             print(f"Epoch: {t} \n")
    #             print(f"Training Error: {train_error} \n")
    #         train_error_history.append(train_error)

    #     return train_error_history

    def generate_plots(self):
        self.plot_data()

    
# Function used for testing LowFidelity NN capabilities
# further showcase of class can be found in MFSNN.ipynb
def TestMFFXN():
    fxn_ind = 6
    hf_fxn = mfs_functions.expensive_fxn(fxn_ind)
    lf_fxn = mfs_functions.cheap_fxn(fxn_ind)
    
    hf_fxn.plot_data(name="Discontinuous MF Fxn",color = "blue", label="HF Fxn", linestyle="dashed")
    lf_fxn.plot_data(name="Discontinuous MF Fxn",color = "black", label="LF Fxn", linestyle="dashed")
    plt.show()

def TestFunctionMLP():
    fxn_ind = 1
    hf_fxn = mfs_functions.expensive_fxn(1)
    lf_fxn = mfs_functions.cheap_fxn(1)

    X_lf = np.zeros([30,1])
    X_lf[:,0] = np.linspace(0.0,1.0,30,endpoint=True)
    y_lf = lf_fxn.evaluate(X_lf)
    # X_lf = t.tensor(X_lf, dtype=t.float32)
    # y_lf = t.tensor(y_lf, dtype=t.float32)
    X_lf = t.tensor(X_lf)
    y_lf = t.tensor(y_lf)

    X_check = np.zeros([300,1])
    X_check[:,0] = np.linspace(0,1.0,300,endpoint=True)
    y_check = lf_fxn.evaluate(X_check)
    # X_check = t.tensor(X_check, dtype=t.float32)
    # y_check = t.tensor(y_check, dtype=t.float32)
    X_check = t.tensor(X_check)
    y_check = t.tensor(y_check)
    loss = t.nn.MSELoss()
    lf_model = MLPNetwork(X_lf,y_lf,hidden_layer_size=[10,10,10],activation=tnn.Tanh(),learning_rate=0.001, weight_decay=0,loss=loss)
    #error_history = lf_model.train()

    # Plot the results
    lf_model.plot_data("LF Model", label = "Data")
    lf_model.plot_model(X_check,"LF Model", color = "blue", label="Model")
    plt.show()

    #lf_model.visualize_shape()
    lf_model.plot_training_error("Training Error")
    plt.show()

def TestFunctionMFSNN():
    fxn_ind = 1
    hf_fxn = mfs_functions.expensive_fxn(fxn_ind)
    lf_fxn = mfs_functions.cheap_fxn(fxn_ind)

    lf_samp = 12
    X_lf = np.zeros([lf_samp,1])
    X_lf[:,0] = np.linspace(0.0,1.0,lf_samp,endpoint=True)
    y_lf = lf_fxn.evaluate(X_lf)
    X_lf = t.tensor(X_lf, dtype=t.float32)
    y_lf = t.tensor(y_lf, dtype=t.float32)
    
    hf_samp = 4
    X_hf = np.zeros([hf_samp,1])
    X_hf[:,0] = np.linspace(0.0,1.0,hf_samp,endpoint=True)
    y_hf = hf_fxn.evaluate(X_hf)
    X_hf = t.tensor(X_hf, dtype=t.float32)
    y_hf = t.tensor(y_hf, dtype=t.float32)

    mfnn_model = MultiFidelityNN(X_lf, y_lf, X_hf, y_hf, lf_nn_dim=[20,20], hf_linear_nn_dim=[10], hf_nl_nn_dim= [10,10],lr_lf=0.0001,wd_lf=0,lr_mf=0.001, wd_mf=0)

    #mfnn_model.high_fidelity_nn.visual_graph()

    # Plot the results
    mfnn_model.plot_data("MF Model")
    X_check = np.zeros([300,1])
    X_check[:,0] = np.linspace(0,1.0,300,endpoint=True)
    X_check = t.tensor(X_check, dtype=t.float32)
    mfnn_model.plot_model(X_check,"MF Model", color = "blue", label="MFNN")
    mfnn_model.low_fidelity_nn.plot_model(X_check, name = "MF Model", label="LFNN")
    hf_fxn.plot_data(name="MF Model",color = "blue", label="HF Fxn", linestyle="dashed")
    lf_fxn.plot_data(name="MF Model",color = "black", label="LF Fxn", linestyle="dashed")
    plt.show()

    # Plot the model error
    mfnn_model.low_fidelity_nn.plot_training_error("Low Fidelity Training Error")
    plt.show()
    mfnn_model.high_fidelity_nn.plot_training_error("Multi-Fidelity Training Error")
    plt.show()

def Test2DFunction():
    fxn_ind = 5
    hf_fxn = mfs_functions.expensive_fxn(5)
    lf_fxn = mfs_functions.cheap_fxn(5)
    dim, xmin, xmax = mfs_functions.function_bounds(5)
    x_sample = np.linspace(xmin[0],xmax[0],25)
    y_sample = np.linspace(xmin[1],xmax[1],25)
    x,y = np.meshgrid(x_sample,y_sample)
    X_in = np.vstack([x.ravel(),y.ravel()]).T
    Y_in = hf_fxn.evaluate(X_in)
    print(Y_in)
    #ax = plt.figure().add_subplot(projection='3d')
    #ax.scatter(X_in[:,0],X_in[:,1],Y_in[:,0],marker='o')
    #plt.show()
    X_in = t.tensor(X_in,dtype=t.float32)
    Y_in = t.tensor(Y_in,dtype=t.float32)
    nn_2d = MLPNetwork(X_in,Y_in)
    Y_star = nn_2d.evaluate(X_in)

#TestFunctionMFSNN()