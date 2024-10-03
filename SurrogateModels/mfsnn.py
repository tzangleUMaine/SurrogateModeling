import torch as t
import torch.nn as tnn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

t.manual_seed(17)

## Doing it the Pytorch way without using our custom feature extraction
DEVICE='cpu'
DTYPE=t.float32
import numpy as np

import mfs_functions
import mf_model
import model


# Class for a linear network of variable shape for regression purposes
class MLPNetwork(t.nn.Module, model.Model):
    def __init__(self, X, y, hidden_layer_size = [10,8], activation = tnn.Tanh(), loss = t.nn.MSELoss()):
        t.nn.Module.__init__(self)
        model.Model.__init__(self,X,y)

        # Hyper parameters
        self.epochs = 1000        

        # Report rate
        self.report_steps = 50

        # Set loss function
        self.loss = loss

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=1)

        # Generate the neural network
        __, input_size = X.shape
        __, output_size = y.shape
        #self.mlp_network = MLPNetwork(input_size, output_size, hidden_layer_size, activation)

        # Setup network shape
        self._layers = t.nn.ModuleList([
            tnn.Linear(input_size, hidden_layer_size[0]),
            activation])
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            self._layers.append(
                tnn.Linear(prev_size,layer_size)
            )
            if activation != "none":
                self._layers.append(activation)
            prev_size = layer_size
        self._layers.append(tnn.Linear(prev_size,output_size))
      
        # Apply initial weights
        self.apply(self.init_weights)

        # Set the optimizer
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.0001)
        #self.optimizer = t.optim.LBFGS(self.parameters(), lr = 1 )

        self.training_error = self.train()        
   
    # Calculates the training error using the given loss function
    def training_accuracy(self, device=DEVICE):
        # Validation loop
        training_size = len(self.training_dataloader.dataset)
        train_error = 0

        with t.no_grad():
            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                train_error = self.loss(pred, y)

        train_error /= training_size

        return train_error   



    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, tnn.Linear):
            tnn.init.xavier_uniform(m.weight)
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

                #Compute prediction and loss
                pred = self.__forward(X)
                loss_t = self.loss(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss_t.backward()
                # def closure():
                #     self.optimizer.zero_grad()
                #     pred = self.__forward(X)
                #     loss_t = self.loss(pred,y)
                #     # for p in self.parameters():
                #     #     loss_t += 0.01*(p**2).sum()
                #     loss_t.backward()
                #     return loss_t
                
                self.optimizer.step()

            train_error = self.training_accuracy()
            if t%self.report_steps==0:
                print(f"----Progress Update----")
                print(f"Epoch: {t} \n")
                print(f"Training Error: {train_error} \n")
            train_error_history.append(train_error)

        return train_error_history

    # # Define a training iteration function to handle memory stuff for LBFGS
    # def _training_iteration(self,device=DEVICE):
    #     running_loss = 0.0
    #     for i, (X,y) in enumerate(self.training_dataloader):
    #         X = X.to(device)
    #         y = y.to(device)

    #         X_ = t.autograd.Variable(X, requires_grad=True)
    #         y_ = t.autograd.Variable(y)
            
    #         def closure():
    #             if t.is_grad_enabled():
    #                 self.optimizer.zero_grad()
    #             output = self.evaluate(X_)
    #             loss_t = self.loss(output,y_)
    #             if loss_t.requires_grad():
    #                 loss_t.backward()
    #             return loss_t
    #         self.optimizer.step(closure)

    #         # calculate the loss again for monitoring
    #         output = self(X_)
    #         loss = closure()
    #         running_loss += loss.item()
    #         return running_loss
        
    # Evaluate function defines method for use in generic model class
    def evaluate(self,x_in):
        with t.no_grad():
            return self.__forward(x_in)
    
    # Plots errors output from training to show path to parameters
    def plot_error(self):
        epoch_number = np.arange(0,self.epochs,1)
        plt.figure("Model Error")
        plt.title("Model Error")
        plt.plot(epoch_number, self.training_error, color = "blue", label = "Training", linestyle = "-")        
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()


    def __forward(self, x):
        for l in self._layers:
            xnext = l(x) # call the layers in sequence
            x = xnext
        return x

# Composite Neural Network 
# Explicitly seperates the network into a linear and non-linear network.
# Fully connected NN architecture in both cases is concatenated at the final layer
class CompositeNN(t.nn.Module, mf_model.MFModel):
    def __init__(self, X, y, hidden_layer_size = [10,8], activation = tnn.Tanh(), loss = t.nn.MSELoss()):
        t.nn.Module.__init__(self)
        model.Model.__init__(self,X,y)

        # Hyper parameters
        self.epochs = 1000        

        # Report rate
        self.report_steps = 50

        # Set loss function
        self.loss = loss

        # Store the model data
        self.training_dataloader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=1)

        # Generate the neural network
        __, input_size = X.shape
        __, output_size = y.shape
        #self.mlp_network = MLPNetwork(input_size, output_size, hidden_layer_size, activation)

        # Setup network shape
        self._layers = t.nn.ModuleList([
            tnn.Linear(input_size, hidden_layer_size[0]),
            activation])
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            self._layers.append(
                tnn.Linear(prev_size,layer_size)
            )
            if activation != "none":
                self._layers.append(activation)
            prev_size = layer_size
        self._layers.append(tnn.Linear(prev_size,output_size))
      
        # Apply initial weights
        self.apply(self.init_weights)

        # Set the optimizer
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.0001)
        #self.optimizer = t.optim.LBFGS(self.parameters(), lr = 1 )

        self.training_error = self.train()        
   
    # Calculates the training error using the given loss function
    def training_accuracy(self, device=DEVICE):
        # Validation loop
        training_size = len(self.training_dataloader.dataset)
        train_error = 0

        with t.no_grad():
            for X, y in self.training_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.evaluate(X)
                train_error = self.loss(pred, y)

        train_error /= training_size

        return train_error   



    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, tnn.Linear):
            tnn.init.xavier_uniform(m.weight)
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

                #Compute prediction and loss
                pred = self.__forward(X)
                loss_t = self.loss(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss_t.backward()
                # def closure():
                #     self.optimizer.zero_grad()
                #     pred = self.__forward(X)
                #     loss_t = self.loss(pred,y)
                #     # for p in self.parameters():
                #     #     loss_t += 0.01*(p**2).sum()
                #     loss_t.backward()
                #     return loss_t
                
                self.optimizer.step()

            train_error = self.training_accuracy()
            if t%self.report_steps==0:
                print(f"----Progress Update----")
                print(f"Epoch: {t} \n")
                print(f"Training Error: {train_error} \n")
            train_error_history.append(train_error)

        return train_error_history

    # # Define a training iteration function to handle memory stuff for LBFGS
    # def _training_iteration(self,device=DEVICE):
    #     running_loss = 0.0
    #     for i, (X,y) in enumerate(self.training_dataloader):
    #         X = X.to(device)
    #         y = y.to(device)

    #         X_ = t.autograd.Variable(X, requires_grad=True)
    #         y_ = t.autograd.Variable(y)
            
    #         def closure():
    #             if t.is_grad_enabled():
    #                 self.optimizer.zero_grad()
    #             output = self.evaluate(X_)
    #             loss_t = self.loss(output,y_)
    #             if loss_t.requires_grad():
    #                 loss_t.backward()
    #             return loss_t
    #         self.optimizer.step(closure)

    #         # calculate the loss again for monitoring
    #         output = self(X_)
    #         loss = closure()
    #         running_loss += loss.item()
    #         return running_loss
        
    # Evaluate function defines method for use in generic model class
    def evaluate(self,x_in):
        with t.no_grad():
            return self.__forward(x_in)
    
    # Plots errors output from training to show path to parameters
    def plot_error(self):
        epoch_number = np.arange(0,self.epochs,1)
        plt.figure("Model Error")
        plt.title("Model Error")
        plt.plot(epoch_number, self.training_error, color = "blue", label = "Training", linestyle = "-")        
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()


    def __forward(self, x):
        for l in self._layers:
            xnext = l(x) # call the layers in sequence
            x = xnext
        return x

# Multifidelity Neural Network
# Setup based on Meng and Karniadakis 2020 paper
# Uses 3 neural networks combined
# A low fidelity NN and a linear and non-linear HF NN
# All nn dimensions are given as the size of the hidden layers.
# Input and output size are given by the x and y data
class MultiFidelityNN(mf_model.MFModel):
    def __init__(self, x_lf, y_lf, x_hf, y_hf, lf_nn_dim, hf_linear_nn_dim, hf_nl_nn_dim):
        # Populate the multi-fidelity model data
        super().__init__(x_lf, y_lf, x_hf, y_hf)

        # Generate a NN model of the lf model data
        print(x_lf)
        self.low_fidelity_nn = MLPNetwork(x_lf, y_lf,hidden_layer_size=lf_nn_dim,activation=tnn.Tanh())

        # Create lf approximations of the high fidelity data using the lf model
        yhat_lf = self.low_fidelity_nn.evaluate(x_hf)
        x_hf_prime = t.cat((yhat_lf,x_hf),dim=1)
        
        self.high_fidelity_nn = MLPNetwork(x_hf_prime,y_hf,hidden_layer_size=hf_nl_nn_dim,activation=tnn.Linear())


    def evaluate(self,x):
        yhat_lf = self.low_fidelity_nn.evaluate(x)
        x_hf = t.cat((yhat_lf,x),dim=1)
        yhat_hf = self.high_fidelity_nn.evaluate(x_hf)
        return yhat_hf
    
    def generate_plots(self):
        self.plot_data()


# Function used for testing LowFidelity NN capabilities
# further showcase of class can be found in MFSNN.ipynb
def TestFunctionMLP():
    fxn_ind = 1
    hf_fxn = mfs_functions.expensive_fxn(1)
    lf_fxn = mfs_functions.cheap_fxn(1)

    X_lf = np.zeros([30,1])
    X_lf[:,0] = np.linspace(0.0,1.0,30,endpoint=True)
    y_lf = lf_fxn.evaluate(X_lf)
    X_lf = t.tensor(X_lf, dtype=t.float32)
    y_lf = t.tensor(y_lf, dtype=t.float32)

    X_check = np.zeros([300,1])
    X_check[:,0] = np.linspace(0,1.0,300,endpoint=True)
    y_check = lf_fxn.evaluate(X_check)
    X_check = t.tensor(X_check, dtype=t.float32)
    y_check = t.tensor(y_check, dtype=t.float32)
    loss = t.nn.MSELoss()
    lf_model = MLPNetwork(X_lf,y_lf,hidden_layer_size=[20,20],loss=loss)
    #error_history = lf_model.train()

    # Plot the results
    lf_model.plot_data("LF Model", label = "Data")
    lf_model.plot_model(X_check,"LF Model", color = "blue", label="Model")
    plt.show()

    lf_model.plot_error()

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

    mfnn_model = MultiFidelityNN(X_lf, y_lf, X_hf, y_hf, lf_nn_dim=[10,10,10], hf_linear_nn_dim=[6,4], hf_nl_nn_dim= [10,10,10])

    # Plot the results
    mfnn_model.plot_data("MF Model")
    X_check = np.zeros([300,1])
    X_check[:,0] = np.linspace(0,1.0,300,endpoint=True)
    X_check = t.tensor(X_check, dtype=t.float32)
    mfnn_model.plot_model(X_check,"MF Model", color = "blue", label="MFNN")
    mfnn_model.low_fidelity_nn.plot_model(X_check, "MF Model")
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

TestFunctionMFSNN()