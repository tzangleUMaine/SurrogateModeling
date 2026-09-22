from __future__ import annotations

import torch
import matplotlib.pyplot as plt
import torch.optim.optimizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

## Doing it the Pytorch way without using our custom feature extraction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import typing

import mfs_functions
import mf_model
import model
import data_handler
import csv

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]


class MLPNetwork(torch.nn.Module, model.Model):
    """Fully connected neural network class"""

    def __init__(
        self,
        X,
        y,
        X_test: torch.Tensor = torch.Tensor(),
        y_test: torch.Tensor = torch.Tensor(),
        hidden_layer_size: list[int] = [10, 8],
        activation: type[torch.nn.Module] = torch.nn.Tanh(),
        loss: torch.optim.Optimizer = torch.nn.MSELoss(),
        epochs: int = 500,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=10000,
        weight_file="network_weights",
        load_weights=False,
        write_weights=False,
        train=True,
        scale_input=False,
        bounds=None,
    ):
        torch.nn.Module.__init__(self)
        model.Model.__init__(self, X, y, bounds)

        self.device = torch.device(DEVICE)
        self.scale_input = scale_input

        # Scale the input matrix
        if self.scale_input:
            self.x = self.x_scale.transform(self.x)
            # X_test = self.x_scale.transform(X_test)

        # Build tensors directly on the target device so batches drawn from the
        # DataLoader are already in place -- no per-batch host-to-device copy.
        X_tensor = torch.as_tensor(self.x, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(self.y, dtype=torch.float32, device=self.device)
        X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_tensor = torch.as_tensor(y_test, dtype=torch.float32, device=self.device)

        # Hyper parameters
        self.epochs = epochs

        # Set loss function
        self.loss = loss

        # Set batch size
        self.batch_size = batch_size

        # Normalize the y data
        self.datahandler = data_handler.DataHandler(y_tensor)
        # y = self.datahandler.scale(y)

        # Store the model data
        self.training_dataloader = DataLoader(
            TensorDataset(X_tensor, y_tensor), shuffle=True, batch_size=self.batch_size
        )

        # Store the test data
        self.test_dataloader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            shuffle=False,
            batch_size=self.batch_size,
        )

        training_size = len(self.test_dataloader.dataset)

        # Generate the neural network
        _, input_size = X_tensor.shape
        _, output_size = y_tensor.shape

        # Setup network shape
        nn_layers = []
        nn_layers.append(torch.nn.Linear(input_size, hidden_layer_size[0]))
        if activation != "none":
            nn_layers.append(activation)
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:-1]:
            nn_layers.append(torch.nn.Linear(prev_size, layer_size))
            if activation != "none":
                nn_layers.append(activation)
            prev_size = layer_size
        nn_layers.append(torch.nn.Linear(prev_size, output_size))

        self._layers = torch.nn.Sequential(*nn_layers)

        # Move the whole module to its target device once, up front, so every
        # later call (train, evaluate, load_state_dict) sees consistent placement
        self.to(self.device)

        # Apply initial weights
        # self.apply(self.init_weights)
        # Set the optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Try loading weights from file, if not train the network!
        if load_weights:
            self.load_state_dict(torch.load(weight_file))
        else:
            self.training_error, self.testing_error = self.train()

        if write_weights:
            torch.save(self.state_dict(), weight_file)

    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Define a forward method here
    def forward(self, x):
        x = self._layers(x)
        return x

    # Evaluate function defines method for use in generic model class
    # This is the only boundary method: it accepts numpy (or tensor) input,
    # handles the device transfer itself, and always returns a plain numpy
    # array so it composes with the numpy-based helpers in model.Model.
    def evaluate(self, x_in):
        if self.scale_input:
            x_in = self.x_scale.transform(x_in)

        x_tensor = torch.as_tensor(x_in, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        return y_hat.detach().cpu().numpy()

    # Calculates the training error using the given loss function
    # Internal accuracy/training code stays entirely in tensor-space on
    # self.device and calls forward() directly -- never evaluate(), which is
    # the public numpy-in/numpy-out boundary.
    def training_accuracy(self):
        # Validation loop
        training_size = len(self.training_dataloader)
        running_error = 0.0

        with torch.no_grad():
            for X, y in self.training_dataloader:
                pred = self.forward(X)
                running_error += self.loss(pred, y).item()

        train_error = running_error / training_size

        return np.sqrt(train_error)

    def testing_accuracy(self):
        # Validation loop
        testing_size = len(self.test_dataloader)
        running_error = 0.0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.forward(X)
                running_error += self.loss(pred, y).item()

        test_error = running_error / testing_size

        return np.sqrt(test_error)

    # Train uses backpropogation to solve for the model parameters
    def train(self, eval_interval=10):
        train_error_history = []
        test_error_history = []
        train_error = test_error = None
        for t in range(self.epochs):
            # Train loop
            for X, y in self.training_dataloader:
                self.optimizer.zero_grad()
                pred = self.forward(X)
                loss_t = self.loss(pred, y)
                loss_t.backward()
                self.optimizer.step()
                if t % 100 == 0:
                    print(loss_t)

            # Computing train/test accuracy requires a full extra forward pass
            # over each dataset, so only do it periodically rather than every
            # epoch, and forward-fill the skipped epochs for plotting.
            if t % eval_interval == 0 or t == self.epochs - 1:
                train_error = self.training_accuracy()
                test_error = self.testing_accuracy()
            train_error_history.append(train_error)
            test_error_history.append(test_error)

        return train_error_history, test_error_history

    # Plots errors output from training to show path to parameters
    def plot_training_error(self, name):
        epoch_number = np.arange(0, self.epochs, 1)
        plt.figure(name)
        plt.title(name)
        plt.plot(
            epoch_number,
            self.training_error,
            color="blue",
            label="Train",
            linestyle="-",
        )
        plt.plot(
            epoch_number,
            self.testing_error,
            color="black",
            label="Test",
            linestyle="-",
        )
        plt.legend()
        # plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Error")


# Composite Neural Network
# Explicitly seperates the network into a linear and non-linear network.
# Fully connected NN architecture in both cases is concatenated at the final layer
class CompositeNN(torch.nn.Module, mf_model.MFModel):
    def __init__(
        self,
        X,
        y,
        X_test=torch.Tensor(),
        y_test=torch.Tensor(),
        layers_nl=[10, 8],
        activation_nl=torch.nn.Tanh(),
        loss=torch.nn.MSELoss(),
        epochs=1000,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=100,
        train=True,
    ):
        torch.nn.Module.__init__(self)
        model.Model.__init__(self, X, y)

        self.device = torch.device(DEVICE)

        # Convert numpy arrays into tensors, built directly on the target
        # device so batches drawn from the DataLoader need no per-batch copy.
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_tensor = torch.as_tensor(y_test, dtype=torch.float32, device=self.device)

        # Hyper parameters
        self.epochs = epochs

        # Report rate
        self.report_steps = 1

        # Set loss function
        self.loss = loss

        # Set batch size
        self.batch_size = batch_size

        # Normalize the y data
        # self.datahandler = data_handler.DataHandler(y)
        # y = self.datahandler.scale(y)

        # Store the model data
        self.training_dataloader = DataLoader(
            TensorDataset(X_tensor, y_tensor), shuffle=True, batch_size=self.batch_size
        )

        # Store the test data
        self.test_dataloader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            shuffle=False,
            batch_size=self.batch_size,
        )

        # Generate the neural network
        __, input_size = X_tensor.shape
        __, output_size = y_tensor.shape

        # Setup non-linear network shape
        self._nl_layers = []
        self._nl_layers.append(torch.nn.Linear(input_size, layers_nl[0]))
        self._nl_layers.append(activation_nl)
        prev_size = layers_nl[0]

        for layer_size in layers_nl[1:-1]:
            self._nl_layers.append(torch.nn.Linear(prev_size, layer_size))
            self._nl_layers.append(activation_nl)
            prev_size = layer_size
        self._nl_layers.append(torch.nn.Linear(prev_size, output_size))
        self._nl_layers = torch.nn.Sequential(*self._nl_layers)

        # Generate a linear neural network to create a shortcut for model training
        self._linear_layers = torch.nn.Linear(
            in_features=input_size, out_features=output_size, bias=True
        )

        # Apply initial weights
        self.apply(self.init_weights)

        # Move the whole module to its target device once, up front
        self.to(self.device)

        # Check whether the network is just being setup or if traing is specified
        if train:
            # Set the optimizer and parameters
            # self.optimizer = t.optim.LBFGS(self.parameters(), lr = learning_rate, max_iter=40)
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Run training on network
            self.training_error, self.testing_error = self.train()

    # Simple function to create csv files of dataloader to check output distributions
    def dataloader_write(self, file_name):
        with open(file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",", quotechar="|")
            for X, y in self.training_dataloader:
                with torch.no_grad():
                    print(X[:][1])
                    csv_writer.writerow(X[:][1])

    def forward(self, x):
        y_linear = self._linear_layers(x)
        y_nl = self._nl_layers(x)
        return y_linear + y_nl

    # Evaluate function defines method for use in generic model class
    # Public boundary method: numpy (or tensor) in, numpy out -- matches the
    # evaluate() contract used by model.Model and every other model class.
    def evaluate(self, x_in):
        x_tensor = torch.as_tensor(x_in, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        return y_hat.detach().cpu().numpy()

    # Calculates the training error using the given loss function
    def training_accuracy(self):
        # Validation loop
        training_size = len(self.training_dataloader.dataset)
        train_error = 0.0

        with torch.no_grad():
            for X, y in self.training_dataloader:
                pred = self.forward(X)
                train_error += self.loss(pred, y).item()

        train_error /= training_size

        return np.sqrt(train_error)

    def testing_accuracy(self):
        # Validation loop
        testing_size = len(self.test_dataloader.dataset)
        running_loss = 0.0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.forward(X)
                running_loss += self.loss(pred, y).item()

        test_error = running_loss / testing_size
        return np.sqrt(test_error)

    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Train uses backpropogation to solve for the model parameters
    def train(self, eval_interval=10):
        train_error_history = []
        test_error_history = []
        train_error = test_error = None
        for t in range(self.epochs):
            # Train loop
            for X, y in self.training_dataloader:

                def closure():
                    # Compute prediction and loss
                    pred = self.forward(X)
                    loss_t = self.loss(pred, y)

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_t.backward()
                    return loss_t

                self.optimizer.step(closure)

            # Computing train/test accuracy requires a full extra forward pass
            # over each dataset, so only do it periodically rather than every
            # epoch, and forward-fill the skipped epochs for plotting.
            if t % eval_interval == 0 or t == self.epochs - 1:
                train_error = self.training_accuracy()
                test_error = self.testing_accuracy()
                if t % self.report_steps == 0:
                    print(f"----Progress Update----")
                    print(
                        f"Epoch: {t}   || Train Error: {train_error} || Test Error: {test_error}\n"
                    )
            train_error_history.append(train_error)
            test_error_history.append(test_error)

        return train_error_history, test_error_history

    # def loss_function(self,weight_decay = 0.001):
    #     return

    # Plots errors output from training to show path to parameters
    def plot_training_error(self, name):
        epoch_number = np.arange(0, self.epochs, 1)
        plt.figure(name)
        plt.title(name)
        plt.plot(
            epoch_number,
            self.training_error,
            color="blue",
            label="Train",
            linestyle="-",
        )
        plt.plot(
            epoch_number,
            self.testing_error,
            color="black",
            label="Test",
            linestyle="-",
        )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Error")

        # Plots errors output from training to show path to parameters

    # def visual_graph(self):
    #     __, input_size = self.x.shape
    #     model_graph = draw_graph(self(), input_size=(2,input_size))


# Multifidelity Neural Network
# Setup based on Meng and Karniadakis 2020 paper
# Uses 3 neural networks combined
# A low fidelity NN and a linear and non-linear HF NN
# All nn dimensions are given as the size of the hidden layers.
# Input and output size are given by the x and y data
class MultiFidelityNN(torch.nn.Module, mf_model.MFModel):
    def __init__(
        self,
        X_lf,
        y_lf,
        X_lf_test,
        y_lf_test,
        X_hf,
        y_hf,
        X_hf_test,
        y_hf_test,
        lf_nn_dim,
        hf_nl_nn_dim,
        epochs_lf=500,
        epochs_mf=1000,
        lr_lf=0.005,
        wd_lf=0.000001,
        lr_mf=0.001,
        wd_mf=0.001,
    ):
        # Populate the multi-fidelity model data
        torch.nn.Module.__init__(self)
        mf_model.MFModel.__init__(self, X_lf, y_lf, X_hf, y_hf)

        # Generate a NN model of the lf model data
        self.low_fidelity_nn = MLPNetwork(
            X_lf,
            y_lf,
            X_test=X_lf_test,
            y_test=y_lf_test,
            hidden_layer_size=lf_nn_dim,
            activation=torch.nn.Tanh(),
            epochs=epochs_lf,
            learning_rate=lr_lf,
            train=True,
        )

        # Evaluate the LF KAN model at the HF locations
        y_lf_at_hf = self.low_fidelity_nn.evaluate(self.x)
        y_lf_at_hf_test = self.low_fidelity_nn.evaluate(X_hf_test)

        # Generate the MF KAN model by combining the evaluated data set with the HF data
        x_mf = np.hstack((self.x, y_lf_at_hf))
        x_mf_test = np.hstack((X_hf_test, y_lf_at_hf_test))

        # Train composite NN
        self.high_fidelity_nn = CompositeNN(
            x_mf,
            y_hf,
            X_test=x_mf_test,
            y_test=y_hf_test,
            layers_nl=hf_nl_nn_dim,
            epochs=epochs_mf,
            learning_rate=lr_mf,
            weight_decay=wd_mf,
            activation_nl=torch.nn.Tanh(),
            train=True,
        )

    def evaluate(self, x):
        return self.forward(x)

    def forward(self, x):
        # Evaluate the LF KAN model at the HF locations
        y_lf_at_x = self.low_fidelity_nn.evaluate(x).reshape(-1, 1)

        # Generate the MF input by combining x and y_lf_at_x
        x_mf = np.hstack((x, y_lf_at_x))

        # Evaluate the MF KAN model
        return self.high_fidelity_nn.evaluate(x_mf)


# Function used for testing LowFidelity NN capabilities
# further showcase of class can be found in MFSNN.ipynb
def TestMFFXN():
    fxn_ind = 6
    hf_fxn = mfs_functions.expensive_fxn(fxn_ind)
    lf_fxn = mfs_functions.cheap_fxn(fxn_ind)

    hf_fxn.plot_data(
        name="Discontinuous MF Fxn", color="blue", label="HF Fxn", linestyle="dashed"
    )
    lf_fxn.plot_data(
        name="Discontinuous MF Fxn", color="black", label="LF Fxn", linestyle="dashed"
    )
    plt.show()


def TestFunctionMLP():
    fxn_ind = 1
    hf_fxn = mfs_functions.expensive_fxn(1)
    lf_fxn = mfs_functions.cheap_fxn(1)

    X_lf = np.zeros([30, 1])
    X_lf[:, 0] = np.linspace(0.0, 1.0, 30, endpoint=True)
    y_lf = lf_fxn.evaluate(X_lf)
    # X_lf = t.tensor(X_lf, dtype=t.float32)
    # y_lf = t.tensor(y_lf, dtype=t.float32)
    X_lf = torch.tensor(X_lf)
    y_lf = torch.tensor(y_lf)

    X_check = np.zeros([300, 1])
    X_check[:, 0] = np.linspace(0, 1.0, 300, endpoint=True)
    y_check = lf_fxn.evaluate(X_check)
    # X_check = t.tensor(X_check, dtype=t.float32)
    # y_check = t.tensor(y_check, dtype=t.float32)
    X_check = torch.tensor(X_check)
    y_check = torch.tensor(y_check)
    loss = torch.nn.MSELoss()
    lf_model = MLPNetwork(
        X_lf,
        y_lf,
        hidden_layer_size=[10, 10, 10],
        activation=torch.nn.Tanh(),
        learning_rate=0.001,
        weight_decay=0,
        loss=loss,
    )
    # error_history = lf_model.train()

    # Plot the results
    lf_model.plot_data("LF Model", label="Data")
    lf_model.plot_model(X_check, "LF Model", color="blue", label="Model")
    plt.show()

    # lf_model.visualize_shape()
    lf_model.plot_training_error("Training Error")
    plt.show()


def TestFunctionMFSNN():
    fxn_ind = 1
    hf_fxn = mfs_functions.expensive_fxn(fxn_ind)
    lf_fxn = mfs_functions.cheap_fxn(fxn_ind)

    lf_samp = 12
    X_lf = np.zeros([lf_samp, 1])
    X_lf[:, 0] = np.linspace(0.0, 1.0, lf_samp, endpoint=True)
    y_lf = lf_fxn.evaluate(X_lf)
    X_lf = torch.tensor(X_lf, dtype=torch.float32)
    y_lf = torch.tensor(y_lf, dtype=torch.float32)

    hf_samp = 4
    X_hf = np.zeros([hf_samp, 1])
    X_hf[:, 0] = np.linspace(0.0, 1.0, hf_samp, endpoint=True)
    y_hf = hf_fxn.evaluate(X_hf)
    X_hf = torch.tensor(X_hf, dtype=torch.float32)
    y_hf = torch.tensor(y_hf, dtype=torch.float32)

    mfnn_model = MultiFidelityNN(
        X_lf,
        y_lf,
        X_hf,
        y_hf,
        lf_nn_dim=[20, 20],
        hf_linear_nn_dim=[10],
        hf_nl_nn_dim=[10, 10],
        lr_lf=0.0001,
        wd_lf=0,
        lr_mf=0.001,
        wd_mf=0,
    )

    # mfnn_model.high_fidelity_nn.visual_graph()

    # Plot the results
    mfnn_model.plot_data("MF Model")
    X_check = np.zeros([300, 1])
    X_check[:, 0] = np.linspace(0, 1.0, 300, endpoint=True)
    X_check = torch.tensor(X_check, dtype=t.float32)
    mfnn_model.plot_model(X_check, "MF Model", color="blue", label="MFNN")
    mfnn_model.low_fidelity_nn.plot_model(X_check, name="MF Model", label="LFNN")
    hf_fxn.plot_data(name="MF Model", color="blue", label="HF Fxn", linestyle="dashed")
    lf_fxn.plot_data(name="MF Model", color="black", label="LF Fxn", linestyle="dashed")
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
    x_sample = np.linspace(xmin[0], xmax[0], 25)
    y_sample = np.linspace(xmin[1], xmax[1], 25)
    x, y = np.meshgrid(x_sample, y_sample)
    X_in = np.vstack([x.ravel(), y.ravel()]).T
    Y_in = hf_fxn.evaluate(X_in)
    print(Y_in)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(X_in[:,0],X_in[:,1],Y_in[:,0],marker='o')
    # plt.show()
    X_in = torch.tensor(X_in, dtype=torch.float32)
    Y_in = torch.tensor(Y_in, dtype=torch.float32)
    nn_2d = MLPNetwork(X_in, Y_in)
    Y_star = nn_2d.evaluate(X_in)


# TestFunctionMFSNN()
