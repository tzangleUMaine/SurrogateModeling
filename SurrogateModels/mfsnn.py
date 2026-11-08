from __future__ import annotations

import torch
import matplotlib.pyplot as plt
import torch.optim.optimizer
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
        scale_input=False,
        scale_output=False,
        y_scale_method="standard",
        bounds=None,
    ):
        torch.nn.Module.__init__(self)
        model.Model.__init__(
            self, X, y, bounds, y_scale_method=y_scale_method, scale_output=scale_output
        )

        self.device = torch.device(DEVICE)
        self.scale_input = scale_input
        self.scale_output = scale_output

        # Scale the input matrix -- X_test must use the same fitted x_scale as
        # the training data, or the model sees out-of-distribution inputs
        # whenever it's evaluated against the test set during training.
        if self.scale_input:
            self.x = self.x_scale.transform(self.x)
            X_test = self.x_scale.transform(X_test)

        # y_scale is fit on the training y above (in Model.__init__); apply
        # that same fitted transform to y_test so train/test stay comparable.
        if self.scale_output:
            y_test = self.y_scale.transform(y_test)

        # Keep the train/test data as whole tensors on the target device and
        # batch by indexing them directly. A DataLoader over a TensorDataset
        # fetches and stacks rows one at a time in Python, which made every
        # epoch cost O(n) Python calls (~60 ms/epoch at n=3900 on CUDA).
        self.X_train = torch.as_tensor(self.x, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(self.y, dtype=torch.float32, device=self.device)
        self.X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        self.y_test = torch.as_tensor(y_test, dtype=torch.float32, device=self.device)

        # Hyper parameters
        self.epochs = epochs

        # Set loss function
        self.loss = loss

        # Set batch size
        self.batch_size = batch_size

        # Generate the neural network
        _, input_size = self.X_train.shape
        _, output_size = self.y_train.shape

        # Setup network shape
        nn_layers = []
        nn_layers.append(torch.nn.Linear(input_size, hidden_layer_size[0]))
        if activation != "none":
            nn_layers.append(activation)
        prev_size = hidden_layer_size[0]

        for layer_size in hidden_layer_size[1:]:
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
            self.training_error, self.testing_error = self.fit()

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
        y_hat = y_hat.detach().cpu().numpy()

        if self.scale_output:
            y_hat = self.y_scale.inverse(y_hat)
        return y_hat

    # Calculates the training error using the given loss function
    # Internal accuracy/training code stays entirely in tensor-space on
    # self.device and calls forward() directly -- never evaluate(), which is
    # the public numpy-in/numpy-out boundary.
    def training_accuracy(self):
        with torch.no_grad():
            train_error = self.loss(self.forward(self.X_train), self.y_train).item()

        return np.sqrt(train_error)

    def testing_accuracy(self):
        with torch.no_grad():
            test_error = self.loss(self.forward(self.X_test), self.y_test).item()

        return np.sqrt(test_error)

    # Train uses backpropogation to solve for the model parameters
    def fit(self, eval_interval=10):
        train_error_history = []
        test_error_history = []
        train_error = test_error = None
        n = self.X_train.shape[0]
        for t in range(self.epochs):
            # Train loop -- shuffle each epoch, then step through minibatches
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                self.optimizer.zero_grad()
                pred = self.forward(self.X_train[idx])
                loss_t = self.loss(pred, self.y_train[idx])
                loss_t.backward()
                self.optimizer.step()
                # if t % 100 == 0:
                #     print(loss_t)

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
        plt.yscale("log")
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
        batch_size=10000,
        train=True,
        scale_input=False,
        scale_output=False,
        x_scale_method="bounded",
        y_scale_method="standard",
        bounds=None,
        solve_linear=False,
    ):
        torch.nn.Module.__init__(self)
        # Model.__init__ fits x_scale/y_scale on the training data and scales
        # self.x/self.y in place when requested
        model.Model.__init__(
            self,
            X,
            y,
            bounds,
            x_scale_method=x_scale_method,
            y_scale_method=y_scale_method,
            scale_input=scale_input,
            scale_output=scale_output,
        )

        self.device = torch.device(DEVICE)
        self.scale_input = scale_input
        self.scale_output = scale_output

        # Apply the same fitted transforms to the test data so train/test
        # stay on the same scale
        if self.scale_input:
            X_test = self.x_scale.transform(X_test)
        if self.scale_output:
            y_test = self.y_scale.transform(y_test)

        # Keep the train/test data as whole tensors on the target device and
        # batch by indexing them directly -- see MLPNetwork for why a
        # DataLoader is avoided here.
        self.X_train = torch.as_tensor(self.x, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(self.y, dtype=torch.float32, device=self.device)
        self.X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        self.y_test = torch.as_tensor(y_test, dtype=torch.float32, device=self.device)

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

        # Generate the neural network
        __, input_size = self.X_train.shape
        __, output_size = self.y_train.shape

        # Setup non-linear network shape
        self._nl_layers = []
        self._nl_layers.append(torch.nn.Linear(input_size, layers_nl[0]))
        self._nl_layers.append(activation_nl)
        prev_size = layers_nl[0]

        for layer_size in layers_nl[1:]:
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

        if solve_linear:
            self.solve_linear_path()

        # Check whether the network is just being setup or if traing is specified
        if train:
            # Set the optimizer and parameters
            # self.optimizer = t.optim.LBFGS(self.parameters(), lr = learning_rate, max_iter=40)
            # Only the nonlinear path is decayed: the linear path carries the
            # correlation with the LF model and shouldn't be pulled toward zero
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self._nl_layers.parameters(),
                        "weight_decay": weight_decay,
                    },
                    {"params": self._linear_layers.parameters(), "weight_decay": 0.0},
                ],
                lr=learning_rate,
            )

            # Run training on network
            self.training_error, self.testing_error = self.fit()

    # Simple function to create csv files of the training inputs to check distributions
    def dataloader_write(self, file_name):
        with open(file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",", quotechar="|")
            for row in self.X_train.cpu().numpy():
                csv_writer.writerow(row)

    def forward(self, x):
        y_linear = self._linear_layers(x)
        y_nl = self._nl_layers(x)
        return y_linear + y_nl

    # Evaluate function defines method for use in generic model class
    # Public boundary method: numpy (or tensor) in, numpy out -- matches the
    # evaluate() contract used by model.Model and every other model class.
    def evaluate(self, x_in):
        if self.scale_input:
            x_in = self.x_scale.transform(x_in)

        x_tensor = torch.as_tensor(x_in, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        y_hat = y_hat.detach().cpu().numpy()

        if self.scale_output:
            y_hat = self.y_scale.inverse(y_hat)
        return y_hat

    # Calculates the training error using the given loss function
    def training_accuracy(self):
        with torch.no_grad():
            train_error = self.loss(self.forward(self.X_train), self.y_train).item()

        return np.sqrt(train_error)

    def testing_accuracy(self):
        with torch.no_grad():
            test_error = self.loss(self.forward(self.X_test), self.y_test).item()

        return np.sqrt(test_error)

    # Sets the linear path to the least-squares fit of the training data, so the
    # nonlinear path starts from the remainder. The path stays trainable.
    def solve_linear_path(self):
        x = self.X_train.cpu().numpy()
        y = self.y_train.cpu().numpy()
        design = np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))
        solution = np.linalg.lstsq(design, y, rcond=None)[0]

        with torch.no_grad():
            self._linear_layers.weight.copy_(torch.as_tensor(solution[:-1].T))
            self._linear_layers.bias.copy_(torch.as_tensor(solution[-1]))

    # Measures how the test predictions split between the two paths. The shares
    # are each path's covariance with the prediction over the prediction's
    # variance, so they sum to 1, though either can fall outside [0, 1] when the
    # paths oppose each other. linear_rmse holds the nonlinear path at its mean.
    def path_usage(self):
        with torch.no_grad():
            y_linear = self._linear_layers(self.X_test)
            y_nl = self._nl_layers(self.X_test)
            y_hat = y_linear + y_nl
            centered = y_hat - y_hat.mean()
            variance = (centered**2).mean()
            linear_share = ((y_linear - y_linear.mean()) * centered).mean() / variance
            nl_share = ((y_nl - y_nl.mean()) * centered).mean() / variance
            linear_error = self.loss(y_linear + y_nl.mean(), self.y_test)

        return {
            "linear_share": linear_share.item(),
            "nl_share": nl_share.item(),
            "linear_rmse": np.sqrt(linear_error.item()),
            "linear_weights": self._linear_layers.weight.detach().cpu().numpy().ravel().copy(),
        }

    # Helper function to initialize weights
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Train uses backpropogation to solve for the model parameters
    def fit(self, eval_interval=10):
        train_error_history = []
        test_error_history = []
        self.path_history = []
        train_error = test_error = None
        n = self.X_train.shape[0]
        for t in range(self.epochs):
            # Train loop -- shuffle each epoch, then step through minibatches
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                X, y = self.X_train[idx], self.y_train[idx]

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
                self.path_history.append({"epoch": t, **self.path_usage()})
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
        lf_nn_dim=None,
        hf_nl_nn_dim=[10, 8],
        epochs_lf=500,
        epochs_mf=1000,
        lr_lf=0.005,
        wd_lf=0.000001,
        lr_mf=0.001,
        wd_mf=0.001,
        bounds=None,
        scale_input=False,
        scale_output=False,
        lf_model: model.Model = None,
        solve_linear=False,
    ):
        if lf_model is None and lf_nn_dim is None:
            raise ValueError("MultiFidelityNN needs either lf_nn_dim or lf_model")

        # Populate the multi-fidelity model data
        torch.nn.Module.__init__(self)
        mf_model.MFModel.__init__(self, X_lf, y_lf, X_hf, y_hf, bounds)

        # Use the trained lf model if one was given. Only its evaluate() is
        # called, which works in original units, so it needs no matching scaling
        if lf_model is not None:
            self.low_fidelity_nn = lf_model
        else:
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
                weight_decay=wd_lf,
                scale_input=scale_input,
                scale_output=scale_output,
                bounds=bounds,
            )

        # Evaluate the LF KAN model at the HF locations
        y_lf_at_hf = self.low_fidelity_nn.evaluate(self.x)
        y_lf_at_hf_test = self.low_fidelity_nn.evaluate(X_hf_test)

        # Generate the MF KAN model by combining the evaluated data set with the HF data
        x_mf = np.hstack((self.x, y_lf_at_hf))
        x_mf_test = np.hstack((X_hf_test, y_lf_at_hf_test))

        # The composite NN's inputs are [x, y_lf]: scale x by the known domain
        # bounds and the y_lf column by the LF training data's range, which
        # covers the domain far better than the few HF locations do
        if bounds is not None:
            mf_bounds = list(bounds) + [[float(np.min(y_lf)), float(np.max(y_lf))]]
        else:
            mf_bounds = None

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
            scale_input=scale_input,
            scale_output=scale_output,
            bounds=mf_bounds,
            solve_linear=solve_linear,
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


# Multifidelity Neural Network trained as in Meng and Karniadakis 2020
# Unlike MultiFidelityNN, all networks are trained at once on one loss (Eq. 6):
#   MSE_yL + MSE_yH + l2_rate * sum(beta^2)
# where beta is every weight in the LF and nonlinear networks plus alpha, and the
# outputs are combined as y_H = alpha * F_l + (1 - alpha) * F_nl (Eq. 5).
# The PDE term and the LF gradient term of the paper's loss are left out, because
# there is no PDE here and the benchmarks provide no gradient data.
# One epoch is one optimizer step, which is up to 20 iterations with L-BFGS.
class JointMultiFidelityNN(torch.nn.Module, mf_model.MFModel):
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
        lf_nn_dim=[20, 20],
        hf_nl_nn_dim=[10, 10],
        epochs=1000,
        learning_rate=0.001,
        l2_rate=0.01,
        optimizer="lbfgs",
        bounds=None,
        scale_input=False,
        scale_output=False,
    ):
        # Populate the multi-fidelity model data
        torch.nn.Module.__init__(self)
        mf_model.MFModel.__init__(self, X_lf, y_lf, X_hf, y_hf, bounds)

        self.device = torch.device(DEVICE)
        self.scale_input = scale_input
        self.scale_output = scale_output
        self.epochs = epochs
        self.l2_rate = l2_rate
        self.loss = torch.nn.MSELoss()

        # Zero epochs leaves the LF network untrained, so it holds the scaled
        # LF data and network layers for the joint training below
        self.low_fidelity_nn = MLPNetwork(
            X_lf,
            y_lf,
            X_test=X_lf_test,
            y_test=y_lf_test,
            hidden_layer_size=lf_nn_dim,
            activation=torch.nn.Tanh(),
            epochs=0,
            scale_input=scale_input,
            scale_output=scale_output,
            bounds=bounds,
        )
        self.low_fidelity_nn.apply(self.low_fidelity_nn.init_weights)

        # The composite inputs are [x, y_lf] with both already scaled by the LF
        # network, so CompositeNN does no input scaling of its own. The y_lf
        # column changes during training and is filled in by forward(), which
        # leaves zeros here as a placeholder that only sets the input size
        x_mf = np.hstack((self._scale_x(X_hf), np.zeros((X_hf.shape[0], 1))))
        x_mf_test = np.hstack(
            (self._scale_x(X_hf_test), np.zeros((X_hf_test.shape[0], 1)))
        )
        self.high_fidelity_nn = CompositeNN(
            x_mf,
            y_hf,
            X_test=x_mf_test,
            y_test=y_hf_test,
            layers_nl=hf_nl_nn_dim,
            activation_nl=torch.nn.Tanh(),
            train=False,
            scale_output=scale_output,
        )

        # Sigmoid keeps alpha in [0, 1], starting from 0.5
        self.alpha_raw = torch.nn.Parameter(torch.zeros((), device=self.device))

        if optimizer == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.parameters(), line_search_fn="strong_wolfe"
            )
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.training_error, self.testing_error = self.fit()

    def _scale_x(self, x):
        return self.low_fidelity_nn.x_scale.transform(x) if self.scale_input else x

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    # Takes scaled tensors and stays differentiable, so the HF error also
    # trains the LF network through the y_lf input
    def forward(self, x):
        x_mf = torch.cat((x, self.low_fidelity_nn(x)), dim=1)
        y_linear = self.high_fidelity_nn._linear_layers(x_mf)
        y_nl = self.high_fidelity_nn._nl_layers(x_mf)
        return self.alpha * y_linear + (1 - self.alpha) * y_nl

    def evaluate(self, x_in):
        x_tensor = torch.as_tensor(
            self._scale_x(x_in), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        y_hat = y_hat.detach().cpu().numpy()

        if self.scale_output:
            y_hat = self.high_fidelity_nn.y_scale.inverse(y_hat)
        return y_hat

    def l2_penalty(self):
        weights = [
            param
            for layers in (
                self.low_fidelity_nn._layers,
                self.high_fidelity_nn._nl_layers,
            )
            for name, param in layers.named_parameters()
            if name.endswith("weight")
        ]
        return sum(w.pow(2).sum() for w in weights) + self.alpha.pow(2)

    def joint_loss(self):
        lf_nn, hf_nn = self.low_fidelity_nn, self.high_fidelity_nn
        mse_lf = self.loss(lf_nn(lf_nn.X_train), lf_nn.y_train)
        mse_hf = self.loss(self.forward(hf_nn.X_train[:, :-1]), hf_nn.y_train)
        return mse_lf + mse_hf + self.l2_rate * self.l2_penalty()

    def training_accuracy(self):
        hf_nn = self.high_fidelity_nn
        with torch.no_grad():
            error = self.loss(self.forward(hf_nn.X_train[:, :-1]), hf_nn.y_train)
        return np.sqrt(error.item())

    def testing_accuracy(self):
        hf_nn = self.high_fidelity_nn
        with torch.no_grad():
            error = self.loss(self.forward(hf_nn.X_test[:, :-1]), hf_nn.y_test)
        return np.sqrt(error.item())

    # Trains on the full data set each step, since L-BFGS needs a consistent loss
    def fit(self, eval_interval=10):
        lf_nn = self.low_fidelity_nn
        train_error_history = []
        test_error_history = []
        lf_train_error_history = []
        lf_test_error_history = []
        for t in range(self.epochs):

            def closure():
                self.optimizer.zero_grad()
                loss_t = self.joint_loss()
                loss_t.backward()
                return loss_t

            self.optimizer.step(closure)

            # Only evaluate periodically and forward-fill the skipped epochs,
            # as in MLPNetwork and CompositeNN
            if t % eval_interval == 0 or t == self.epochs - 1:
                train_error = self.training_accuracy()
                test_error = self.testing_accuracy()
                lf_train_error = lf_nn.training_accuracy()
                lf_test_error = lf_nn.testing_accuracy()
            train_error_history.append(train_error)
            test_error_history.append(test_error)
            lf_train_error_history.append(lf_train_error)
            lf_test_error_history.append(lf_test_error)

        lf_nn.training_error = lf_train_error_history
        lf_nn.testing_error = lf_test_error_history
        return train_error_history, test_error_history


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
    # error_history = lf_model.fit()

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
