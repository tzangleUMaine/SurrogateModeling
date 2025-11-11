from __future__ import annotations

import kan
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

## Doing it the Pytorch way without using our custom feature extraction
DEVICE = "cpu"
import typing

import examples.multi_fidelity_functions as mfs
import mf_model
import model
import mfsnn
import data_handler
import csv


class KANModel(kan.KAN, model.Model):
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        x_mat_test: np.ndarray[float] = None,
        y_vec_test: np.ndarray[float] = None,
        spline_order: int = 3,
        width: list = [1, 2, 1],
        grid: int = 3,
        opt: str = "LBFGS",
        n_iter: int = 100,
        use_canned_dataset: bool = False,
    ):
        kan.KAN.__init__(
            self, width=width, grid=grid, k=spline_order, seed=1, device=DEVICE
        )
        model.Model.__init__(self, x_mat=x_mat, y_vec=y_vec)

        x_mat_tensor = torch.from_numpy(x_mat)
        y_vec_tensor = torch.from_numpy(y_vec)
        x_mat_test_tensor = torch.from_numpy(x_mat_test)
        y_vec_test_tensor = torch.from_numpy(y_vec_test)
        self.dataset = KANModel.create_dataset_from_test_train_data(
            x_mat_tensor,
            y_vec_tensor,
            x_mat_test_tensor,
            y_vec_test_tensor,
            device=DEVICE,
        )

        if use_canned_dataset:
            f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
            self.dataset = kan.create_dataset(f, n_var=2, device=DEVICE)

        self.results = self.fit(dataset=self.dataset, opt=opt, steps=n_iter)

    def evaluate(self, x):
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            return self.forward(x_tensor)

    @classmethod
    def create_dataset_from_test_train_data(
        self, inputs_train, labels_train, inputs_test, labels_test, device="cpu"
    ):
        """
        create dataset from data

        Args:
        -----
            inputs_train : 2D torch.float
            labels_train : 2D torch.float
            inputs_test : 2D torch.float
            labels_test : 2D torch.float
            device : str

        Returns:
        --------
            dataset (dictionary)
        """
        dataset = {}
        dataset["train_input"] = inputs_train[:].detach().to(device)
        dataset["test_input"] = inputs_test[:].detach().to(device)
        dataset["train_label"] = labels_train[:].detach().to(device)
        dataset["test_label"] = labels_test[:].detach().to(device)

        return dataset


class CompositeKANModel(kan.KAN, model.Model):
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        x_mat_test: np.ndarray[float] = None,
        y_vec_test: np.ndarray[float] = None,
        spline_order: int = 3,
        width: list = [1, 2, 1],
        grid: int = 3,
        opt: str = "LBFGS",
        n_iter: int = 100,
    ):
        kan.KAN.__init__(
            self, width=width, grid=grid, k=spline_order, seed=1, device=DEVICE
        )
        model.Model.__init__(self, x_mat=x_mat, y_vec=y_vec)
        x_mat_tensor = torch.from_numpy(x_mat)
        y_vec_tensor = torch.from_numpy(y_vec)
        x_mat_test_tensor = torch.from_numpy(x_mat_test)
        y_vec_test_tensor = torch.from_numpy(y_vec_test)
        self.dataset = KANModel.create_dataset_from_test_train_data(
            x_mat_tensor,
            y_vec_tensor,
            x_mat_test_tensor,
            y_vec_test_tensor,
            device=DEVICE,
        )

        # Get the input and output shape of the network
        __, input_size = x_mat.shape
        __, output_size = y_vec.shape

        # Generate a linear neural network to create a shortcut for model training
        self._linear_nn = torch.nn.Linear(
            in_features=input_size, out_features=output_size, bias=True
        ).to(DEVICE)

        self.results = self.fit(dataset=self.dataset, opt=opt, steps=n_iter)

    # def forward(self, x_in, singularity_avoiding=False, y_th=1000.0):
    #     y_kan = kan.KAN.forward(self,x_in,singularity_avoiding=singularity_avoiding,y_th=y_th)
    #     y_linear = self._linear_nn(x_in)
    #     y_hat = (1.0 - self.alpha) * y_kan + self.alpha * y_linear
    #     return y_hat
    def forward(self, x_in, singularity_avoiding=False, y_th=1000.0):
        y_linear = self._linear_nn(x_in)
        y_kan = kan.KAN.forward(
            self, x_in, singularity_avoiding=singularity_avoiding, y_th=y_th
        )
        y_hat = y_linear + y_kan
        return y_hat

    def evaluate(self, x_in):
        x_tensor = torch.from_numpy(x_in)
        with torch.no_grad():
            return self.forward(x_tensor)


class MFKANModel(mf_model.MFModel):
    def __init__(
        self,
        x_lf: np.ndarray[float],
        y_lf: np.ndarray[float],
        x_lf_test: np.ndarray[float],
        y_lf_test: np.ndarray[float],
        x_hf: np.ndarray[float],
        y_hf: np.ndarray[float],
        x_hf_test: np.ndarray[float],
        y_hf_test: np.ndarray[float],
        spline_order_lf: int = 3,
        width_lf: list = [1, 2, 1],
        grid_lf: int = 3,
        opt_lf: str = "LBFGS",
        n_iter_lf: int = 100,
        spline_order_mf: int = 3,
        width_mf: list = [1, 2, 1],
        grid_mf: int = 3,
        opt_mf: str = "LBFGS",
        n_iter_mf: int = 100,
    ):
        # Init the multifidelity model data
        mf_model.MFModel.__init__(self, x_lf, y_lf, x_hf, y_hf)

        # Create the LF KAN model using low fidelity data
        self.lf_kan_model = KANModel(
            x_mat=self.x_lf,
            y_vec=self.y_lf,
            x_mat_test=x_lf_test,
            y_vec_test=y_lf_test,
            spline_order=spline_order_lf,
            width=width_lf,
            grid=grid_lf,
            opt=opt_lf,
            n_iter=n_iter_lf,
        )

        # Evaluate the LF KAN model at the HF locations
        y_lf_at_hf = self.lf_kan_model.evaluate(self.x).numpy()
        y_lf_at_hf_test = self.lf_kan_model.evaluate(x_hf_test).numpy()

        # Generate the MF KAN model by combining the evaluated data set with the HF data
        x_mf = np.hstack((self.x, y_lf_at_hf))
        x_mf_test = np.hstack((x_hf_test, y_lf_at_hf_test))

        self.mf_kan_model = CompositeKANModel(
            x_mat=x_mf,
            y_vec=self.y,
            x_mat_test=x_mf_test,
            y_vec_test=y_hf_test,
            spline_order=spline_order_mf,
            width=width_mf,
            grid=grid_mf,
            opt=opt_mf,
            n_iter=n_iter_mf,
        )

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)

    def forward(self, x):
        # Evaluate the LF KAN model at the HF locations
        y_lf_at_x = self.lf_kan_model.evaluate(x).numpy().reshape(-1, 1)

        # Generate the MF input by combining x and y_lf_at_x
        x_mf = np.hstack((x, y_lf_at_x))

        # Evaluate the MF KAN model
        return self.mf_kan_model.evaluate(x_mf)
