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
        grid: list = [3],
        opt: str = "LBFGS",
        n_iter: int = 100,
    ):
        g = grid.pop(0)

        kan.KAN.__init__(
            self, width=width, grid=g, k=spline_order, seed=1, device=DEVICE
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

        mins = x_mat.min(axis=0)
        maxs = x_mat.max(axis=0)

        self.bounds = [[float(mn), float(mx)] for mn, mx in zip(mins, maxs)]

        self.results = self.fit(
            dataset=self.dataset,
            opt=opt,
            steps=n_iter,
        )
        for g in grid:
            self.refine_model(g)
            self.results = self.fit(
                dataset=self.dataset,
                opt=opt,
                steps=n_iter,
            )

    def evaluate(self, x):
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            return self.forward(x_tensor)

    def refine_model(self, new_grid: int = None):
        refined = self.refine(new_grid)
        self.__dict__.update(refined.__dict__)
        return self

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
        grid: int = [3],
        opt: str = "LBFGS",
        n_iter: int = 100,
    ):
        g = grid.pop(0)
        kan.KAN.__init__(
            self, width=width, grid=g, k=spline_order, seed=1, device=DEVICE
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

        mins = x_mat.min(axis=0)
        maxs = x_mat.max(axis=0)

        self.bounds = [[float(mn), float(mx)] for mn, mx in zip(mins, maxs)]

        # Get the input and output shape of the network
        __, self.input_size = x_mat.shape
        __, self.output_size = y_vec.shape

        # Generate a linear neural network to create a shortcut for model training
        self._linear_nn = torch.nn.Linear(
            in_features=self.input_size, out_features=self.output_size, bias=True
        ).to(DEVICE)

        self.results = self.fit(dataset=self.dataset, opt=opt, steps=n_iter)

        for g in grid:
            self.refine_model(g)
            self.results = self.fit(
                dataset=self.dataset,
                opt=opt,
                steps=n_iter,
            )

    def forward(self, x_in, singularity_avoiding=False, y_th=1000.0, loud=False):
        y_linear = self._linear_nn(x_in)
        y_kan = kan.KAN.forward(
            self, x_in, singularity_avoiding=singularity_avoiding, y_th=y_th
        )
        y_hat = y_linear + y_kan
        if loud == True:
            print(f"linear output: {y_linear/y_hat}")
            print(f"non-lin output: {y_kan/y_hat}")
        return y_hat

    def refine_model(self, new_grid: int = None):
        refined = self.refine(new_grid)
        self.__dict__.update(refined.__dict__)
        # Generate a linear neural network to create a shortcut for model training
        self._linear_nn = torch.nn.Linear(
            in_features=self.input_size, out_features=self.output_size, bias=True
        ).to(DEVICE)
        print("Does it make it here?")
        return self

    def evaluate(self, x_in, loud=False):
        x_tensor = torch.from_numpy(x_in)
        with torch.no_grad():
            return self.forward(x_tensor, loud=loud)


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
        grid_lf: int = [3],
        opt_lf: str = "LBFGS",
        n_iter_lf: int = 100,
        spline_order_mf: int = 3,
        width_mf: list = [1, 2, 1],
        grid_mf: int = [3],
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
        self.lf_at_hf_min = y_lf_at_hf.min()
        self.lf_at_hf_max = y_lf_at_hf.max()
        y_lf_at_hf = data_handler.scale_array(
            y_lf_at_hf, self.lf_at_hf_min, self.lf_at_hf_max
        )

        y_lf_at_hf_test = self.lf_kan_model.evaluate(x_hf_test).numpy()
        y_lf_at_hf_test = data_handler.scale_array(
            y_lf_at_hf_test, self.lf_at_hf_min, self.lf_at_hf_max
        )

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

    def evaluate(self, x, loud=False):
        with torch.no_grad():
            return self.forward(x, loud=loud)

    def forward(self, x, loud=False):
        # Evaluate the LF KAN model at the HF locations
        y_lf_at_x = self.lf_kan_model.evaluate(x).numpy().reshape(-1, 1)
        y_lf_at_x = data_handler.scale_array(
            y_lf_at_x, self.lf_at_hf_min, self.lf_at_hf_max
        )

        # Generate the MF input by combining x and y_lf_at_x
        x_mf = np.hstack((x, y_lf_at_x))

        # Evaluate the MF KAN model
        return self.mf_kan_model.evaluate(x_mf, loud=loud)
