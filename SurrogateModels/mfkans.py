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
            x_mat_tensor, y_vec_tensor, x_mat_tensor, y_vec_test_tensor, device=DEVICE
        )

        self.fit(dataset=self.dataset, opt=opt, steps=100)

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
        spline_order_mf: int = 3,
        width_mf: list = [1, 2, 1],
        grid_mf: int = 3,
        opt_mf: str = "LBFGS",
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
        )

        # Evaluate the LF KAN model at the HF locations
        y_lf_at_hf = self.lf_kan_model.evaluate(self.x).numpy()
        y_lf_at_hf_test = self.lf_kan_model.evaluate(x_hf_test).numpy()

        # Generate the MF KAN model by combining the evaluated data set with the HF data
        x_mf = np.vstack((self.x, y_lf_at_hf))
        x_mf_test = np.vstack((x_hf_test, y_lf_at_hf_test))

        self.mf_kan_model = KANModel(
            x_mat=x_mf,
            y_vec=self.y,
            x_mat_test=x_mf_test,
            y_vec_test=y_hf_test,
            spline_order=spline_order_mf,
            width=width_mf,
            grid=grid_mf,
            opt=opt_mf,
        )

    def evaluate(self, x):
        return
