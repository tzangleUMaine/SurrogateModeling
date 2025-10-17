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
        spline_order: int = 3,
        width: list = [2, 1, 1],
        grid: int = 3,
        opt: str = "LBFGS",
    ):
        kan.KAN.__init__(
            self, width=width, grid=grid, k=spline_order, seed=1, device=DEVICE
        )
        model.Model.__init__(self, x_mat=x_mat, y_vec=y_vec)
        x_mat_tensor = torch.from_numpy(x_mat)
        y_vec_tensor = torch.from_numpy(y_vec)
        dataset = kan.create_dataset_from_data(
            x_mat_tensor, y_vec_tensor, train_ratio=1.0, device=DEVICE
        )

        self.fit(dataset=dataset, opt=opt, steps=20)

    def evaluate(self, x):
        x_tensor = torch.from_numpy(x)
        return self(x)


def test_kan():
    mf_example = mfs.ForresterExample()

    # Define the example function bounds and dimensions here
    dim = mf_example.dimensions
    xmin = mf_example.bounds[0][0]
    xmax = mf_example.bounds[0][1]

    n_lf = 11  # number of low fidelity points
    n_hf = 6  # number of high fidelity points
    n_test = 40  # number of test points

    # Generate the analytical functions
    hf_fxn = mf_example.high_fidelity
    lf_fxn = mf_example.low_fidelity

    X_lf = np.linspace(xmin, xmax, n_lf, dtype=np.float32).reshape(-1, 1)
    y_lf = lf_fxn(X_lf)

    X_hf = np.linspace(xmin, xmax, n_hf, dtype=np.float32).reshape(-1, 1)
    y_hf = hf_fxn(X_hf)

    X_test = np.linspace(xmin, xmax, n_test, dtype=np.float32).reshape(-1, 1)
    y_lf_test = lf_fxn(X_test)
    y_hf_test = hf_fxn(X_test)

    plot_name = mf_example.name + " Samples"

    ax = mf_example.plot_functions(name=plot_name)

    ax.scatter(X_hf, y_hf, color="black", label="HF samples", marker="o")
    ax.scatter(X_lf, y_lf, color="blue", label="LF samples", marker="o")

    plt.grid()
    plt.legend()
    plt.show()

    forrester_kan = KANModel(X_hf, y_hf)


test_kan()
