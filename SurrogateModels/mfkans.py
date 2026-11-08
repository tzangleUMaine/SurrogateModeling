from __future__ import annotations

import contextlib
import functools
import random

import kan
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

## Doing it the Pytorch way without using our custom feature extraction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import typing

import examples.multi_fidelity_functions as mfs
import mf_model
import model
import mfsnn
import data_handler
import csv


# torch.linalg.lstsq on CUDA only implements the 'gels' driver, which requires
# a full-rank input matrix and raises torch.linalg.LinAlgError the moment one
# isn't -- common whenever a fit batch is small relative to a KAN's grid/width
# (e.g. a small-n point in a sample-size sweep). pykan's own curve2coef (used
# both by our refine_model() CPU workaround's caller, kan.MultKAN.refine(),
# AND internally by MultKAN.fit()'s default update_grid=True grid-adaptation
# step) wraps the call in a bare `except: print('lstsq failed')` that
# swallows the real error and leaves its `coef` local unset, surfacing as an
# opaque UnboundLocalError far from the actual cause. CPU's default driver
# (gelsd/gelsy) handles rank-deficient systems correctly instead of crashing,
# so transparently retry there and move the result back -- this patches every
# call site project-wide, not just the ones we've hit so far.
_torch_linalg_lstsq = torch.linalg.lstsq


def _lstsq_with_cpu_fallback(*args, **kwargs):
    try:
        return _torch_linalg_lstsq(*args, **kwargs)
    except torch.linalg.LinAlgError:
        device = next(
            (a.device for a in args if isinstance(a, torch.Tensor)), None
        )
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_kwargs = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }
        result = _torch_linalg_lstsq(*cpu_args, **cpu_kwargs)
        if device is None:
            return result
        return torch.return_types.linalg_lstsq(
            tuple(
                getattr(result, f).to(device) for f in ("solution", "residuals", "rank", "singular_values")
            )
        )


torch.linalg.lstsq = _lstsq_with_cpu_fallback


# kan.KAN.__init__ (and refine()) reseed the global torch, numpy and random
# generators, so every build would leave them in the same state and any
# unseeded draws afterwards would repeat. Building and training run under the
# model's own seed, then the caller's generator state comes back.
@contextlib.contextmanager
def _global_rng_restored():
    np_state = np.random.get_state()
    py_state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _keeps_global_rng(init):
    @functools.wraps(init)
    def wrapper(*args, **kwargs):
        with _global_rng_restored():
            init(*args, **kwargs)

    return wrapper


class KANModel(kan.KAN, model.Model):
    @_keeps_global_rng
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        x_mat_test: np.ndarray[float] = None,
        y_vec_test: np.ndarray[float] = None,
        spline_order: int = 3,
        hidden_layers: list = [2],
        grid: int | list = [3],
        opt: str = "LBFGS",
        n_iter: int = 100,
        lr: float = 1.0,
        device: str = DEVICE,
        scale_input: bool = False,
        scale_output: bool = False,
        x_scale_method: str = "bounded",
        y_scale_method: str = "standard",
        bounds=None,
        lamb: float = 0.0,
        seed: int = 1,
    ):
        # grid is one size, or a list of sizes to refine through in order
        grids = [grid] if isinstance(grid, int) else list(grid)

        # hidden_layers only holds the hidden sizes: the data sets the input and
        # output sizes, because pykan silently drops any extra input columns
        kan.KAN.__init__(
            self,
            width=[x_mat.shape[1], *hidden_layers, y_vec.shape[1]],
            grid=grids[0],
            k=spline_order,
            seed=seed,
            device=device,
        )
        model.Model.__init__(
            self,
            x_mat=x_mat,
            y_vec=y_vec,
            bounds=bounds,
            x_scale_method=x_scale_method,
            y_scale_method=y_scale_method,
            scale_input=scale_input,
            scale_output=scale_output,
        )

        self.scale_input = scale_input
        self.scale_output = scale_output

        # self.x / self.y were scaled in place above (by Model.__init__) when
        # requested; apply that same fitted transform to the held-out test
        # data so train/test stay on the same scale.
        if self.scale_input and x_mat_test is not None:
            x_mat_test = self.x_scale.transform(x_mat_test)
        if self.scale_output and y_vec_test is not None:
            y_vec_test = self.y_scale.transform(y_vec_test)

        # kan.KAN runs in float32 and doesn't cast its inputs, and the scale
        # transforms above can upcast to float64, so cast explicitly here.
        x_mat_tensor = torch.from_numpy(self.x.astype(np.float32))
        y_vec_tensor = torch.from_numpy(self.y.astype(np.float32))
        x_mat_test_tensor = torch.from_numpy(x_mat_test.astype(np.float32))
        y_vec_test_tensor = torch.from_numpy(y_vec_test.astype(np.float32))
        self.dataset = KANModel.create_dataset_from_test_train_data(
            x_mat_tensor,
            y_vec_tensor,
            x_mat_test_tensor,
            y_vec_test_tensor,
            device=device,
        )

        mins = x_mat.min(axis=0)
        maxs = x_mat.max(axis=0)

        self.bounds = [[float(mn), float(mx)] for mn, mx in zip(mins, maxs)]

        # Initialize storage for test and training monitoring
        self.train_losses = []
        self.test_losses = []

        results = self.fit(
            dataset=self.dataset,
            opt=opt,
            steps=n_iter,
            lr=lr,
            lamb=lamb,
        )
        self.train_losses += results["train_loss"]
        self.test_losses += results["test_loss"]

        for g in grids[1:]:
            self.refine_model(g)
            self.results = self.fit(
                dataset=self.dataset,
                opt=opt,
                steps=n_iter,
                lr=lr,
                lamb=lamb,
            )
            self.train_losses += self.results["train_loss"]
            self.test_losses += self.results["test_loss"]

    def evaluate(self, x):
        if self.scale_input:
            x = self.x_scale.transform(x)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        y_hat = y_hat.detach().cpu().numpy()

        if self.scale_output:
            y_hat = self.y_scale.inverse(y_hat)
        return y_hat

    def refine_model(self, new_grid: int = None):
        # pykan's refine() fits new spline coefficients via torch.linalg.lstsq,
        # which on CUDA only supports the full-rank 'gels' driver and crashes
        # (UnboundLocalError, via a swallowed _LinAlgError) the moment a
        # spline-fit batch is rank-deficient; CPU's default driver tolerates
        # that. Refine on CPU -- it's a one-off per grid stage, not the hot
        # training loop -- then restore the model to its working device.
        target_device = self.device
        self.to("cpu")
        if self.cache_data is not None:
            self.cache_data = self.cache_data.to("cpu")
        refined = self.refine(new_grid)
        self.__dict__.update(refined.__dict__)
        self.to(target_device)
        return self

    def plot_fit_error(
        self,
        name="Model Error",
        color_train="black",
        color_test="blue",
        label_train="train",
        label_test="test",
        ax=None,
    ):
        ax = self.plot_train_error(
            name=name, color=color_train, label=label_train, ax=ax
        )
        ax = self.plot_test_error(name=name, color=color_test, label=label_test, ax=ax)
        return ax

    def plot_test_error(self, name="Model Error", color="black", label="test", ax=None):
        # Plot the function
        if ax == None:
            fig = plt.figure(name)
            ax = fig.add_subplot()
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.plot(self.test_losses, color=color, label=label)
        return ax

    def plot_train_error(
        self, name="Model Error", color="blue", label="train", ax=None
    ):
        # Plot the function
        if ax == None:
            fig = plt.figure(name)
            ax = fig.add_subplot()
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.plot(self.train_losses, color=color, label=label)
        return ax

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
    @_keeps_global_rng
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        x_mat_test: np.ndarray[float] = None,
        y_vec_test: np.ndarray[float] = None,
        spline_order: int = 3,
        hidden_layers: list = [2],
        grid: int | list = [3],
        opt: str = "LBFGS",
        n_iter: int = 100,
        lr: float = 1.0,
        device: str = DEVICE,
        scale_input: bool = False,
        scale_output: bool = False,
        x_scale_method: str = "bounded",
        y_scale_method: str = "standard",
        bounds=None,
        seed: int = 1,
        lamb: float = 0.0,
    ):
        # grid is one size, or a list of sizes to refine through in order
        grids = [grid] if isinstance(grid, int) else list(grid)
        # hidden_layers only holds the hidden sizes: the data sets the input and
        # output sizes, because pykan silently drops any extra input columns
        kan.KAN.__init__(
            self,
            width=[x_mat.shape[1], *hidden_layers, y_vec.shape[1]],
            grid=grids[0],
            k=spline_order,
            seed=seed,
            device=device,
        )
        model.Model.__init__(
            self,
            x_mat=x_mat,
            y_vec=y_vec,
            bounds=bounds,
            x_scale_method=x_scale_method,
            y_scale_method=y_scale_method,
            scale_input=scale_input,
            scale_output=scale_output,
        )

        self.scale_input = scale_input
        self.scale_output = scale_output

        # self.x / self.y were scaled in place above when requested; apply the
        # same fitted transforms to the test data (see KANModel)
        if self.scale_input and x_mat_test is not None:
            x_mat_test = self.x_scale.transform(x_mat_test)
        if self.scale_output and y_vec_test is not None:
            y_vec_test = self.y_scale.transform(y_vec_test)

        # kan.KAN runs in float32 and doesn't cast its inputs (see KANModel)
        x_mat_tensor = torch.from_numpy(self.x.astype(np.float32))
        y_vec_tensor = torch.from_numpy(self.y.astype(np.float32))
        x_mat_test_tensor = torch.from_numpy(x_mat_test.astype(np.float32))
        y_vec_test_tensor = torch.from_numpy(y_vec_test.astype(np.float32))
        self.dataset = KANModel.create_dataset_from_test_train_data(
            x_mat_tensor,
            y_vec_tensor,
            x_mat_test_tensor,
            y_vec_test_tensor,
            device=device,
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
        ).to(device)

        # Initialize storage for test and training monitoring
        self.train_losses = []
        self.test_losses = []

        results = self.fit(
            dataset=self.dataset,
            opt=opt,
            steps=n_iter,
            lr=lr,
            lamb=lamb,
        )

        self.train_losses += results["train_loss"]
        self.test_losses += results["test_loss"]

        for g in grids[1:]:
            self.refine_model(g)
            results = self.fit(
                dataset=self.dataset,
                opt=opt,
                steps=n_iter,
                lr=lr,
                lamb=lamb,
            )
            self.train_losses += results["train_loss"]
            self.test_losses += results["test_loss"]

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
        # See KANModel.refine_model -- pykan's refine() crashes on CUDA via a
        # swallowed torch.linalg.lstsq rank-deficiency error, so do this step
        # on CPU and restore the model to its working device afterward.
        target_device = self.device
        # refine() builds a plain KAN, so the __dict__ update below replaces
        # _modules and drops _linear_nn. Keep the trained layer and reattach it
        # so the linear shortcut carries across grid stages.
        linear_nn = self._linear_nn
        self.to("cpu")
        if self.cache_data is not None:
            self.cache_data = self.cache_data.to("cpu")
        refined = self.refine(new_grid)
        self.__dict__.update(refined.__dict__)
        self._linear_nn = linear_nn
        self.to(target_device)
        return self

    def evaluate(self, x_in, loud=False):
        if self.scale_input:
            x_in = self.x_scale.transform(x_in)
        x_tensor = torch.as_tensor(x_in, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = self.forward(x_tensor, loud=loud)
        y_hat = y_hat.detach().cpu().numpy()

        if self.scale_output:
            y_hat = self.y_scale.inverse(y_hat)
        return y_hat

    def plot_fit_error(
        self,
        name="Model Error",
        color_train="black",
        color_test="blue",
        label_train="train",
        label_test="test",
        ax=None,
    ):
        ax = self.plot_train_error(
            name=name, color=color_train, label=label_train, ax=ax
        )
        ax = self.plot_test_error(name=name, color=color_test, label=label_test, ax=ax)
        return ax

    def plot_test_error(self, name="Model Error", color="black", label="test", ax=None):
        # Plot the function
        if ax == None:
            fig = plt.figure(name)
            ax = fig.add_subplot()
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.plot(self.test_losses, color=color, label=label)
        return ax

    def plot_train_error(
        self, name="Model Error", color="blue", label="train", ax=None
    ):
        # Plot the function
        if ax == None:
            fig = plt.figure(name)
            ax = fig.add_subplot()
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.plot(self.train_losses, color=color, label=label)
        return ax


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
        hidden_layers_lf: list = [2],
        grid_lf: int | list = [3],
        opt_lf: str = "LBFGS",
        n_iter_lf: int = 100,
        lr_lf: float = 1.0,
        lamb_lf: float = 0.0,
        spline_order_mf: int = 3,
        hidden_layers_mf: list = [2],
        grid_mf: int | list = [3],
        opt_mf: str = "LBFGS",
        n_iter_mf: int = 100,
        lr_mf: float = 1.0,
        lamb_mf: float = 0.0,
        scale_input: bool = False,
        scale_output: bool = False,
        bounds=None,
        seed_lf: int = 1,
        seed_mf: int = 1,
        lf_model: model.Model = None,
    ):
        # The composite model has at least two inputs, and sampled bounds only
        # work for one (see DataTransform)
        if scale_input and bounds is None:
            raise ValueError("MFKANModel needs bounds when scale_input is True")

        # Init the multifidelity model data
        mf_model.MFModel.__init__(self, x_lf, y_lf, x_hf, y_hf, bounds)
        self.scale_input = scale_input

        # Use the trained lf model if one was given. Only its evaluate() is
        # called, which works in original units, so it needs no matching scaling
        if lf_model is not None:
            self.lf_kan_model = lf_model
        else:
            # Create the LF KAN model using low fidelity data
            self.lf_kan_model = KANModel(
                x_mat=self.x_lf,
                y_vec=self.y_lf,
                x_mat_test=x_lf_test,
                y_vec_test=y_lf_test,
                spline_order=spline_order_lf,
                hidden_layers=hidden_layers_lf,
                grid=grid_lf,
                opt=opt_lf,
                n_iter=n_iter_lf,
                lr=lr_lf,
                scale_input=scale_input,
                scale_output=scale_output,
                bounds=bounds,
                seed=seed_lf,
                lamb=lamb_lf,
            )

        # The range used to scale the y_lf column when the composite doesn't
        y_lf_at_hf = self.lf_kan_model.evaluate(self.x)
        self.lf_at_hf_min = y_lf_at_hf.min()
        self.lf_at_hf_max = y_lf_at_hf.max()

        # Generate the MF KAN model by combining the evaluated data set with the HF data
        x_mf = self._composite_input(self.x)
        x_mf_test = self._composite_input(x_hf_test)

        # The composite's inputs are [x, y_lf]: scale x by the known domain bounds
        # and the y_lf column by the LF training data's range, which covers the
        # domain far better than the few HF locations do
        if scale_input:
            mf_bounds = list(bounds) + [[float(np.min(y_lf)), float(np.max(y_lf))]]
        else:
            mf_bounds = None

        self.mf_kan_model = CompositeKANModel(
            x_mat=x_mf,
            y_vec=self.y,
            x_mat_test=x_mf_test,
            y_vec_test=y_hf_test,
            spline_order=spline_order_mf,
            hidden_layers=hidden_layers_mf,
            grid=grid_mf,
            opt=opt_mf,
            n_iter=n_iter_mf,
            lr=lr_mf,
            scale_input=scale_input,
            scale_output=scale_output,
            bounds=mf_bounds,
            seed=seed_mf,
            lamb=lamb_mf,
        )

    def _composite_input(self, x):
        # Evaluate the LF KAN model at the given locations
        y_lf_at_x = self.lf_kan_model.evaluate(x).reshape(-1, 1)

        # With scale_input the composite model scales this column itself
        if not self.scale_input:
            y_lf_at_x = data_handler.scale_array(
                y_lf_at_x, self.lf_at_hf_min, self.lf_at_hf_max
            )

        # Generate the MF input by combining x and y_lf_at_x
        return np.hstack((x, y_lf_at_x))

    def evaluate(self, x, loud=False):
        return self.forward(x, loud=loud)

    def forward(self, x, loud=False):
        # Evaluate the MF KAN model
        return self.mf_kan_model.evaluate(self._composite_input(x), loud=loud)

    def plot_lf_fit_error(self):
        self.lf_kan_model.plot_fit_error()
        plt.show()

    def plot_mf_fit_error(self):
        self.mf_kan_model.plot_fit_error()
        plt.show()
