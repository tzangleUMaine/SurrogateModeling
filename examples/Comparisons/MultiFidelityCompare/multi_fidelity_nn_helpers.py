import concurrent.futures
import contextlib
import csv
import io
import os
import time

import numpy as np
import torch
from torch import nn as tnn

import examples.multi_fidelity_functions as mfs
from SurrogateModels import mfsnn


def generate_mf_nn_model_comparisons(
    name,
    func: mfs.MultiFidelityTemplate,
    X_test,
    y_test,
    n_lf=400,
    sample_low=10,
    sample_high=201,
    step_size=10,
    seeds=3,
    points=None,
    model_error=None,
    training_time=None,
    evaluation_time=None,
    lf_nn_dim=[30, 30, 10],
    hf_nl_nn_dim=[6, 6],
    epochs_lf=30000,
    epochs_mf=30000,
    lr_lf=1e-4,
    wd_lf=1e-6,
    lr_mf=1e-4,
    wd_mf=1e-6,
    bounds=None,
    output_file="mf_nn_model_comparison_output.csv",
):
    # Start fresh lists per call; pass existing lists to extend them instead
    points = [] if points is None else points
    model_error = [] if model_error is None else model_error
    training_time = [] if training_time is None else training_time
    evaluation_time = [] if evaluation_time is None else evaluation_time

    bounds = bounds if bounds is not None else func.bounds
    y_lf_test = func.low_fidelity(X_test)

    # Each run appends its own row immediately, so a crash partway through a
    # sweep still leaves completed runs on disk. There is no header, because
    # load_multi_fidelity_csv_results in data_viewer.ipynb doesn't expect one.
    n_points = list(range(sample_low, sample_high, step_size))
    for seed in range(0, seeds):
        X_lf = func.sample_random(n_lf, seed=seed)
        y_lf = func.low_fidelity(X_lf)

        # The LF data doesn't change with the HF sample count, so train the LF
        # network once per seed and share it across this seed's runs. Its
        # weights are saved next to the CSV, e.g. results_lf_seed0.pt for
        # results.csv. Loading them back needs the same LF samples and bounds,
        # because MLPNetwork fits its scaling to the data it's given.
        lf_weight_file = f"{os.path.splitext(output_file)[0]}_lf_seed{seed}.pt"
        lf_model = mfsnn.MLPNetwork(
            X=X_lf,
            y=y_lf,
            X_test=X_test,
            y_test=y_lf_test,
            activation=tnn.Tanh(),
            hidden_layer_size=lf_nn_dim,
            epochs=epochs_lf,
            learning_rate=lr_lf,
            weight_decay=wd_lf,
            bounds=bounds,
            scale_input=True,
            scale_output=True,
            weight_file=lf_weight_file,
            write_weights=True,
        )

        for n in n_points:
            X_hf = func.sample_random(n, seed=seed)
            y_hf = func.high_fidelity(X_hf)

            # CompositeNN prints a progress line every 10 epochs, so capture it
            with contextlib.redirect_stdout(io.StringIO()):
                model = mfsnn.MultiFidelityNN(
                    X_lf,
                    y_lf,
                    X_test,
                    y_lf_test,
                    X_hf,
                    y_hf,
                    X_test,
                    y_test,
                    hf_nl_nn_dim=hf_nl_nn_dim,
                    epochs_mf=epochs_mf,
                    lr_mf=lr_mf,
                    wd_mf=wd_mf,
                    bounds=bounds,
                    scale_input=True,
                    scale_output=True,
                    lf_model=lf_model,
                )

            # Includes the shared LF network's training, since each model depends on it
            training_time.append(model.training_time)

            t0 = time.perf_counter()
            error = float(model.model_error(X_test, y_test))
            evaluation_time.append(time.perf_counter() - t0)
            model_error.append(error)
            points.append(n)
            print(
                f"Run {name} :: seed {seed} :: lf points {n_lf} :: hf points {n} :: error {error} :: train_time {training_time[-1]} "
            )
            with open(output_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        n_lf,
                        points[-1],
                        model_error[-1],
                        training_time[-1],
                        evaluation_time[-1],
                    ]
                )
    return points, model_error, evaluation_time, training_time


COMPOSITE_SIZE_HEADER = [
    "depth",
    "width",
    "repeat",
    "test_rmse",
    "train_rmse",
    "training_time",
    "weight_decay",
    "solve_linear",
]


def _use_one_cpu_thread():
    # These networks are small enough that a single CPU thread trains them about
    # twice as fast as the GPU, which also leaves the other cores free for more fits
    mfsnn.DEVICE = "cpu"
    torch.set_num_threads(1)


def _fit_composite(
    x_mf,
    y_hf,
    x_mf_test,
    y_test,
    mf_bounds,
    depth,
    width,
    repeat,
    epochs,
    lr,
    wd,
    solve_linear=False,
):
    # The repeat number seeds the initial weights, so each run is reproducible
    torch.manual_seed(repeat)

    # CompositeNN prints a progress line every 10 epochs, so capture it
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        composite = mfsnn.CompositeNN(
            x_mf,
            y_hf,
            X_test=x_mf_test,
            y_test=y_test,
            layers_nl=[width] * depth,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=wd,
            activation_nl=tnn.Tanh(),
            scale_input=True,
            scale_output=True,
            bounds=mf_bounds,
            solve_linear=solve_linear,
        )
    training_time = time.perf_counter() - t0

    return {
        "depth": depth,
        "width": width,
        "repeat": repeat,
        "test_rmse": float(composite.model_error(x_mf_test, y_test)),
        "train_rmse": float(composite.model_error(x_mf, y_hf)),
        "training_time": training_time,
        "weight_decay": wd,
        "solve_linear": solve_linear,
        "training_error": composite.training_error,
        "testing_error": composite.testing_error,
        "path_history": composite.path_history,
    }


# Trains the composite network of MultiFidelityNN at each depth and width, with
# every hidden layer the same width. The LF model is only evaluated here, so all
# runs see identical [x, y_lf(x)] inputs and differ only in the composite network.
# wd_mf can also be a list of values, and every size is then run at each of them.
def generate_composite_size_comparisons(
    lf_model,
    func: mfs.MultiFidelityTemplate,
    y_lf,
    X_hf,
    y_hf,
    X_test,
    y_test,
    depths=(1, 2, 3),
    widths=(3, 6, 9, 12, 16, 20),
    repeats=5,
    epochs_mf=30000,
    lr_mf=1e-4,
    wd_mf=1e-6,
    solve_linear=False,
    n_workers=8,
    output_file="composite_size_comparison_output.csv",
):
    x_mf = np.hstack((X_hf, lf_model.evaluate(X_hf)))
    x_mf_test = np.hstack((X_test, lf_model.evaluate(X_test)))

    # Same input scaling as MultiFidelityNN
    mf_bounds = list(func.bounds) + [[float(np.min(y_lf)), float(np.max(y_lf))]]

    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as file:
            csv.writer(file).writerow(COMPOSITE_SIZE_HEADER)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_use_one_cpu_thread
    ) as executor:
        runs = [
            executor.submit(
                _fit_composite,
                x_mf,
                y_hf,
                x_mf_test,
                y_test,
                mf_bounds,
                depth,
                width,
                repeat,
                epochs_mf,
                lr_mf,
                wd,
                solve_linear,
            )
            for wd in np.atleast_1d(wd_mf).tolist()
            for depth in depths
            for width in widths
            for repeat in range(repeats)
        ]

        # Each run appends its own row immediately, so a crash partway through a
        # sweep still leaves completed runs on disk
        for run in concurrent.futures.as_completed(runs):
            result = run.result()
            results.append(result)
            print(
                f"{len(results)}/{len(runs)} :: wd {result['weight_decay']:g} :: depth {result['depth']} :: width {result['width']} :: "
                f"repeat {result['repeat']} :: test {result['test_rmse']:.4g} :: train {result['train_rmse']:.4g}"
            )
            with open(output_file, "a", newline="") as file:
                csv.writer(file).writerow(
                    [result[column] for column in COMPOSITE_SIZE_HEADER]
                )
    return results


HF_SWEEP_HEADER = [
    "points_lf",
    "points_hf",
    "hf_sample_seed",
    "seed",
    "test_rmse",
    "train_rmse",
    "training_time",
]


# Trains the composite network of MultiFidelityNN at each HF sample count, for
# each HF sampling seed and each seed for the composite's initial weights. The LF
# model is trained once by the caller and shared, so runs at one count differ
# only in where the HF samples fall and in the composite's starting weights.
def generate_mf_nn_hf_sweep(
    lf_model,
    func: mfs.MultiFidelityTemplate,
    y_lf,
    X_test,
    y_test,
    hf_points=range(20, 201, 20),
    seeds=3,
    hf_sample_seeds=(0,),
    hf_nl_nn_dim=[16, 16, 16],
    epochs_mf=30000,
    lr_mf=1e-4,
    wd_mf=1e-6,
    n_workers=8,
    output_file="mf_nn_hf_sweep_output.csv",
):
    # _fit_composite builds every hidden layer at one width
    if len(set(hf_nl_nn_dim)) != 1:
        raise ValueError("hf_nl_nn_dim must use the same width for every layer")
    depth, width = len(hf_nl_nn_dim), hf_nl_nn_dim[0]

    x_mf_test = np.hstack((X_test, lf_model.evaluate(X_test)))

    # Same input scaling as MultiFidelityNN
    mf_bounds = list(func.bounds) + [[float(np.min(y_lf)), float(np.max(y_lf))]]

    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as file:
            csv.writer(file).writerow(HF_SWEEP_HEADER)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_use_one_cpu_thread
    ) as executor:
        runs = {}
        for hf_sample_seed in hf_sample_seeds:
            for n in hf_points:
                X_hf = func.sample_random(n, seed=hf_sample_seed)
                y_hf = func.high_fidelity(X_hf)
                x_mf = np.hstack((X_hf, lf_model.evaluate(X_hf)))
                for seed in range(seeds):
                    run = executor.submit(
                        _fit_composite,
                        x_mf,
                        y_hf,
                        x_mf_test,
                        y_test,
                        mf_bounds,
                        depth,
                        width,
                        seed,
                        epochs_mf,
                        lr_mf,
                        wd_mf,
                    )
                    runs[run] = (n, hf_sample_seed, seed)

        # Each run appends its own row immediately, so a crash partway through a
        # sweep still leaves completed runs on disk
        for run in concurrent.futures.as_completed(runs):
            n, hf_sample_seed, seed = runs[run]
            result = dict(
                run.result(),
                points_lf=len(y_lf),
                points_hf=n,
                hf_sample_seed=hf_sample_seed,
                seed=seed,
            )
            results.append(result)
            print(
                f"{len(results)}/{len(runs)} :: hf points {n} :: sampling seed {hf_sample_seed} :: seed {seed} :: "
                f"test {result['test_rmse']:.4g} :: train {result['train_rmse']:.4g}"
            )
            with open(output_file, "a", newline="") as file:
                csv.writer(file).writerow(
                    [result[column] for column in HF_SWEEP_HEADER]
                )
    return results
