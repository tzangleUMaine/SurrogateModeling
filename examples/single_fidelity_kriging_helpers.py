import os
import time
import csv

import matplotlib.pyplot as plt
import numpy as np

import examples.multi_fidelity_functions as mfs
from SurrogateModels import kriging

# Absolute, not relative -- a relative default here would resolve against
# whatever the *kernel's* cwd happens to be (the notebook's own directory
# under both nbconvert and VSCode's interactive Jupyter, not the project
# root), silently scattering plots into the wrong place. See the identical
# fix in single_fidelity_kan_helpers.py for how this actually bit us.
_DEFAULT_PLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "kriging_comparisons"
)


def save_plot(ax, name, directory=_DEFAULT_PLOT_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def create_test_points(func: mfs.MultiFidelityTemplate, n=2000):
    # Unlike KANModel, Kriging works directly with numpy float64 -- no dtype
    # cast needed here.
    X_test = func.sample_random(n)
    y_test = func.high_fidelity(X_test)
    return X_test, y_test


def generate_kriging_model(
    name,
    func: mfs.MultiFidelityTemplate,
    n,
    seed,
    X_test,
    y_test,
    nugget=1e-7,
    scale_input=True,
    bounds=None,
):
    X_train = func.sample_random(n, seed=seed)
    y_train = func.high_fidelity(X_train)

    model = kriging.Kriging(
        x_mat=X_train,
        y_vec=y_train,
        nugget=nugget,
        scale_input=scale_input,
        # Scale input by the function's known domain bounds by default,
        # rather than the (possibly unrepresentative) sampled min/max.
        bounds=bounds if bounds is not None else func.bounds,
    )
    # Kriging.__init__ already times its own fit via time.perf_counter(),
    # stored as model.training_time -- reuse it instead of timing again.
    model_training_time = model.training_time

    t0 = time.perf_counter()
    error = float(model.model_error(X_test, y_test))
    evaluation_time = time.perf_counter() - t0
    print(
        f"Run {name} :: seed {seed} :: points {n} :: error {error} :: train_time {model_training_time} :: evaluation_time {evaluation_time}"
    )
    return model, error, model_training_time, evaluation_time


def generate_kriging_model_comparisons(
    name,
    func: mfs.MultiFidelityTemplate,
    X_test,
    y_test,
    sample_low=5,
    sample_high=105,
    step_size=5,
    seeds=5,
    points=None,
    model_error=None,
    training_time=None,
    evaluation_time=None,
    nugget=1e-5,
    scale_input=True,
    bounds=None,
    output_file="kriging_model_comparison_output.csv",
):
    # Start fresh lists per call; pass existing lists to extend them instead
    points = [] if points is None else points
    model_error = [] if model_error is None else model_error
    training_time = [] if training_time is None else training_time
    evaluation_time = [] if evaluation_time is None else evaluation_time

    # Write a header only if the file doesn't already exist, so calling this
    # again (e.g. to add more seeds/points) appends new rows instead of
    # wiping prior results -- each run below appends its own row immediately,
    # so a crash partway through a sweep still leaves completed runs on disk.
    if not os.path.isfile(output_file):
        with open(output_file, "w", newline="") as file:
            csv.writer(file).writerow(
                ["points", "model_error", "training_time", "evaluation_time"]
            )

    # Generate a Kriging model of the data and test accuracy using a number of random sample points
    n_points = list(range(sample_low, sample_high, step_size))
    for seed in range(0, seeds):
        for n in n_points:
            X_train = func.sample_random(n, seed=seed)
            y_train = func.high_fidelity(X_train)

            model = kriging.Kriging(
                x_mat=X_train,
                y_vec=y_train,
                nugget=nugget,
                scale_input=scale_input,
                bounds=bounds if bounds is not None else func.bounds,
            )

            training_time.append(model.training_time)

            t0 = time.perf_counter()
            error = float(model.model_error(X_test, y_test))
            evaluation_time.append(time.perf_counter() - t0)
            model_error.append(error)
            points.append(n)
            print(
                f"Run {name} :: seed {seed} :: points {n} :: error {error} :: train_time {training_time[-1]} "
            )
            with open(output_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        points[-1],
                        model_error[-1],
                        training_time[-1],
                        evaluation_time[-1],
                    ]
                )
    return points, model_error, evaluation_time, training_time


def plot_error_measure(
    points, error, name, color="blue", marker="o", label="Test", ax=None
):
    if ax == None:
        fig = plt.figure(name)
        ax = fig.add_subplot()
    ax.set_title(name)
    ax.scatter(
        points,
        error,
        color=color,
        label=label,
        marker=marker,
    )
    ax.set_xlabel("Samples")
    ax.set_ylabel("RMS Error")
    ax.legend()
    ax.grid()
    ax.set_yscale("log")
    return ax


def export_to_csv(output_file, points, model_error, training_time, evaluation_time):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["points", "model_error", "training_time", "evaluation_time"])
        for i, val in enumerate(points):
            writer.writerow(
                [
                    val,
                    model_error[i],
                    training_time[i],
                    evaluation_time[i],
                ]
            )
