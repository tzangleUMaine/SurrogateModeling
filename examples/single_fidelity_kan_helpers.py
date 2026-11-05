import os
import time
import csv

import matplotlib.pyplot as plt
import numpy as np

import examples.multi_fidelity_functions as mfs
from SurrogateModels import mfkans


# Absolute, not relative -- a relative default here would resolve against
# whatever the *kernel's* cwd happens to be, which is the notebook's own
# directory (e.g. examples/Comparisons/) under both nbconvert and VSCode's
# interactive Jupyter, not the project root. That silently scattered plots
# into examples/Comparisons/examples/figures/kan_comparisons/ for an entire
# sweep run before anyone noticed, since save_plot never raises either way.
_DEFAULT_PLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "kan_comparisons"
)


def save_plot(ax, name, directory=_DEFAULT_PLOT_DIR):
    # Writes the figure to disk immediately, so it survives a kernel
    # restart or an unsaved notebook -- unlike a plot left only in the
    # notebook's own (easily-lost) cell output.
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def convergence_report(train_losses, test_losses, degradation_warn=1.5, gap_ratio_warn=5.0):
    """Rough convergence-quality check from a model's (train_losses, test_losses) history.

    `gap_ratio` (final_test / final_train) is reported but is NOT the primary
    signal -- it blows up any time train loss approaches zero regardless of
    whether test loss is actually bad, which makes it misleading on its own
    (verified against two real 4D KAN runs: an unregularized model with much
    lower test error than a regularized one still scored a *worse* gap_ratio,
    10.8x vs 4.3x, purely because its train loss was smaller).

    `degradation_ratio` (final_test / best_test_ever_seen) is the primary
    signal: it asks whether training continued past the point of best
    generalization -- exactly the pattern seen when a KAN grid refinement
    adds more capacity than the data supports (test error plateaus, then
    rises after refinement and never recovers). A value near 1.0 means
    training ended at (or near) its best generalization point; well above 1
    means it ran past that point and got worse without recovering.
    """
    train = np.asarray(train_losses, dtype=float)
    test = np.asarray(test_losses, dtype=float)

    final_train = float(train[-1])
    final_test = float(test[-1])
    best_test = float(test.min())
    best_test_epoch_fraction = float((test.argmin() + 1) / len(test))

    gap_ratio = final_test / final_train if final_train > 0 else float("inf")
    degradation_ratio = final_test / best_test if best_test > 0 else float("inf")

    if degradation_ratio > degradation_warn:
        verdict = "overfit"
    elif gap_ratio > gap_ratio_warn:
        verdict = "watch"
    else:
        verdict = "ok"

    return {
        "final_train": final_train,
        "final_test": final_test,
        "gap_ratio": gap_ratio,
        "best_test": best_test,
        "best_test_epoch_fraction": best_test_epoch_fraction,
        "degradation_ratio": degradation_ratio,
        "verdict": verdict,
    }


def print_convergence_report(name, train_losses, test_losses, **kwargs):
    r = convergence_report(train_losses, test_losses, **kwargs)
    print(
        f"{name}: train={r['final_train']:.4g} test={r['final_test']:.4g} "
        f"gap={r['gap_ratio']:.2f}x | best_test={r['best_test']:.4g} "
        f"(at {r['best_test_epoch_fraction'] * 100:.0f}% of training) "
        f"degradation={r['degradation_ratio']:.2f}x -> {r['verdict'].upper()}"
    )
    return r


def create_test_points(func: mfs.MultiFidelityTemplate, n=2000):
    # Create a set of samples to create a ground truth
    # KANModel does not cast its inputs, and the underlying kan.KAN network
    # runs in float32, so cast here rather than relying on every caller to.
    X_test = func.sample_random(n).astype(np.float32)
    y_test = func.high_fidelity(X_test).astype(np.float32)
    return X_test, y_test


def generate_kan_model(
    name,
    func: mfs.MultiFidelityTemplate,
    n,
    seed,
    X_test,
    y_test,
    width=[1, 2, 1],
    grid=[3],
    spline_order=3,
    opt="LBFGS",
    n_iter=100,
    lr=1.0,
    scale_input=True,
    scale_output=True,
    x_scale_method="bounded",
    y_scale_method="standard",
    bounds=None,
    lamb=0.0,
):
    X_train = func.sample_random(n, seed=seed).astype(np.float32)
    y_train = func.high_fidelity(X_train).astype(np.float32)

    # Setup timer to gauge model training time
    t0 = time.perf_counter()
    model = mfkans.KANModel(
        x_mat=X_train,
        y_vec=y_train,
        x_mat_test=X_test,
        y_vec_test=y_test,
        spline_order=spline_order,
        width=width,
        grid=grid,
        opt=opt,
        n_iter=n_iter,
        lr=lr,
        scale_input=scale_input,
        scale_output=scale_output,
        x_scale_method=x_scale_method,
        y_scale_method=y_scale_method,
        # Scale input by the function's known domain bounds by default,
        # rather than the (possibly unrepresentative) sampled min/max.
        bounds=bounds if bounds is not None else func.bounds,
        lamb=lamb,
    )
    model_training_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    error = float(model.model_error(X_test, y_test))
    evaluation_time = time.perf_counter() - t0
    print(
        f"Run {name} :: seed {seed} :: points {n} :: error {error} :: train_time {model_training_time} :: evaluation_time {evaluation_time}"
    )
    return model, error, model_training_time, evaluation_time


def generate_kan_model_comparisons(
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
    width=[1, 2, 1],
    grid=[3],
    spline_order=3,
    opt="LBFGS",
    n_iter=100,
    lr=1.0,
    scale_input=True,
    scale_output=True,
    x_scale_method="bounded",
    y_scale_method="standard",
    bounds=None,
    lamb=0.0,
    output_file="kan_model_comparison_output.csv",
):
    # Start fresh lists per call; pass existing lists to extend them instead
    points = [] if points is None else points
    model_error = [] if model_error is None else model_error
    training_time = [] if training_time is None else training_time
    evaluation_time = [] if evaluation_time is None else evaluation_time

    # Write a header only if the file doesn't already exist, so calling this
    # again (e.g. to add more seeds/points, or after a previous call already
    # populated the file) appends new rows instead of wiping prior results --
    # each run below then appends its own row immediately, in "a" mode, so
    # a crash partway through a sweep still leaves every completed run on
    # disk rather than just the single most recent one.
    if not os.path.isfile(output_file):
        with open(output_file, "w", newline="") as file:
            csv.writer(file).writerow(
                ["points", "model_error", "training_time", "evaluation_time"]
            )

    # Generate a KAN model of the data and test accuracy using a number of random sample points
    n_points = list(range(sample_low, sample_high, step_size))
    for seed in range(0, seeds):
        for n in n_points:
            X_train = func.sample_random(n, seed=seed).astype(np.float32)
            y_train = func.high_fidelity(X_train).astype(np.float32)

            # Setup timer to gauge model training time
            t0 = time.perf_counter()

            model = mfkans.KANModel(
                x_mat=X_train,
                y_vec=y_train,
                x_mat_test=X_test,
                y_vec_test=y_test,
                spline_order=spline_order,
                width=width,
                grid=grid,
                opt=opt,
                n_iter=n_iter,
                lr=lr,
                scale_input=scale_input,
                scale_output=scale_output,
                x_scale_method=x_scale_method,
                y_scale_method=y_scale_method,
                bounds=bounds if bounds is not None else func.bounds,
                lamb=lamb,
            )

            training_time.append(time.perf_counter() - t0)

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
        for i, val in enumerate(points):
            writer.writerow(
                [
                    val,
                    model_error[i],
                    training_time[i],
                    evaluation_time[i],
                ]
            )
