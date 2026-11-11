import csv
import os
import time

import numpy as np

import examples.multi_fidelity_functions as mfs
from SurrogateModels import cokriging, kriging
from SurrogateModels.model import DataTransform


HF_SWEEP_HEADER = [
    "points_lf",
    "points_hf",
    "hf_sample_seed",
    "model",
    "test_rmse",
    "train_rmse",
    "training_time",
]


# Fits the LF Kriging model that the sweep shares. With scale_input=True,
# CoKriging scales x before calling its LF model, so the LF model has to be fit
# on x that is already scaled by the same bounds, with scale_input=False.
def fit_lf_kriging(func: mfs.MultiFidelityTemplate, X_lf, y_lf, nugget=1e-3):
    x_scale = DataTransform(bounds=func.bounds)
    return kriging.Kriging(x_scale.transform(X_lf), y_lf, nugget=nugget)


# Fits CoKriging (once per mean option) and an HF-only Kriging model at each HF
# sample count, for each HF sampling seed. The models are deterministic for
# given samples, so the sampling seed is the only source of spread. The LF model
# from fit_lf_kriging is shared, so CoKriging's training_time here covers only
# the difference model and the combined covariance.
def generate_cokriging_hf_sweep(
    lf_kriging,
    func: mfs.MultiFidelityTemplate,
    X_lf,
    y_lf,
    X_test,
    y_test,
    hf_points=range(20, 201, 20),
    hf_sample_seeds=(0,),
    nugget=1e-3,
    means=("separate", "shared"),
    output_file="cokriging_hf_sweep_output.csv",
):
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as file:
            csv.writer(file).writerow(HF_SWEEP_HEADER)

    results = []
    for hf_sample_seed in hf_sample_seeds:
        for n in hf_points:
            X_hf = func.sample_random(n, seed=hf_sample_seed)
            y_hf = func.high_fidelity(X_hf)

            fits = []
            for mean in means:
                t0 = time.perf_counter()
                ck_model = cokriging.CoKriging(
                    X_lf,
                    y_lf,
                    X_hf,
                    y_hf,
                    nugget=nugget,
                    scale_input=True,
                    bounds=func.bounds,
                    lf_kriging_model=lf_kriging,
                    mean=mean,
                )
                fits.append((f"cokriging_{mean}", ck_model, time.perf_counter() - t0))

            t0 = time.perf_counter()
            hf_model = kriging.Kriging(
                X_hf, y_hf, nugget=nugget, scale_input=True, bounds=func.bounds
            )
            fits.append(("hf_only", hf_model, time.perf_counter() - t0))

            for name, fit, training_time in fits:
                result = {
                    "points_lf": len(y_lf),
                    "points_hf": n,
                    "hf_sample_seed": hf_sample_seed,
                    "model": name,
                    "test_rmse": float(fit.model_error(X_test, y_test)),
                    "train_rmse": float(fit.model_error(X_hf, y_hf)),
                    "training_time": training_time,
                }
                results.append(result)

                # Each run appends its own row immediately, so a crash partway
                # through a sweep still leaves completed runs on disk
                with open(output_file, "a", newline="") as file:
                    csv.writer(file).writerow(
                        [result[column] for column in HF_SWEEP_HEADER]
                    )
            tests = " :: ".join(
                f"{r['model']} test {r['test_rmse']:.4g}" for r in results[-len(fits):]
            )
            print(f"hf points {n} :: sampling seed {hf_sample_seed} :: {tests}")
    return results


# CoKriging counterpart of generate_mf_nn_model_comparisons in
# multi_fidelity_nn_helpers.py: the same LF and HF sample locations per seed,
# the same return values, and the same CSV rows.
def generate_cokriging_model_comparisons(
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
    nugget=1e-3,
    mean="separate",
    output_file="cokriging_model_comparison_output.csv",
):
    # Start fresh lists per call; pass existing lists to extend them instead
    points = [] if points is None else points
    model_error = [] if model_error is None else model_error
    training_time = [] if training_time is None else training_time
    evaluation_time = [] if evaluation_time is None else evaluation_time

    # Each run appends its own row immediately, so a crash partway through a
    # sweep still leaves completed runs on disk. There is no header, because
    # load_multi_fidelity_csv_results in data_viewer.ipynb doesn't expect one.
    n_points = list(range(sample_low, sample_high, step_size))
    for seed in range(0, seeds):
        X_lf = func.sample_random(n_lf, seed=seed)
        y_lf = func.low_fidelity(X_lf)

        # The LF data doesn't change with the HF sample count, so fit the LF
        # model once per seed and share it across this seed's runs
        lf_kriging = fit_lf_kriging(func, X_lf, y_lf, nugget=nugget)

        for n in n_points:
            X_hf = func.sample_random(n, seed=seed)
            y_hf = func.high_fidelity(X_hf)

            model = cokriging.CoKriging(
                X_lf,
                y_lf,
                X_hf,
                y_hf,
                nugget=nugget,
                scale_input=True,
                bounds=func.bounds,
                lf_kriging_model=lf_kriging,
                mean=mean,
            )

            # Includes the shared LF model's fit, since each model depends on it
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
                csv.writer(file).writerow(
                    [n_lf, points[-1], model_error[-1], training_time[-1], evaluation_time[-1]]
                )
    return points, model_error, evaluation_time, training_time
