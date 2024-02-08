import numpy as np
import matplotlib.pyplot as plt
import kriging
import mfs_functions

def main():
    # Start with a synthetic dataset
    rng = np.random.RandomState(0)
    fxn_ind = 1
    dim,xmin,xmax=mfs_functions.function_bounds(fxn_ind)
    x_lf = np.linspace(xmin,xmax,num=11)
    y_lf = mfs_functions.cheap_test_function(fxn_ind,x_lf)
    x_hf = np.linspace(xmin,xmax,num=4)
    y_hf = mfs_functions.expensive_test_function(fxn_ind,x_hf)
    model = kriging.Kriging(x_lf,y_lf)
    
    training_sample_indices = rng.choice(np.arange(0, 400), size=40, replace=False)
    training_data = data[training_sample_indices]
    training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(
        len(training_sample_indices)
    )
    
    plt.plot(data, target, label="True signal", linewidth=2)
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    plt.legend()
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title(
        "Illustration of the true generative process and \n"
        "noisy measurements available during training"
    )
    plt.show()

if __name__ == "__main__":
    main()