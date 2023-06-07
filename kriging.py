import numpy as np
import matplotlib.pyplot as plt

def main():
    # Start with a synthetic dataset
    rng = np.random.RandomState(0)
    data = np.linspace(0,30,num=1_000).reshape(-1,1)
    target = np.sin(data).ravel()
    
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