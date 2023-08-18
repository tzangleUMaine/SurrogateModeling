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

class Kriging:
    def __init__(self,x_mat,y_vec):
        # Expects input values of:
        # x_mat: matrix of explanatory variables (LOCKED TO 1D currently out of laziness)
        # y_vec: vector of response variables
        self.x = x_mat
        self.y = y_vec

        self.d,self.n = self.x.shape

        #Generate the correlation matrix for the given x values
        self.corr_mat = np.zeros(self.n,self.n)
        for i in range(1,self.n):
            for j in range(i,self.n):
                val = self.Gaussian(self.x[i],self.x[j])
                self.corr_mat[i,j] = self.corr_mat[j,i] = val
        self.mu = mu
        self.theta = np.zeros(self.d)

    def Gaussian(self,x0,x1):
        #probably could vectorize this
        np.power(np.exp(-np.abs(x0-x1)),2)
        return

if __name__ == "__main__":
    main()