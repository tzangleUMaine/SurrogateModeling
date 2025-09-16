import numpy as np
import matplotlib.pyplot as plt
import model
import scipy.optimize
import scipy.linalg
import typing


class Kriging(model.Model):
    def __init__(
        self, x_mat: np.ndarray[float], y_vec: np.ndarray[float], nugget: float = 0.0
    ):

        super().__init__(x_mat, y_vec)

        self.nugget = nugget

        self.pre_psi = self.calculate_pre_psi_matrix()

        self.theta = self.calculate_hyper_params()

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi, self.mu)

        self.b = self.calculate_b()

    def calculate_hyper_params(self):
        fun = lambda theta: (-self.calculate_mle(theta))
        theta0 = np.ones((self.d, 1), dtype=float)
        theta0 = theta0 * 40
        print(self.d)
        theta_opt = scipy.optimize.fmin(fun, theta0)
        return theta_opt

    def calculate_pre_psi_matrix(self):
        pre_psi_ij = np.zeros((self.n, self.n), dtype=float)
        for i in range(0, self.n):
            for j in range(0, self.n):
                diffX = self.x[i, :] - self.x[j, :]
                pre_psi_ij[i, j] = np.dot(diffX, diffX)
        return pre_psi_ij

    def calculate_kriging_mean(self, inv_psi):
        F = np.ones((self.n, 1))
        mu = np.transpose(F) @ inv_psi @ self.y / (np.transpose(F) @ inv_psi @ F)
        return mu

    def calculate_sigma(self, inv_psi, mu):
        F = np.ones((self.n, 1))
        sigma = (
            (1.0 / self.n) * np.transpose(self.y - F * mu) @ inv_psi @ (self.y - F * mu)
        )
        return sigma

    def calculate_b(self):
        b = self.inv_psi @ (self.y - np.ones((self.n, 1)) * self.mu)
        return b

    def calculate_psi_matrix(self, theta):
        psi = np.exp(-theta * self.pre_psi)
        # Add a constant term to the matrix diagonal to increase stability
        psi += np.eye(self.n) * 1e-6
        if self.nugget:
            psi = psi + np.ones((self.n, self.n)) * (self.nugget)
        inv_psi = np.linalg.inv(psi)
        return psi, inv_psi

    def calculate_mle(self, theta):
        psi, inv_psi = self.calculate_psi_matrix(theta)
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi, mu_hat)
        mle = -0.5 * (self.n * np.log(sigma) + np.log(np.linalg.det(psi)))
        return mle

    def calculate_mle2(self, theta):
        psi, inv_psi = self.calculate_psi_matrix(theta)
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi, mu_hat)
        mle = -0.5 * (self.n * np.log(sigma) + np.log(np.linalg.det(psi)))
        return mle, sigma, np.linalg.det(psi)

    def evaluate(self, x_new):
        F = np.ones((self.n, 1))
        x_diff = F * x_new - self.x
        y_pred = np.exp(-self.theta * (x_diff * x_diff))
        y_pred = y_pred * self.b
        sum_y = np.sum(y_pred)
        return np.sum(y_pred) + self.mu

    def plot_MLE(self):
        theta = np.linspace(1.0, 20.0, 1000)
        mle = np.linspace(1.0, 20.0, 1000)
        sigma = np.linspace(1.0, 20.0, 1000)
        det = np.linspace(1.0, 20.0, 1000)
        for i, val in enumerate(theta):
            mle[i], sigma[i], det[i] = self.calculate_mle2(val)
        plt.figure("sigma")
        plt.title("sigma")
        plt.plot(theta, sigma)
        plt.xlabel("theta")
        plt.ylabel("sigma")

        plt.figure("det")
        plt.title("det")
        plt.plot(theta, det)
        plt.xlabel("det")
        plt.ylabel("mle")

        plt.figure("mle graph")
        plt.title("mle graph")
        plt.plot(theta, mle)
        plt.xlabel("theta")
        plt.ylabel("mle")

    def print_model_params(self, filename: str):

        np.savetxt(filename + "prepsi", self.pre_psi, delimiter=",")
        np.savetxt(filename + "psi", self.psi, delimiter=",")
        np.savetxt(filename + "inv_psi", self.inv_psi, delimiter=",")


class DiffKriging(Kriging):
    def __init__(self, x_mat, yh, yl, scale_fixed=False, nugget=0.0):
        # Debugging fun for checking stability
        self.scale_fixed = scale_fixed

        self.x = x_mat

        # Store the size of the inputs
        self.n, self.d = np.shape(self.x)

        self.nugget = nugget

        # Initialize hyperparameters
        self.scale_factor = 1.0
        self.theta = np.ones((self.d, 1), dtype=float)

        # Set the high fidelity and low fidelity data
        self.yh = yh
        self.yl = yl

        # Set y vector to be difference of two data streams
        self.y = self.yh - self.scale_factor * self.yl

        self.pre_psi = self.calculate_pre_psi_matrix()

        self.theta, self.scale_factor = self.calculate_hyper_params(
            self.theta, self.scale_factor
        )

        # Override the scale_factor
        # self.scale_factor = 1.87

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi, self.mu)

        self.b = self.calculate_b()

    def fun(self, params):
        theta = params[0:-1]
        scale_factor = params[-1]
        return -self.calculate_mle(theta, scale_factor)

    def calculate_hyper_params(self, theta0, scale_factor0):
        if self.scale_fixed == True:
            fixed_scale_factor = 1.87
            fun1 = lambda theta: (-self.calculate_mle(theta, fixed_scale_factor))
            theta_opt = scipy.optimize.fmin(fun1, theta0)
            return theta_opt, fixed_scale_factor
        init_vars = np.append(theta0, scale_factor0)
        b = (1e-3, None)
        bounds = (b, b)
        result = scipy.optimize.minimize(self.fun, init_vars, bounds=bounds)
        theta_opt = result.x[0:-1]
        scale_factor_opt = result.x[-1]
        return theta_opt, scale_factor_opt

    def calculate_mle(self, theta, scale_factor):
        psi, inv_psi = self.calculate_psi_matrix(theta)
        self.y = self.yh - scale_factor * self.yl
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi, mu_hat)
        mle = -0.5 * (self.n * np.log(sigma) + np.log(np.linalg.det(psi)))
        return mle
