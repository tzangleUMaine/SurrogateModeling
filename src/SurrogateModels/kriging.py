import numpy as np
import matplotlib.pyplot as plt
import model
import scipy.optimize
import scipy.linalg
import typing


class Kriging(model.Model):
    def __init__(
        self, x_mat: np.ndarray[float], y_vec: np.ndarray[float], nugget: float = 1e-6
    ):

        super().__init__(x_mat, y_vec)

        self.nugget = nugget

        self.theta, self.p = self.calculate_hyper_params()

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta, self.p)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi, self.mu)

        self.b = self.calculate_b()

    def calculate_hyper_params(self):
        fun = lambda vars: (-self.calculate_mle(vars[: self.d], vars[self.d :]))
        x0 = np.ones(self.d * 2, dtype=float)
        bounds = [(0, None)] * self.d + [(1, 2)] * self.d
        result = scipy.optimize.minimize(fun, x0, method="L-BFGS-B", bounds=bounds)
        theta_opt = result.x[: self.d]
        p_opt = result.x[self.d :]
        return theta_opt, p_opt

    def calculate_pre_psi_matrix(self):
        pre_psi_ij = np.zeros((self.n, self.n), dtype=float)
        for i in range(0, self.n):
            for j in range(0, self.n):
                diffX = self.x[i, :] - self.x[j, :]
                pre_psi_ij[i, j] = np.dot(diffX, diffX)
        return pre_psi_ij

    def calculate_diffs(self):
        return np.abs(self.x[:, None, :] - self.x[None, :, :])

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

    def calculate_psi_matrix(self, theta, p):
        psi = self._calculate_correlation(theta, p, self.x, self.x)
        # Add a constant term to the matrix diagonal to increase stability
        # nugget defaults to a near zero number
        psi += np.eye(self.n) * self.nugget
        inv_psi = np.linalg.inv(psi)
        return psi, inv_psi

    def _calculate_correlation(self, theta, p, X_1, X_2):
        # Generate a matrix of theta elements of the same size as the dimensions
        diffs = np.abs(X_1[:, None, :] - X_2[None, :, :])

        # Use the power vector for each of the difference matrix elements
        diffs_powered = diffs ** p[np.newaxis, np.newaxis, :]

        # Multiply each element by the theta vector
        psi = np.tensordot(diffs_powered, theta, axes=([2], [0]))

        # Take the negative exponent of the psi matrix
        psi = np.exp(-psi)

        return psi

    def calculate_mle(self, theta, p):
        psi, inv_psi = self.calculate_psi_matrix(theta, p)
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
        psi_new = self._calculate_correlation(self.theta, self.p, x_new, self.x)
        y_hat = psi_new @ self.b + self.mu
        return y_hat

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
    def __init__(self, x_mat, yh, yl, nugget=1e-6):

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

        self.theta, self.p, self.scale_factor = self.calculate_hyper_params()

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta, self.p)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi, self.mu)

        self.b = self.calculate_b()

    def calculate_hyper_params(self):

        # Generate version of MLE compatible with scipy optimizers
        fun = lambda vars: (
            -self.calculate_mle(vars[: self.d], vars[self.d : -1], vars[-1])
        )

        # Set initial guess to 1 for all variables
        x0 = np.ones(self.d * 2 + 1, dtype=float)

        # Set Bounds
        bounds = [(0, None)] * self.d  # Bounds for theta values
        bounds = bounds + [(1, 2)] * self.d  # Bounds for p values
        bounds = bounds + [(1e-3, None)]  # Bounds for scale factor

        # Run the optimizer for the lbfgs
        result = scipy.optimize.minimize(fun, x0, method="L-BFGS-B", bounds=bounds)

        theta_opt = result.x[: self.d]
        p_opt = result.x[self.d : -1]
        scale_factor_opt = result.x[-1]
        return theta_opt, p_opt, scale_factor_opt

    def calculate_mle(self, theta, p, scale_factor):
        psi, inv_psi = self.calculate_psi_matrix(theta, p)
        self.y = self.yh - scale_factor * self.yl
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi, mu_hat)
        mle = -0.5 * (self.n * np.log(sigma) + np.log(np.linalg.det(psi)))
        return mle
