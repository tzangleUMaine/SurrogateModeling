import numpy as np
import matplotlib.pyplot as plt
import model
import scipy.optimize
import scipy.linalg
import typing
import time


class Kriging(model.Model):
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        nugget: float = 1e-5,
        scale_input=False,
        bounds=None,
    ):
        # Set a timer to track training time
        t0 = time.perf_counter()

        super().__init__(x_mat, y_vec, bounds)
        self.scale_input = scale_input
        if scale_input:
            self.x = self.x_scale.transform(self.x)

        # Generate a matrix of theta elements of the same size as the dimensions
        self.diffs = np.abs(self.x[:, None, :] - self.x[None, :, :])

        self.nugget = nugget

        self.theta, self.p = self.calculate_hyper_params()

        self.mu, self.sigma, self.b, self.psi = self._calculate_kriging_constants(
            self.theta, self.p
        )

        self.training_time = time.perf_counter() - t0

    def calculate_hyper_params(self):
        # fun = lambda vars: (-self.calculate_mle(vars[: self.d], vars[self.d :]))
        assume_gaussian = True
        if assume_gaussian:
            x0 = np.zeros(self.d, dtype=float)
            bounds = [(-3, 3)] * self.d

            logger = MinimizeLogger(self._objective_function2)
            res, history = logger.run(x0, method="L-BFGS-B", bounds=bounds)
            self.iter = logger.iter

            theta_opt = 10 ** res.x[: self.d]
            p_opt = np.ones(self.d, dtype=float) * 2.0
        else:
            x0 = np.zeros(self.d * 2, dtype=float)
            x0[self.d :] = 2
            bounds = [(-3, 3)] * self.d + [(1, 2)] * self.d

            logger = MinimizeLogger(self._objective_function)
            res, history = logger.run(x0, method="L-BFGS-B", bounds=bounds)
            self.iter = logger.iter

            theta_opt = 10 ** res.x[: self.d]
            p_opt = res.x[self.d :]
        return theta_opt, p_opt

    def _objective_function(self, vars):
        beta = vars[: self.d]
        p = vars[self.d :]

        # optimize theta as log10(theta)
        theta = 10**beta
        return -self.calculate_mle(theta, p)

    def _objective_function2(self, vars):
        beta = vars[: self.d]
        p = np.ones(self.d, dtype=float) * 2

        # optimize theta as log10(theta)
        theta = 10**beta
        return -self.calculate_mle(theta, p)

    def calculate_mle(self, theta, p):

        psi = self._calculate_correlation(theta, p, self.diffs)
        psi += np.eye(self.n) * self.nugget

        try:
            L = np.linalg.cholesky(psi)
        except np.linalg.LinAlgError:
            print("Error in Cholesky Decomp")
            return -1e20  # invalid region for optimizer

        ones = np.ones((self.n, 1))

        # solve Ψ⁻¹y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        # solve Ψ⁻¹1
        beta = np.linalg.solve(L.T, np.linalg.solve(L, ones))

        mu = (ones.T @ alpha) / (ones.T @ beta)
        r = self.y - ones * mu
        gamma = np.linalg.solve(L.T, np.linalg.solve(L, r))
        sigma2 = (r.T @ gamma) / self.n
        logdet = 2 * np.sum(np.log(np.diag(L)))

        mle = -0.5 * (self.n * np.log(sigma2) + logdet)
        return mle.item()

    def _calculate_kriging_constants(self, theta, p):

        psi = self._calculate_correlation(theta, p, self.diffs)
        psi += np.eye(self.n) * self.nugget

        try:
            L = np.linalg.cholesky(psi)
        except np.linalg.LinAlgError:
            print("Error in Cholesky Decomp")
            return -1e20  # invalid region for optimizer

        ones = np.ones((self.n, 1))

        # solve Ψ⁻¹y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        # solve Ψ⁻¹1
        beta = np.linalg.solve(L.T, np.linalg.solve(L, ones))

        mu = (ones.T @ alpha) / (ones.T @ beta)
        r = self.y - ones * mu
        b = np.linalg.solve(L.T, np.linalg.solve(L, r))
        sigma2 = (r.T @ b) / self.n

        return mu, sigma2, b, psi

    def _calculate_correlation(self, theta, p, diffs):

        # Use the power vector for each of the difference matrix elements
        diffs_powered = diffs ** p[np.newaxis, np.newaxis, :]

        # Multiply each element by the theta vector
        psi = np.tensordot(diffs_powered, theta, axes=([2], [0]))

        # Take the negative exponent of the psi matrix
        psi = np.exp(-psi)

        return psi

    def evaluate(self, x_new, scale=False):
        # Scale x input if necessary
        if self.scale_input:
            x_new = self.x_scale.transform(x_new)

        diff_x_new = np.abs(x_new[:, None, :] - self.x[None, :, :])

        psi_new = self._calculate_correlation(self.theta, self.p, diff_x_new)
        y_hat = psi_new @ self.b + self.mu

        return y_hat


class DiffKriging(Kriging):

    def __init__(self, x_mat, yh, yl, nugget=1e-7):

        self.x = x_mat

        # Store the size of the inputs
        self.n, self.d = np.shape(self.x)

        self.diffs = np.abs(self.x[:, None, :] - self.x[None, :, :])

        self.nugget = nugget

        # Set the high fidelity and low fidelity data
        self.yh = yh
        self.yl = yl

        self.theta, self.p, self.scale_factor = self.calculate_hyper_params()

        self.mu, self.sigma, self.b, self.psi = self._calculate_kriging_constants(
            self.theta, self.p
        )

    def calculate_hyper_params(self):

        # Set initial guess to 1 for all variables
        x0 = np.ones(self.d * 2 + 1, dtype=float)
        x0[self.d : -1] = 2

        # Set Bounds
        bounds = [(-3, 3)] * self.d  # Bounds for theta values
        bounds = bounds + [(1, 2)] * self.d  # Bounds for p values
        bounds = bounds + [(1e-3, None)]  # Bounds for scale factor

        # Run the optimizer for the lbfgs
        logger = MinimizeLogger(self._objective_function)
        result, _ = logger.run(x0, method="L-BFGS-B", bounds=bounds)

        theta_opt = 10 ** result.x[: self.d]
        p_opt = result.x[self.d : -1]
        scale_factor_opt = result.x[-1]
        return theta_opt, p_opt, scale_factor_opt

    def _objective_function(self, vars):
        beta = vars[: self.d]
        p = vars[self.d : -1]
        scale_factor = vars[-1]

        # optimize theta as log10(theta)
        theta = 10**beta
        return -self.calculate_mle(theta, p, scale_factor)

    def calculate_mle(self, theta, p, scale_factor):

        self.y = self.yh - scale_factor * self.yl

        return Kriging.calculate_mle(self, theta, p)


class MinimizeLogger:
    def __init__(self, func, grad=None):
        self.func = func
        self.grad = grad
        self.history = []
        self.iter = 0

    def wrapped_func(self, x):
        f = self.func(x)
        return f

    def wrapped_grad(self, x):
        if self.grad is None:
            return None
        return self.grad(x)

    def callback(self, xk):
        record = {"iter": self.iter, "x": xk.copy(), "f": self.func(xk)}

        if self.grad is not None:
            record["grad"] = self.grad(xk)

        self.history.append(record)
        self.iter += 1

    def run(self, x0, **minimize_kwargs):
        result = scipy.optimize.minimize(
            self.wrapped_func,
            x0,
            jac=self.wrapped_grad if self.grad else None,
            callback=self.callback,
            **minimize_kwargs
        )

        return result, self.history
