import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import mf_model
import kriging


class CoKriging(mf_model.MFModel):
    def __init__(
        self,
        x_lf,
        y_lf,
        x_hf,
        y_hf,
        nugget: float = 1e-3,
        scale_input=False,
        bounds=None,
    ):

        # Init the multifidelity model data
        super().__init__(x_lf, y_lf, x_hf, y_hf, bounds=bounds)

        self.nugget = nugget

        self.scale_input = scale_input
        if self.scale_input:
            self.x = self.x_scale.transform(self.x)
            self.x_lf = self.x_scale.transform(self.x_lf)

        # Store the shapes of the input matrices
        self.n_hf, self.d = np.shape(self.x)
        self.n_lf, __ = np.shape(self.x_lf)
        self.n = self.n_lf + self.n_hf

        # Set up the low fidelity kriging model
        self.kriging_lf = kriging.Kriging(self.x_lf, self.y_lf, nugget=self.nugget)

        # Match find values for the low fidelity data set at exactly the same
        # locations as the high fidelity data
        y_lf_hat = self.kriging_lf.evaluate(self.x)

        # Set up kriging model of the difference between the low and
        # high fidelity data set
        self.kriging_diff = kriging.DiffKriging(
            self.x, self.y, y_lf_hat, nugget=self.nugget
        )

        # Assemble the combined y vector of cheap and expensive values
        self.yck = np.vstack((self.y_lf, self.y))

        self.C, self.mean, self.sigma2, self.b = self._calculate_constants()

    def _calculate_constants(self):
        C = self._calculate_covariance()
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -1e20  # invalid region for optimizer

        ones = np.ones((self.n, 1))

        # solve Ψ⁻¹y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.yck))

        # solve Ψ⁻¹1
        beta = np.linalg.solve(L.T, np.linalg.solve(L, ones))

        mu = (ones.T @ alpha) / (ones.T @ beta)
        r = self.yck - ones * mu
        b = np.linalg.solve(L.T, np.linalg.solve(L, r))
        sigma2 = (r.T @ b) / self.n

        return C, mu, sigma2, b

    # Function calculates the covariance matrix for cokriging using kriging constants solved
    # for in the low fidelity and difference models
    def _calculate_covariance(self):

        # psi_c(X_e, X_c)
        psic_ec = self.kriging_lf._calculate_correlation(
            self.kriging_lf.theta, self.kriging_lf.p, self.x, self.x_lf
        )
        # psi_c(X_c, X_e)
        psic_ce = self.kriging_lf._calculate_correlation(
            self.kriging_lf.theta, self.kriging_lf.p, self.x_lf, self.x
        )
        # psi_c(X_e, X_e)
        psic_e = self.kriging_lf._calculate_correlation(
            self.kriging_lf.theta, self.kriging_lf.p, self.x, self.x
        )

        n = self.n_hf + self.n_lf
        C_mat = np.zeros((n, n))

        # set n_c x n_c in top left
        C_mat[0 : self.n_lf, 0 : self.n_lf] = (
            self.kriging_lf.sigma * self.kriging_lf.psi
        )

        # Set n_c x n_e in top right
        C_mat[0 : self.n_lf, self.n_lf : n] = (
            self.kriging_diff.scale_factor * self.kriging_lf.sigma * psic_ce
        )

        # Set n_e x n_c in bottom left
        C_mat[self.n_lf : n, 0 : self.n_lf] = (
            self.kriging_diff.scale_factor * self.kriging_lf.sigma * psic_ec
        )

        # Set n_e x n_e in bottom right of matrix
        C_mat[self.n_lf : n, self.n_lf : n] = (
            self.kriging_diff.scale_factor**2 * self.kriging_lf.sigma * psic_e
            + self.kriging_diff.sigma * self.kriging_diff.psi
        )

        return C_mat

    def evaluate(self, x_new):

        # Scale the input if a scale was used
        if self.scale_input:
            x_new = self.x_scale.transform(x_new)

        # Solve for correlation values of evaluation points
        psic_c_new = self.kriging_lf._calculate_correlation(
            self.kriging_lf.theta, self.kriging_lf.p, self.x_lf, x_new
        )

        psic_e_new = self.kriging_lf._calculate_correlation(
            self.kriging_lf.theta, self.kriging_lf.p, self.x, x_new
        )

        psid_e_new = self.kriging_diff._calculate_correlation(
            self.kriging_diff.theta, self.kriging_diff.p, self.x, x_new
        )

        # Construct the covariance matrix for evaluation points
        c_top = self.kriging_diff.scale_factor * self.kriging_lf.sigma * psic_c_new
        c_bot = (
            (self.kriging_diff.scale_factor**2) * self.kriging_lf.sigma * psic_e_new
            + self.kriging_diff.sigma * psid_e_new
        )
        c_vec = np.vstack((c_top, c_bot))

        # Generate evaluation points
        F = np.ones((self.n, 1))
        y_pred = self.mean + np.transpose(c_vec) @ self.b
        return y_pred
