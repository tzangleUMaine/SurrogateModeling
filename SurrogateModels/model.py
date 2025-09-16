import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, x_mat, y_vec):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        self.x = x_mat
        self.y = y_vec
        self.n, self.d = np.shape(self.x)

    def evaluate(self, x_in):
        raise NotImplementedError()

    def plot_data(self, name="Data", color="black", label="model", marker="o"):
        if self.d == 1:
            plt.figure(name)
            plt.title(name)
            plt.scatter(self.x, self.y, color=color, label=label, marker=marker)
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y(x)")
        if self.d == 2:
            plt.figure(name, projection="3d")
            plt.title(name)
            plt.scatter(self.x[:, 0], self.x[:, 1], self.y, color=color, label=label)
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y(x)")

    def plot_model(
        self, x_vals, name="Model", color="black", label="model", linestyle="solid"
    ):
        if self.d == 1:
            y_hat = np.zeros(len(self.n))
            for i, x in enumerate(x_vals):
                y_hat[i] = self.evaluate(x)
            plt.figure(name)
            plt.title(name)
            plt.plot(x_vals, y_hat, color=color, label=label, linestyle=linestyle)
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y(x)")
        if self.d == 2:
            y_hat = np.zeros(self.n)
            for i, x in enumerate(x_vals):
                y_hat[i] = self.evaluate(x)
            plt.figure(name, projection="3d")
            plt.title(name)
            plt.scatter(
                x_vals[:, 0],
                x_vals[:, 1],
                y_hat,
                color=color,
                label=label,
                linestyle=linestyle,
            )
            plt.legend()
            plt.xlabel("x0")
            plt.ylabel("x1")
            plt.zlabel("y")
        return

    def rms_error(self, model, x_vals):
        y_hat1 = np.zeros(len(x_vals[:, 0]))
        y_hat2 = np.zeros(len(x_vals[:, 0]))
        for i, x in enumerate(x_vals):
            y_hat1[i] = self.evaluate(x)
            y_hat2[i] = model.evaluate(x)
        total_error = 0
        for val1, val2 in zip(y_hat1, y_hat2):
            diff = val1 - val2
            total_error += diff**2
        n = np.size(x_vals, 0)
        return np.sqrt(total_error / n)
