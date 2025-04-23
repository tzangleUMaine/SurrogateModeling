import numpy as np
import matplotlib.pyplot as plt
import model


class MFModel(model.Model):
    def __init__(self, x_lf, y_lf, x_hf, y_hf):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        super().__init__(x_hf, y_hf)
        self.x_lf = x_lf
        self.y_lf = y_lf

    def plot_data(
        self,
        name="Data",
        color_lf="black",
        color_hf="blue",
        label_hf="hf",
        label_lf="lf",
        marker_hf="^",
        marker_lf="s",
    ):
        plt.figure(name)
        plt.title(name)
        plt.scatter(self.x, self.y, color=color_hf, label=label_hf, marker=marker_hf)
        plt.scatter(
            self.x_lf, self.y_lf, color=color_lf, label=label_lf, marker=marker_lf
        )
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")

    def plot_model(
        self, x_vals, name="Model", color="black", label="model", linestyle="solid"
    ):
        y_hat = np.zeros(len(x_vals[:, 0]))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x_vals, y_hat, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")
