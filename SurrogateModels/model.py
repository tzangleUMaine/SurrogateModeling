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

    def plot_data(
        self,
        name="Data",
        color="black",
        label="model",
        marker="o",
        ax: plt.subplot = None,
    ):
        if self.d == 1:
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.scatter(self.x, self.y, color=color, label=label, marker=marker)
            ax.legend()
        if self.d == 2:
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")
            ax.set_title(name)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y(x)")

            ax.scatter(
                self.x[:, 0], self.x[:, 1], self.y[:, 0], color=color, label=label
            )
            ax.legend()
        return ax

    def plot_model(
        self,
        bounds,
        name="Model",
        color="black",
        label="model",
        linestyle="solid",
        ax=None,
        n_samples=100,
    ):
        if self.d == 1:
            # Read in bounds
            x_min = bounds[0][0]
            x_max = bounds[0][1]

            # Generate the sample point locations and evaluate the model
            x_test = np.linspace(x_min, x_max, n_samples, dtype=np.float32)
            x_test = x_test.reshape(-1, 1)
            y_hat = self.evaluate(x_test)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()

            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y(x)")

            ax.plot(x_test, y_hat, color=color, label=label, linestyle=linestyle)
            ax.legend()
            return ax
        elif self.d == 2:
            # Read in bounds
            x1_min = bounds[0][0]
            x1_max = bounds[0][1]
            x2_min = bounds[1][0]
            x2_max = bounds[1][1]

            # Generate the sample point locations and meshgrid then evaluate the model
            X1, X2 = Model.generate_grid(x1_min, x1_max, x2_min, x2_max, n_samples)

            x_sample = np.column_stack([X1.ravel(), X2.ravel()])
            y_hat = self.evaluate(x_sample)
            y_hat = y_hat.reshape(n_samples, n_samples)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")

            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y(x)")

            ax.plot(X1, X2, y_hat, color=color, label=label, linestyle=linestyle)
            ax.legend()
            return ax

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

    @staticmethod
    def generate_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
        x = np.linspace(xmin, xmax, num)
        y = np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def generate_grid_array(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
        x = np.linspace(xmin, xmax, num)
        y = np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        grid = np.column_stack([X.ravel(), Y.ravel()])
        return grid
