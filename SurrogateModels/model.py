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
            ax.set_zlabel("f(x)")

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
            ax.set_ylabel("f(x)")

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
            x_sample = Model.generate_grid_array(
                x1_min, x1_max, x2_min, x2_max, n_samples
            )
            y_hat = self.evaluate(x_sample)
            y_hat = y_hat.reshape(n_samples, n_samples)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")

            ax.set_title(name)
            ax.set_xlabel("x_1")
            ax.set_ylabel("x_2")
            ax.set_zlabel("f(x)")

            ax.plot_surface(
                X1, X2, y_hat, color=color, label=label, linestyle=linestyle
            )
            ax.legend()
            return ax

    def plot_rms_error(
        self,
        bounds,
        fxn_truth,
        name="Error",
        color="black",
        label="model",
        linestyle="solid",
        ax=None,
        n_samples=100,
    ):
        """Calculates and plots the error between two functions using the root mean square value.
        Values are taken between a ground truth function and the evaluated model

        Args:
            bounds (_type_): bounds that the model is defined over in n-dimensions
            fxn_truth (_type_): the ground truth function for comparison
            name (str, optional): Graph name and title. Defaults to "Error".
            color (str, optional): Plot color. Defaults to "black".
            label (str, optional): Label for plotted data. Defaults to "model".
            linestyle (str, optional): Linestyle for plot. Defaults to "solid".
            ax (_type_, optional): Active axis for plotting multiple. Defaults to None.
            n_samples (int, optional): number of samples in each direction. Defaults to 100.

        Returns:
            Axis: Active Axis
        """

        if self.d == 1:
            # Read in bounds
            x_min = bounds[0][0]
            x_max = bounds[0][1]

            # Generate the sample point locations and evaluate the model
            x_test = np.linspace(x_min, x_max, n_samples, dtype=np.float32)
            x_test = x_test.reshape(-1, 1)
            y_hat = self.evaluate(x_test)
            y_truth = fxn_truth(x_test)
            error = np.sqrt((y_hat - y_truth) ** 2)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()

            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("Error")

            ax.plot(x_test, error, color=color, label=label, linestyle=linestyle)
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
            x_sample = Model.generate_grid_array(
                x1_min, x1_max, x2_min, x2_max, n_samples
            )
            y_hat = self.evaluate(x_sample)
            y_truth = fxn_truth(x_sample)
            error = np.sqrt((y_hat - y_truth) ** 2)
            error = error * 0
            error = error.reshape(n_samples, n_samples)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")

            ax.set_title(name)
            ax.set_xlabel("x_1")
            ax.set_ylabel("x_2")
            ax.set_zlabel("Error")

            ax.plot_surface(
                X1, X2, error, color=color, label=label, linestyle=linestyle
            )
            ax.legend()
            return ax

    def model_error(self, x_test, y_test):
        y_hat = self.evaluate(x_test)
        error = np.sqrt((y_hat - y_test) ** 2)
        return error / x_test.shape[0]

    def bounded_model_error(self, bounds, fxn_truth, n_samples=100):
        if self.d == 1:
            # Read in bounds
            x_min = bounds[0][0]
            x_max = bounds[0][1]

            # Generate the sample point locations and evaluate the model
            x_test = np.linspace(x_min, x_max, n_samples, dtype=np.float32)
            x_test = x_test.reshape(-1, 1)
            y_truth = fxn_truth(x_test)
            return self.model_error(x_test, y_truth)

        elif self.d == 2:
            # Read in bounds
            x1_min = bounds[0][0]
            x1_max = bounds[0][1]
            x2_min = bounds[1][0]
            x2_max = bounds[1][1]

            # Generate the sample point locations and meshgrid then evaluate the model
            x_sample = Model.generate_grid_array(
                x1_min, x1_max, x2_min, x2_max, n_samples
            )
            y_truth = fxn_truth(x_test)
            return self.model_error(x_test, y_truth)

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
