import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(
        self,
        x_mat,
        y_vec,
        bounds=None,
        x_scale_method="bounded",
        y_scale_method="standard",
        scale_input=False,
        scale_output=False,
    ):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        self.x = x_mat
        self.y = y_vec
        self.n, self.d = np.shape(self.x)
        self.x_scale = DataTransform(self.x, bounds, method=x_scale_method)
        self.y_scale = DataTransform(self.y, method=y_scale_method)

        if scale_input:
            self.x = self.x_scale.transform(self.x)
        if scale_output:
            self.y = self.y_scale.transform(self.y)

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
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("f(x)")

            ax.plot_surface(
                X1, X2, y_hat, color=color, label=label, linestyle=linestyle
            )
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
            error = error.reshape(n_samples, n_samples)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")

            ax.set_title(name)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("Error")

            ax.plot_surface(
                X1, X2, error, color=color, label=label, linestyle=linestyle
            )
            return ax

    def plot_testpoint_error(
        self,
        X_test,
        y_test,
        name="Error",
        color="black",
        marker="o",
        label="model",
        ax=None,
    ):
        """Plots the absolute error beteween the model and a set of test points

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

            # Generate the sample point locations and evaluate the model
            y_hat = self.evaluate(X_test)
            error = np.abs(y_hat - y_test)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()

            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("Error")

            ax.scatter(X_test, error, color=color, label=label, marker=marker)
            return ax
        else:
            print("this isn't working yet for the bigger dimensions, sorry!")
            return ax

    def model_error(self, x_test, y_test):
        y_hat = self.evaluate(x_test)
        error = (y_hat - y_test) ** 2
        error = (error.sum(axis=0) / x_test.shape[0])[0]
        return np.sqrt(error)

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
            y_truth = fxn_truth(x_sample)
            return self.model_error(x_sample, y_truth)

    @staticmethod
    def generate_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
        x = np.linspace(xmin, xmax, num)
        y = np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def generate_grid_array(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
        x = np.linspace(xmin, xmax, num, dtype=np.float32)
        y = np.linspace(ymin, ymax, num, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        grid = np.column_stack([X.ravel(), Y.ravel()])
        return grid


# Simple class to deal with function scaling
# Probably will need a rewrite but forces users to use a specified data transform type
class DataTransform:
    def __init__(self, x=None, bounds=None, method="bounded"):
        self.method = method
        if method == "bounded":
            if bounds is None:
                self.bounds = np.vstack((np.min(x, axis=0), np.max(x, axis=0)))
                if x.shape[1] == 1:
                    self.bounds = self.bounds.T
            else:
                self.bounds = np.array(bounds, dtype=float)
            self.transform = self.bounded_transform
            self.inverse = self.bounded_inverse
        elif method == "standard":
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
            self.transform = self.standard_transform
            self.inverse = self.standard_inverse
        else:
            raise ValueError(f"Unknown DataTransform method: {method}")

    # For transforms with known bounds -- scales into [-1, 1]
    def bounded_transform(self, x):
        return 2 * (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0]) - 1

    def bounded_inverse(self, x_inv):
        return (x_inv + 1) / 2 * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    # Z-score transform: distance from the mean in units of standard deviation
    def standard_transform(self, x):
        return (x - self.mean) / self.std

    def standard_inverse(self, x_inv):
        return x_inv * self.std + self.mean
