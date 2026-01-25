import numpy as np
import matplotlib.pyplot as plt

import model


class MultiFidelityTemplate:
    def __init__(self):
        self.bounds = None
        self.dimensions = None
        self.name = None

    def low_fidelity(self, x):
        pass

    def high_fidelity(self, x):
        pass

    def get_bounds(self):
        return self.bounds

    def get_dimensions(self):
        return self.dimensions

    def plot_high_fidelity(
        self,
        name="Model",
        color="dodgerblue",
        label="HF Function",
        linestyle="solid",
        ax=None,
        n_samples=100,
    ):
        return self._plot_function(
            self.high_fidelity,
            name=name,
            color=color,
            label=label,
            linestyle=linestyle,
            ax=ax,
            n_samples=n_samples,
        )

    def plot_low_fidelity(
        self,
        name="Model",
        color="orange",
        label="LF Function",
        linestyle="solid",
        ax=None,
        n_samples=100,
    ):
        return self._plot_function(
            self.low_fidelity,
            name=name,
            color=color,
            label=label,
            linestyle=linestyle,
            ax=ax,
            n_samples=n_samples,
        )

    def plot_functions(self, name="Model", linestyle="solid", ax=None):
        ax = self.plot_low_fidelity(name=name, linestyle=linestyle, ax=ax)
        ax = self.plot_high_fidelity(name=name, linestyle=linestyle, ax=ax)
        return ax

    def _plot_function(
        self,
        function,
        name="Model",
        color="black",
        label="function plot",
        linestyle="solid",
        ax=None,
        n_samples=100,
    ):
        if self.dimensions == 1:
            # Read in bounds
            x_min = self.bounds[0][0]
            x_max = self.bounds[0][1]

            # Generate the sample point locations and evaluate the model
            x_test = np.linspace(x_min, x_max, n_samples, dtype=np.float32)
            x_test = x_test.reshape(-1, 1)
            y_test = function(x_test)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()

            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")

            ax.plot(x_test, y_test, color=color, label=label, linestyle=linestyle)
            return ax
        elif self.dimensions == 2:
            # Read in bounds
            x1_min = self.bounds[0][0]
            x1_max = self.bounds[0][1]
            x2_min = self.bounds[1][0]
            x2_max = self.bounds[1][1]

            # Generate the sample point locations and meshgrid then evaluate the model
            X1, X2 = model.Model.generate_grid(
                x1_min, x1_max, x2_min, x2_max, n_samples
            )
            x_sample = model.Model.generate_grid_array(
                x1_min, x1_max, x2_min, x2_max, n_samples
            )
            y_sample = function(x_sample)
            y_sample = y_sample.reshape(n_samples, n_samples)

            # Plot the function
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot(projection="3d")

            ax.set_title(name)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("f(x)")

            ax.plot_surface(
                X1, X2, y_sample, color=color, label=label, linestyle=linestyle
            )
            return ax


class ZhangExample1(MultiFidelityTemplate):
    """Example 1 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    Only includes first low fidelity option

    """

    def __init__(self):
        self.bounds = [[0, 2 * np.pi]]
        self.dimensions = 1
        self.name = "Zhang Example 1"

    def high_fidelity(self, x):
        return np.sin(x)

    def low_fidelity(self, x):
        return np.sin(x) + 0.1 * (x - np.pi) ** 2


class ForresterExample(MultiFidelityTemplate):
    """Example 2 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    Matches the example from the Forrester cokriging paper

    """

    def __init__(self):
        self.bounds = [[0, 1]]
        self.dimensions = 1
        self.name = "Forrester Example"

    def high_fidelity(self, x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    def low_fidelity(self, x):
        return 0.5 * self.high_fidelity(x) + 10 * (x - 0.5) - 5


class ZhangExample3(MultiFidelityTemplate):
    """Example 3 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    """

    def __init__(self):
        self.bounds = [[0, 10]]
        self.dimensions = 1
        self.name = "Zhang Example 3"

    def high_fidelity(self, x):
        return np.sin(x) + 0.2 * x + (x - 5) ** 2 / 16 + 0.5

    def low_fidelity(self, x):
        return (x - 0.5) * (x - 4) * (x - 9) / 20 + 2


class ZhangExample4(MultiFidelityTemplate):
    """Example 4 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    """

    def __init__(self):
        self.bounds = [[0, 10]]
        self.dimensions = 1
        self.name = "Zhang Example 4"

    def high_fidelity(self, x):
        return 2 * np.sin(np.pi * x / 5)

    def low_fidelity(self, x):
        return x * (x - 5) * (x - 12) / 30


class ZhangExample6(MultiFidelityTemplate):
    """Example 6 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    2D input to function
    """

    def __init__(self):
        self.bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        self.dimensions = 2
        self.name = "Zhang Example 6"

    def high_fidelity(self, x):
        y_hf = (
            4 * x[:, 0] ** 2
            - 2.1 * x[:, 0] ** 4
            + (x[:, 0] ** 6) / 3
            + x[:, 0] * x[:, 1]
            - 4 * x[:, 1] ** 2
            + 4 * x[:, 1] ** 4
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf_extra = x[:, 0] * x[:, 1] - 65
        return self.high_fidelity(0.7 * x) + y_lf_extra.reshape(-1, 1)


class ZhangExample7(MultiFidelityTemplate):
    """Example 7 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    2D input to function
    """

    def __init__(self):
        self.bounds = [[0.0, 1.0], [0.0, 1.0]]
        self.dimensions = 2
        self.name = "Zhang Example 7"

    def high_fidelity(self, x):
        y_hf = -np.sin(x[:, 0]) - np.exp(x[:, 1] * 0.01) + 10 + x[:, 1] ** 2 * 0.1
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            -np.sin(x[:, 0])
            - np.exp(x[:, 1] * 0.01)
            + 10.3
            + 0.03 * (x[:, 0] - 0.3) ** 2
            + (x[:, 1] - 1) ** 2 * 0.1
        )
        return y_lf.reshape(-1, 1)


class ZhangExample8(MultiFidelityTemplate):
    """Example 8 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    4D input to function
    """

    def __init__(self):
        self.bounds = [
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
        ]
        self.dimensions = 4
        self.name = "Zhang Example 8"

    def high_fidelity(self, x):
        y_hf = (
            np.sin(5 * np.pi * x[:, 0]) ** 6
            + 0.1 * (x[:, 1] - 0.5) ** 2
            + 0.5 * x[:, 1]
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            0.5 * np.sin(5 * np.pi * x[:, 0]) ** 6
            + 0.05 * (x[:, 1] - 0.5) ** 2
            + 0.25 * x[:, 1]
            + 0.15
        )
        return y_lf.reshape(-1, 1)


class MengDiscontinuous(MultiFidelityTemplate):
    """Example from Meng and Karniadakis 2020 paper
    "A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems"

    Defines the discontinuous example in paper
    """

    def __init__(self):
        self.bounds = [[0, 1]]
        self.dimensions = 1
        self.name = "Meng Discontinuous"

    def high_fidelity(self, x):
        return np.where(
            (x > 0.5),
            4 + 2 * self.low_fidelity(x) - 20 * x + 20,
            2 * self.low_fidelity(x) - 20 * x + 20,
        )

    def low_fidelity(self, x):
        return np.where(
            (x > 0.5),
            3 + 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5,
            0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5,
        )


class MengNonLinearCorrelation(MultiFidelityTemplate):
    """Example from Meng and Karniadakis 2020 paper
    "A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems"

    Defines the continuous function with nonlinear correlation between high and low fidelity
    """

    def __init__(self):
        self.bounds = [[0, 1]]
        self.dimensions = 1
        self.name = "Meng Non-Linear Correlation"

    def high_fidelity(self, x):
        return (x - np.sqrt(2)) * self.low_fidelity(x) ** 2

    def low_fidelity(self, x):
        return np.sin(8 * np.pi * x)


class MengPhaseShift(MultiFidelityTemplate):
    """Example from Meng and Karniadakis 2020 paper
    "A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems"

    Defines the phase-shifted oscillations from the paper
    """

    def __init__(self):
        self.bounds = [[0, 1]]
        self.dimensions = 1
        self.name = "Meng Phase Shifted Oscillations"

    def high_fidelity(self, x):
        return x**2 + np.sin(8 * np.pi * x + np.pi / 10) ** 2

    def low_fidelity(self, x):
        return np.sin(8 * np.pi * x)


class PeaksShifted(MultiFidelityTemplate):
    """Example generated from Peaks function
    Utilizes a non-linear scaling to generate data from peaks function"""

    def __init__(self):
        self.bounds = [[-3.0, 3.0], [-3.0, 3.0]]
        self.dimensions = 2
        self.name = "Peaks Function Shifted"

    def high_fidelity(self, x):
        return 3.0 * self.low_fidelity(0.95 * x) + (
            0.15 * x[:, 0] * x[:, 1] + 12
        ).reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
            - 10
            * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
            * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
            - 1 / 3 * np.exp(-((x[:, 0] + 1) ** 2) - x[:, 1] ** 2)
        )
        return y_lf.reshape(-1, 1)


class PeaksShifted2(MultiFidelityTemplate):
    """Example generated from Peaks function
    Utilizes a non-linear scaling to generate data from peaks function"""

    def __init__(self):
        self.bounds = [[-3.0, 3.0], [-3.0, 3.0]]
        self.dimensions = 2
        self.name = "Peaks Function Shifted"

    def high_fidelity(self, x):
        y_hf = (
            3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
            - 10
            * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
            * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
            - 1 / 3 * np.exp(-((x[:, 0] + 1) ** 2) - x[:, 1] ** 2)
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            2.5 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
            - 6
            * (x[:, 0] / 5 - x[:, 1] ** 3)
            * np.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2))
        ) - 2
        return y_lf.reshape(-1, 1)


class PeaksShifted3(MultiFidelityTemplate):
    """Example generated from Peaks function
    Utilizes a non-linear scaling to generate data from peaks function"""

    def __init__(self):
        self.bounds = [[-3.0, 3.0], [-3.0, 3.0]]
        self.dimensions = 2
        self.name = "Peaks Function Shifted"

    def high_fidelity(self, x):
        y_hf = (
            3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
            - 10
            * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
            * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
            - 1 / 3 * np.exp(-((x[:, 0] + 1) ** 2) - x[:, 1] ** 2)
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            -10
            * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
            * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
        ) - 2
        return y_lf.reshape(-1, 1)


class PeaksShifted4(MultiFidelityTemplate):
    """Example generated from Peaks function
    Utilizes a non-linear scaling to generate data from peaks function"""

    def __init__(self):
        self.bounds = [[-3.0, 3.0], [-3.0, 3.0]]
        self.dimensions = 2
        self.name = "Peaks Function Shifted"

    def high_fidelity(self, x):
        y_hf = (
            3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
            - 10
            * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
            * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
            - 1 / 3 * np.exp(-((x[:, 0] + 1) ** 2) - x[:, 1] ** 2)
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            0.9
            * (
                -10
                * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5)
                * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
            )
            - 2
        )
        return y_lf.reshape(-1, 1)


class Zhang22Example5(MultiFidelityTemplate):
    """Example from the Zhang 2022 paper in the appendix"""

    def __init__(self):
        self.bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        self.dimensions = 2
        self.name = "Zhang Example 5"

    def high_fidelity(self, x):
        return (
            4
            - 2.1 * x[:, 0] ** 4
            + (1.0 / 3.0) * x[:, 0] ** 6
            + x[:, 0] * x[:, 1]
            - 4 * x[:, 1] ** 2
            + 4 * x[:, 1] ** 4
        ).reshape(-1, 1)

    def low_fidelity(self, x):
        return (
            4
            - 0.5 * x[:, 0] ** 4
            + 0.04 * x[:, 0] ** 6
            + 0.5 * x[:, 0] * x[:, 1]
            - 1.96 * x[:, 1] ** 2
            + 1.96 * x[:, 1] ** 4
        ).reshape(-1, 1)


multi_fidelity_functions = {
    "Zhang Example 1": ZhangExample1,
    "Zhang Example 3": ZhangExample3,
    "Zhang Example 4": ZhangExample4,
    "Forrester Example": ForresterExample,
    "Meng Discontinuous": MengDiscontinuous,
    "Meng Non-Linear Correlation": MengNonLinearCorrelation,
    "Meng Phase Shifted Oscillations": MengPhaseShift,
}
