import numpy as np
from scipy.stats import qmc
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

    def sample_uniform(self, n):
        axes = [
            np.linspace(self.bounds[i][0], self.bounds[i][1], n)
            for i in range(self.dimensions)
        ]
        # Create full tensor grid
        grids = np.meshgrid(*axes, indexing="ij")

        # Stack into (N_total, dim)
        X = np.stack([g.ravel() for g in grids], axis=-1)
        return X

    def sample_random(self, n, seed=1234):
        X = latin_hypercube_sampler(
            n=n, d=self.dimensions, bounds=self.bounds, seed=seed
        )
        return X


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
            0.79 * (1 + np.sin(x[:, 0]) / 10) * self.low_fidelity(x)
            - 2 * x[:, 0]
            + x[:, 1] ** 2
            + x[:, 2] ** 2
            + 0.5
        )
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (x[:, 0] / 2) * (
            np.sqrt(1 + (x[:, 0] + x[:, 2] ** 2) * x[:, 3] / x[:, 0] ** 20) - 1
        ) + (x[:, 0] + 3 * x[:, 3]) * np.exp(1 + np.sin(x[:, 2]))
        return y_lf.reshape(-1, 1)


class ZhangExample9(MultiFidelityTemplate):
    """Example 9 from
    "A multi-fidelity surrogate modeling approach for incorporating multiple
    non-hierarchical low-fidelity data" (2022)   Zhang et al.

    4D input to function
    """

    def __init__(self):
        self.bounds = [
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
        ]
        self.dimensions = 6
        self.name = "Zhang Example 9"

    def high_fidelity(self, x):
        y_hf = (
            100 * (x[:, 1] - x[:, 0] ** 2) ** 2
            + 4 * (x[:, 0] - 1) ** 4
            + 100 * (x[:, 2] - x[:, 1] ** 2) ** 2
            + 4 * (x[:, 1] - 1) ** 4
            + 100 * (x[:, 3] - x[:, 2] ** 2) ** 2
            + 4 * (x[:, 2] - 1) ** 4
            + 100 * (x[:, 4] - x[:, 3] ** 2) ** 2
            + 4 * (x[:, 3] - 1) ** 4
            + 100 * (x[:, 5] - x[:, 4] ** 2) ** 2
            + 4 * (x[:, 4] - 1) ** 4
        ) / 100000
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = (
            100 * (x[:, 1] - x[:, 0] ** 2) ** 2
            + (x[:, 0] - 1) ** 2
            + 100 * (x[:, 2] - x[:, 1] ** 2)
            + (x[:, 1] - 1) ** 2
            + 100 * (x[:, 3] - x[:, 2] ** 2) ** 2
            + (x[:, 2] - 1) ** 2
            + 100 * (x[:, 4] - x[:, 3] ** 2) ** 2
            + (x[:, 3] - 1) ** 2
            + 100 * (x[:, 5] - x[:, 4] ** 2) ** 2
            + (x[:, 4] - 1) ** 2
        ) / 100000
        return y_lf.reshape(-1, 1)


class Zhang22Example17(MultiFidelityTemplate):
    """Example from Zhang 2022 paper Multi-Fidelity Surrogate Model based on feasible intervals

    This is a 4D example case ove a [-1,1] interval
    """

    def __init__(self):
        self.bounds = [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]
        self.dimensions = 4
        self.name = "Zhang Example 17"

    def low_fidelity(self, x):
        y_lf = self.high_fidelity(x) - 1.5 * (
            5 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2 + 3 * x[:, 2] + x[:, 3] ** 2
        )
        return y_lf.reshape(-1, 1)

    def high_fidelity(self, x):
        x1, x2, x3, x4 = x.T

        y_hf = (
            100 * (x1**2 - x2) ** 2
            + (x1 - 1) ** 2
            + 90 * (x3**2 - x4) ** 2
            + 10.1 * (x2 - 1) ** 2
            + (x4 - 1) ** 2
            + 19.8 * (x2 - 1) * (x4 - 1)
        )

        return y_hf[:, None]


class Zhang22Example18(MultiFidelityTemplate):
    """Example from Zhang 2022 paper Multi-Fidelity Surrogate Model based on feasible intervals

    This is a 6D example case over a [-5,10] interval
    """

    def __init__(self):
        self.bounds = [
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
            [-5, 10],
        ]
        self.dimensions = 6
        self.name = "Zhang Example 18"

    def high_fidelity(self, x):
        y_1 = 0
        y_2 = 0
        for i in range(1, 6):
            y_1 = y_1 + x[:, i - 1] ** 2
            y_2 = y_2 + 0.5 * i * x[:, i - 1]

        y_hf = y_1 + y_2**2 + y_2**4
        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_1 = 0
        for i in range(1, 6):
            y_1 = y_1 + 0.5 * i * x[:, i - 1]

        y_lf = 2.5 * (y_1**2 + y_1**4)
        return y_lf.reshape(-1, 1)


class Zhang22Example19(MultiFidelityTemplate):
    """Example from Zhang 2022 paper Multi-Fidelity Surrogate Model based on feasible intervals

    This is an 8D example case over a [0,1] interval
    """

    def __init__(self):
        self.bounds = [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
        self.dimensions = 8
        self.name = "Zhang Example 19"

    def high_fidelity(self, x):
        y_hf = 0
        for i in range(1, 2):
            y_hf = (
                (x[:, 4 * i - 4] + 10 * x[:, 4 * i - 3]) ** 2
                + 5 * (x[:, 4 * i - 2] - x[:, 4 * i - 1]) ** 2
                + (x[:, 4 * i - 3] - 2 * x[:, 4 * i - 2]) ** 4
                + 10 * (x[:, 4 * i - 4] - x[:, 4 * i - 1]) ** 4
            )

        return y_hf.reshape(-1, 1)

    def low_fidelity(self, x):
        y_lf = 0
        for i in range(1, 2):
            y_lf = (
                (x[:, 4 * i - 4] + x[:, 4 * i - 3]) ** 2
                + 5 * (x[:, 4 * i - 2] - x[:, 4 * i - 1]) ** 2
                + (x[:, 4 * i - 3] - 2 * x[:, 4 * i - 2]) ** 4
                + 10 * (x[:, 4 * i - 4] - x[:, 4 * i - 1]) ** 4
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


def latin_hypercube_sampler(n: int, d: int, bounds: list, seed: int):
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_sample = sampler.random(n)

    bounds = np.asarray(bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    sample = lower + unit_sample * (upper - lower)
    return sample


multi_fidelity_functions = {
    "Zhang Example 1": ZhangExample1,
    "Zhang Example 3": ZhangExample3,
    "Zhang Example 4": ZhangExample4,
    "Forrester Example": ForresterExample,
    "Meng Discontinuous": MengDiscontinuous,
    "Meng Non-Linear Correlation": MengNonLinearCorrelation,
    "Meng Phase Shifted Oscillations": MengPhaseShift,
}

increasing_dimensions = {
    "1D": ForresterExample,
    "2D": PeaksShifted2,
    "4D": Zhang22Example17,
    "6D": Zhang22Example18,
    "8D": Zhang22Example19,
}
