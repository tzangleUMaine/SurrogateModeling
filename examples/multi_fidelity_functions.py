import numpy as np
import matplotlib.pyplot as plt


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

    def plot_high_fidelity_data_1d(
        self, name="Model", color="black", label="high fidelity fxn", linestyle="solid"
    ):
        if self.dimensions != 1:
            print("Wrong plotting function used for current dimensionality")
            return

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 300)
        y = self.high_fidelity(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x, y, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")
        return

    def plot_low_fidelity_data_1d(
        self, name="Model", color="blue", label="low fidelity fxn", linestyle="solid"
    ):
        if self.dimensions != 1:
            print("Wrong plotting function used for current dimensionality")
            return

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 300)
        y = self.low_fidelity(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x, y, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")
        return

    def plot_functions_1d(self, name="Model"):
        self.plot_low_fidelity_data_1d(name=name)
        self.plot_high_fidelity_data_1d(name=name)
        return


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


multi_fidelity_functions = {
    "Zhang Example 1": ZhangExample1,
    "Zhang Example 3": ZhangExample3,
    "Zhang Example 4": ZhangExample4,
    "Forrester Example": ForresterExample,
    "Meng Discontinuous": MengDiscontinuous,
    "Meng Non-Linear Correlation": MengNonLinearCorrelation,
    "Meng Phase Shifted Oscillations": MengPhaseShift,
}


def generate_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
    x = np.linspace(xmin, xmax, num)
    y = np.linspace(ymin, ymax, num)
    X, Y = np.meshgrid(x, y)
    return X, Y


def generate_grid_array(xmin=-2, xmax=2, ymin=-2, ymax=2, num=50):
    x = np.linspace(xmin, xmax, num)
    y = np.linspace(ymin, ymax, num)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    return grid
