import matplotlib.pyplot as plt
import numpy as np


def N(i, p, u, U):
    """
    B-spline basis function N_{i,p}(u) using the Cox–de Boor recursion.

    i : basis function index
    p : degree
    u : parameter value
    U : knot vector (array-like)
    """

    # Convert to list/array for safety
    U = list(U)

    # Base case: degree 0
    if p == 0:
        return 1.0 if (U[i] <= u < U[i + 1]) else 0.0

    # First term
    left = 0.0
    denom1 = U[i + p] - U[i]
    if denom1 != 0:
        left = (u - U[i]) / denom1 * N(i, p - 1, u, U)

    # Second term
    right = 0.0
    denom2 = U[i + p + 1] - U[i + 1]
    if denom2 != 0:
        right = (U[i + p + 1] - u) / denom2 * N(i + 1, p - 1, u, U)

    return left + right


knot_vector = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
degree = 3
u_values = np.linspace(-3, 7, 100)
i = 0
fig = plt.figure(figsize=(2, 1))
for i in range(0, 7):
    plt.xlabel("x")
    plt.ylabel("B(x)")
    plt.xticks(knot_vector)
    basis_values = [N(i, degree, u, knot_vector) for u in u_values]
    plt.plot(u_values, basis_values, label=f"N_{{{i},{degree}}}(u)")
plt.show()
