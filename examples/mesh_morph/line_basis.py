import numpy as np


def eval_shape_funcs(xi):
    N = np.array(
        [
            0.5 * (1.0 - xi),
            0.5 * (1.0 + xi),
        ]
    )
    Nxi = np.array([-0.5, 0.5])

    return N, Nxi


def dot(N, u):
    return N[0] * u[0] + N[1] * u[1]


def compute_detJ(xi, X):
    N, N_xi = eval_shape_funcs(xi)

    x_xi = dot(N_xi, X)

    detJ = x_xi
    return x_xi, detJ


def compute_shape_derivs(xi, X):
    N, N_xi = eval_shape_funcs(xi)

    x_xi, detJ = compute_detJ(xi, X)

    invJ = 1 / x_xi

    Nx = [
        invJ * N_xi[0],
        invJ * N_xi[1],
    ]

    return N, N_xi, Nx, detJ
