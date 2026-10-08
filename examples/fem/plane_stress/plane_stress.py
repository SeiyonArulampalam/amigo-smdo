from amigo.fem import (
    Mesh,
    Problem,
    CellType,
    Space,
    FunctionSpace,
    SolutionSpace,
    build_grid,
)
import amigo as am
import matplotlib.pyplot as plt
import numpy as np
import argparse


def potential_plane_stress(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    # Displacement gradients in physical space
    u_grad = soln["u"].grad
    v_grad = soln["v"].grad

    # Strain components
    exx = u_grad[0]
    eyy = v_grad[1]
    exy = u_grad[1] + v_grad[0]

    # Material properties (hardcoded or pull from data)
    E = 1.0
    nu = 0.3
    c = E / (1.0 - nu**2)  # poisson's ratio

    # Constitutive matrix C acting on [e11, e22, e12]
    # W = 0.5 * eT C e
    W = 0.5 * c * (exx**2 + eyy**2 + 2.0 * nu * exx * eyy + 0.5 * (1.0 - nu) * exy**2)

    return W


def potential_traction(soln, data=None, geo=None):
    """External Work Line integral integrand"""
    u = soln["u"].value
    v = soln["v"].value
    # traction force for element
    # W = uT t
    tx = 0
    ty = -1

    W = u * tx + v * ty
    return -W


# Two displacement DOFs per node
H1 = FunctionSpace(func_space=Space.H1, degree=1)
soln_space = SolutionSpace({"soln": {("u", "v"): H1}})
geo_space = SolutionSpace({"geo": {("x", "y"): H1}})
data_space = SolutionSpace({})

integrand_map = {
    "plane_stress": {
        "target": ["SURFACE1"],
        "integrand": potential_plane_stress,
    },
    "traction": {
        "target": ["LINE2"],
        "integrand": potential_traction,
    },
}

bc_map = {
    "clamp_x": {
        "type": "dirichlet",
        "target": ["LINE4"],  # left edge — fix ux
        "input": ["u", "v"],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

mesh = Mesh("plate.inp")
problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)

model = problem.create_model("plane_stress")

if args.build:
    model.build_module()

model.initialize()

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Solve the equations
print("Solving...")
chol = am.SparseLDL(mat, solver_type=am.SolverType.CHOLESKY)
flag = chol.factor()

# Solve the equations
x[:] = g[:]
chol.solve(x.get_vector())

u = x["soln.u"]
v = x["soln.v"]

# Plot the solution
grid = build_grid(problem, "u", x)
grid.plot(scalars="u", cmap="coolwarm", show_edges=True)
