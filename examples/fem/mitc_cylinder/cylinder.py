import argparse
import numpy as np
import amigo as am
from scipy.sparse.linalg import spsolve
from utils import write_vtu, get_exact_solution
import time
from amigo.fem import MITCElement, SolutionSpace, Mesh, Problem
from shell_element import (
    NaturalShellGeoBasis,
    ShellSolnBasis,
    MITC4ShellTying,
    integrand,
)

parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy", "cuda"],
    default="cholesky",
)
args = parser.parse_args()

# Load the mesh
mesh = Mesh("cylinder.inp")
domains = mesh.get_domains()

lateral_surfaces = ["SURFACE1"]
bottom_line = "LINE3"
lateral_line = "LINE2"
top_line = "LINE1"
print(f"Lateral: {lateral_surfaces}, bottom: {bottom_line}, top: {top_line}")

# 6 DOF/node: u, v, w translations + rx, ry, rz global rotations
soln_space = SolutionSpace(
    {"u": "H1", "v": "H1", "w": "H1", "rx": "H1", "ry": "H1", "rz": "H1"}
)
geo_space = SolutionSpace(
    {"x": "H1", "y": "H1", "z": "H1", "nx": "H1", "ny": "H1", "nz": "H1"}
)
data_space = SolutionSpace({})

etype = "CPS4"

degree = 1
soln_basis = ShellSolnBasis(degree, kind="input")
geo_basis = NaturalShellGeoBasis(degree, ["x", "y", "z", "nx", "ny", "nz"], kind="data")
quadrature = mesh.get_quadrature(etype)
data_basis = mesh.get_basis(data_space, etype, kind="data")
mitc = MITC4ShellTying()

shell_elem = MITCElement(
    "Shell", soln_basis, data_basis, geo_basis, quadrature, mitc, integrand
)

integrand_map = {
    "shell": {
        "target": lateral_surfaces,
        "integrand": integrand,
    },
}
bc_map = {
    "pinned_bottom": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [bottom_line],
    },
    "pinned_top": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [top_line],
    },
}

problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs={("shell", etype): shell_elem},
)

model = problem.create_model("cylinder_shell")

# Add a fixed boundary condition for the zeroth node.
model.add_fixed("soln.w[0]")

if args.build:
    model.build_module()

model.initialize()

print("Number of variables... ", model.num_variables)

R = 1.0
data = model.get_data_vector()
data["geo.nx"] = data["geo.x"] / R
data["geo.ny"] = data["geo.y"] / R

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

# Copy the data over to the GPU
data = model.get_data_vector()
data.copy_host_to_device()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

num_factors = 1
if args.solver == "cuda":
    from amigo.amigo import CSRMatFactorCuda

    # Duplicate the matrix
    mat_copy = mat.duplicate()
    mat_copy.copy(mat)

    pivot_eps = 1e-12
    solver = CSRMatFactorCuda(mat_copy, pivot_eps)
    solver.factor()

    start_time = time.perf_counter()
    for i in range(num_factors):
        mat_copy.copy(mat)
        solver.factor()
    end_time = time.perf_counter()
    tfactor = (end_time - start_time) / num_factors

    solver.solve(g.get_vector(), x.get_vector())

    x.copy_device_to_host()
else:
    g.copy_device_to_host()
    mat.copy_data_device_to_host()

    if args.solver == "cholesky" or args.solver == "ldl":
        stype = am.SolverType.CHOLESKY
        if args.solver == "ldl":
            stype = am.SolverType.LDL

        ldl = am.SparseLDL(mat, stype, ustab=0.05)
        flag = ldl.factor()

        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = ldl.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"LDL factor flag {flag}")

        x[:] = g[:]
        ldl.solve(x.get_vector())
        if stype == am.SolverType.LDL:
            print("Inertia: ", ldl.get_inertia())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "cholesky_left":
        chol = am.SparseCholesky(mat)
        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = chol.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"Cholesky factor flag {flag}")

        x[:] = g[:]
        chol.solve(x.get_vector())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "scipy":
        csr = am.tocsr(mat)

        # This isn't a completely fair comparison
        start_time = time.perf_counter()
        x[:] = spsolve(csr, g[:])
        end_time = time.perf_counter()

        tfactor = end_time - start_time

print(f"Factor time... {tfactor:.6f} seconds")

u = x["soln.u"]
v = x["soln.v"]
w = x["soln.w"]

u_ex, v_ex, w_ex = get_exact_solution(data["geo.x"], data["geo.y"], data["geo.z"])

w_diff = w - w_ex
print("Error = ", np.max(np.absolute(w_diff)) / np.max(w))

conn = np.vstack([mesh.get_conn(s, "CPS4") for s in lateral_surfaces])
write_vtu(mesh, conn, u, v, w, filename="cylinder_shell.vtu")
write_vtu(mesh, conn, u_ex, v_ex, w_ex, filename="cylinder_shell_exact.vtu")
