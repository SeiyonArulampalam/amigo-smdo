import sys
import argparse
import numpy as np
import amigo as am

sys.path.append("../mitc_cylinder")
from shell_element import (
    NaturalShellGeoBasis,
    ShellSolnBasis,
    MITC4ShellTying,
)
from amigo.fem import (
    MITCElement,
    Space,
    FunctionSpace,
    Conformity,
    SolutionSpace,
    CellType,
    Mesh,
    Problem,
    make_quadrature,
    build_grid,
)


def integrand(soln, data=None, geo=None):
    nx = geo["nx"].value
    ny = geo["ny"].value
    nz = geo["nz"].value

    u = soln["u"].value
    v = soln["v"].value
    w = soln["w"].value

    rx = soln["rx"].value
    ry = soln["ry"].value
    rz = soln["rz"].value

    # Gradients for the bending terms
    u1x, u1y = soln["u1"].grad
    v1x, v1y = soln["v1"].grad

    # In-plane strains from MITC interpolation
    ex = soln["ex"].value
    ey = soln["ey"].value
    gxy = soln["gxy"].value

    # Shear strains from MITC interpolation
    gxz = soln["gxz"].value
    gyz = soln["gyz"].value

    kx = u1x
    ky = v1y
    kxy = u1y + v1x

    E = 70e9
    nu = 0.3
    ks = 5.0 / 6.0

    t = 0.01
    penalty = 1e-4
    k_drill = penalty * E * t

    # Compute the in-plane energy
    A = E * t / (1.0 - nu**2)
    Um = 0.5 * A * ((ex**2 + ey**2 + 2.0 * nu * ex * ey) + 0.5 * (1.0 - nu) * gxy**2)

    # Compute the bending energy
    D = E * t**3 / (12 * (1.0 - nu**2))
    Ub = 0.5 * D * ((kx**2 + ky**2 + 2.0 * nu * kx * ky) + 0.5 * (1.0 - nu) * kxy**2)

    # Compute the shear energy
    G = 0.5 * E / (1.0 + nu)
    Us = 0.5 * ks * G * t * (gxz**2 + gyz**2)

    # Compute the drill penalty
    rot = rx * nx + ry * ny + rz * nz

    Ud = 0.5 * k_drill * rot**2

    U = Um + Ub + Us + Ud

    fx = 0.0
    fy = 0.0
    fz = -1.0
    W = fx * u + fy * v + fz * w

    return U - W


def local_normal(soln, data=None, geo=None):
    x1, x2 = geo["x"].grad
    y1, y2 = geo["y"].grad
    z1, z2 = geo["z"].grad

    # Cross product t1 x t2
    nx = y1 * z2 - z1 * y2
    ny = z1 * x2 - x1 * z2
    nz = x1 * y2 - y1 * x2

    # Normalize
    norm = am.sqrt(nx**2 + ny**2 + nz**2)

    nx /= norm
    ny /= norm
    nz /= norm

    return {"nx0": nx, "ny0": ny, "nz0": nz, "weight": 1.0}


def set_normals(model):
    """Set the output normals in the model"""
    x = model.create_vector()
    output = model.create_output_vector()

    model.compute_output(x, output)

    data = model.get_data_vector()

    data["normals.nx"] = output["out_normals.nx0"] / output["out_normals.weight"]
    data["normals.ny"] = output["out_normals.ny0"] / output["out_normals.weight"]
    data["normals.nz"] = output["out_normals.nz0"] / output["out_normals.weight"]

    return


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
filename = "CRM_box_2nd.bdf"
mesh = Mesh(filename)

# Get the domain names
domains = mesh.get_domain_names()

# Seet the function spaces
H1 = FunctionSpace(func_space=Space.H1, degree=1, conformity=Conformity.GLOBAL)
H1c = FunctionSpace(func_space=Space.H1, degree=1, conformity=Conformity.COMPONENT)

# 6 DOF/node: u, v, w translations + rx, ry, rz global rotations
soln_space = SolutionSpace({"soln": {("u", "v", "w", "rx", "ry", "rz"): H1}})

# Set the geometry space
geo_space = SolutionSpace(
    {"geo": {("x", "y", "z"): H1}, "normals": {("nx", "ny", "nz"): H1c}}
)

# The field output space
output_space = SolutionSpace({"out_normals": {("nx0", "ny0", "nz0", "weight"): H1c}})

ctype = CellType.QUADRILATERAL
soln_basis = ShellSolnBasis(kind="input")
geo_basis = NaturalShellGeoBasis(["x", "y", "z", "nx", "ny", "nz"], H1, kind="data")
quadrature = make_quadrature(soln_space, ctype)
data_basis = None
mitc = MITC4ShellTying()

shell_elem = MITCElement(
    "Shell", soln_basis, data_basis, geo_basis, quadrature, mitc, integrand
)

targets = []
for name in domains:
    targets.append(name)

integrand_map = {
    "shell": {
        "target": targets,
        "integrand": integrand,
    },
}

output_map = {
    "normals": {
        "target": targets,
        "function": local_normal,
    }
}

problem = Problem(
    mesh,
    soln_space,
    geo_space,
    integrand_map=integrand_map,
    element_objs={("shell", ctype): shell_elem},
)

model = problem.create_model("crm_model")

problem.add_field_output(model, output_space=output_space, output_map=output_map)

if args.build:
    model.build_module()

model.initialize()

# Set the normals
set_normals(model)

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

ldl = am.SparseLDL(mat, am.SolverType.CHOLESKY, ustab=0.05)
flag = ldl.factor()

x[:] = -g[:]
ldl.solve(x.get_vector())

grid = build_grid(problem, "w", x)
grid.plot(scalars="w", cmap="coolwarm", show_edges=True)
