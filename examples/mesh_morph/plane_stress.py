import fea_comps
from parser import InpParser
import numpy as np
import argparse
import amigo as am
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import examples.mesh_morph.utils as utils


def plot_matrix(dense_mat):
    plt.figure(figsize=(6, 5))
    plt.imshow(dense_mat, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="value")
    plt.tight_layout()
    return


# Retrieve mesh information for the analysis
inp_filename = "mesh.inp"
parser = InpParser()
parser.parse_inp(inp_filename)

# Get the node locations
X = parser.get_nodes()

# Get element connectivity
conn_surface1 = parser.get_conn("SURFACE1", "CPS3")  # Outer region
conn_surface2 = parser.get_conn("SURFACE2", "CPS3")  # Inner region
conn = np.concatenate((conn_surface1, conn_surface2))

# Get the boundary condition nodes on the outer edge connectivity
edge1 = parser.get_conn("LINE1", "T3D2")
edge2 = parser.get_conn("LINE2", "T3D2")
edge3 = parser.get_conn("LINE3", "T3D2")
edge4 = parser.get_conn("LINE4", "T3D2")

# Get the boundary condition nodes on the inner edge connetivity
edge5 = parser.get_conn("LINE5", "T3D2")
edge6 = parser.get_conn("LINE6", "T3D2")
edge7 = parser.get_conn("LINE7", "T3D2")
edge8 = parser.get_conn("LINE8", "T3D2")

# Concatenate the unique node tags for the outer loop
outer_boundary_bc_tags = np.concatenate(
    (
        edge1.flatten(),
        edge2.flatten(),
        edge3.flatten(),
        edge4.flatten(),
    ),
    axis=None,
)
outer_bc_tags = np.unique(outer_boundary_bc_tags, sorted=True)

# Concatenate the unique node tags for the inner loop
inner_boundary_bc_tags = np.concatenate(
    (
        edge5.flatten(),
        edge6.flatten(),
        edge7.flatten(),
        edge8.flatten(),
    ),
    axis=None,
)
inner_bc_tags = np.unique(inner_boundary_bc_tags, sorted=True)

# Total number of elements and nodes in the mesh
nelems = conn.shape[0]
nnodes = X.shape[0]

# Define parser arguments
par = argparse.ArgumentParser()
par.add_argument(
    "--build",
    dest="build",
    action="store_true",
    default=False,
    help="Enable building",
)
args = par.parse_args()

# Define amigo module
model = am.Model("plane_stress_demo")
ps = fea_comps.PlaneStress()
ps_node_src = fea_comps.NodeSourceMorph()
ps_dirichlet = fea_comps.DirichletBcMorph()

# Components and links
model.add_component("ps", nelems, ps)
model.add_component("ps_node_src", nnodes, ps_node_src)
model.add_component("ps_dirichlet", len(outer_bc_tags), ps_dirichlet)

model.link("ps.x_coord", "ps_node_src.x_coord", tgt_indices=conn)
model.link("ps.y_coord", "ps_node_src.y_coord", tgt_indices=conn)

model.link("ps.u", "ps_node_src.u", tgt_indices=conn)
model.link("ps.v", "ps_node_src.v", tgt_indices=conn)

model.link("ps_node_src.u", "ps_dirichlet.dof", src_indices=outer_bc_tags)
model.link("ps_node_src.v", "ps_dirichlet.dof", src_indices=outer_bc_tags)

# Build module
if args.build:
    model.build_module()

# Initialize model
model.initialize()

# Set problem data
data = model.get_data_vector()
data["ps_node_src.x_coord"] = X[:, 0]
data["ps_node_src.y_coord"] = X[:, 1]

# Vectors for solving the problem
problem = model.get_problem()
mat = problem.create_matrix()
alpha = 1.0
x = problem.create_vector()
ans = problem.create_vector()
g = problem.create_vector()
rhs = problem.create_vector()
problem.hessian(alpha, x, mat)
problem.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)
# print(csr_mat.todense())

# Plot matrix
# plot_matrix(csr_mat.todense())

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
u = ans_local.get_array()[model.get_indices("ps_node_src.u")]
v = ans_local.get_array()[model.get_indices("ps_node_src.v")]

X[:, 0] = X[:, 0] + u
X[:, 1] = X[:, 1] + v


utils.plot_morph(X, [conn])

plt.show()
