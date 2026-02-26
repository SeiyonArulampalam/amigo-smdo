import fea_comps
from parser import InpParser
import numpy as np
import argparse
import amigo as am
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


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
module_name = "morph"
model = am.Model(module_name=module_name)

# Add physics components
truss_x = fea_comps.Truss1Dx()
node_src_truss_x = fea_comps.NodeSourceTruss_x()
node1_dirichlet = fea_comps.DirichletBcNode1()
node2_dirichlet = fea_comps.DirichletBcNode2()

# Add component for line 8
# print(edge8.shape[0])
model.add_component(
    name="truss_line8",
    size=edge8.shape[0],  # Number of elements on edge 8
    comp_obj=truss_x,
)

# Define a global node source component
model.add_component(
    name="node_src_x",
    size=edge8.shape[0] + 1,  # Number of nodes on edge 8
    comp_obj=node_src_truss_x,
)

# Define dirichlet BC
model.add_component(
    name="n1_dirichlet",
    size=1,
    comp_obj=node1_dirichlet,
)

model.add_component(
    name="n2_dirichlet",
    size=1,
    comp_obj=node2_dirichlet,
)

# Link objects
# print(edge8)
# print(X[edge8, 0])
# print(np.unique(X[edge8], 0, sorted=False))
l = np.unique(X[edge8, 0], sorted=False)
for i in range(edge8.shape[0]):
    # Loop through each element and link x_coord
    # print(f"loop{i}:", X[edge8, 0][i])
    # print(l[i], l[i+1])
    # print(l[i : i + 2])
    model.link(f"truss_line8.x_coord[{i}]", f"node_src_x.x_coord[{i}:{i+2}]")

c = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
]
model.link(f"truss_line8.dx", f"node_src_x.dx", tgt_indices=c)

# Link BC
model.link("node_src_x.dx", "n1_dirichlet.dx", src_indices=[0])
model.link("node_src_x.dx", "n2_dirichlet.dx", src_indices=[-1])

# Build module
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

# Set the problem data
data = model.get_data_vector()
data["node_src_x.x_coord"] = np.unique(X[edge8, 0], sorted=False)
problem = model.get_problem()
mat = problem.create_matrix()

# Vectors for solving the problem
alpha = 1.0
x = problem.create_vector()
ans = problem.create_vector()
g = problem.create_vector()
rhs = problem.create_vector()
problem.hessian(alpha, x, mat)
problem.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)
print(csr_mat.todense())

# Plot matrix
# plot_matrix(csr_mat.todense())
# plt.show()

# Solve the problem
ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
vals = ans_local.get_array()[model.get_indices("truss_line8.dx")]
print(vals)

# Plot before and after
fig, ax = plt.subplots()
baseline_x = np.unique(X[edge8, 0], sorted=False)
morph_x = np.unique(vals, sorted=False)
y_vals = -1 * np.ones(8)
ax.plot(baseline_x, y_vals, "ko-", label="Baseline")
ax.plot(morph_x, y_vals, "bx--", label="Morph")
plt.show()