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
truss_x = fea_comps.PDE1Dx()
truss_y = fea_comps.PDE1Dy()
node_src_truss_x = fea_comps.NodeSourceTruss_x()
node_src_truss_y = fea_comps.NodeSourceTruss_y()
node7x_dirichlet = fea_comps.DirichletBcNode7x()
node4x_dirichlet = fea_comps.DirichletBcNode4x()
node4y_dirichlet = fea_comps.DirichletBcNode4y()
node5y_dirichlet = fea_comps.DirichletBcNode5y()
node5x_dirichlet = fea_comps.DirichletBcNode7x()
node6x_dirichlet = fea_comps.DirichletBcNode4x()
node6y_dirichlet = fea_comps.DirichletBcNode6y()
node7y_dirichlet = fea_comps.DirichletBcNode7y()

# Add component for line 7
model.add_component(
    name="pde_line7",
    size=edge7.shape[0],
    comp_obj=truss_y,
)

model.add_component(
    name="node_src_line7_y",
    size=edge7.shape[0] + 1,  # Number of nodes on edge 8
    comp_obj=node_src_truss_y,
)
model.add_component(
    name="n6y_dirichlet",
    size=1,
    comp_obj=node7y_dirichlet,
)
model.add_component(
    name="n7y_dirichlet",
    size=1,
    comp_obj=node7y_dirichlet,
)

for i in range(edge7.shape[0]):
    model.link(f"pde_line7.y_coord[{i}]", f"node_src_line7_y.y_coord[{i}:{i+2}]")
    model.link(f"pde_line7.dy[{i}]", f"node_src_line7_y.dy[[{i},{i+1}]]")

model.link("node_src_line7_y.dy", "n6y_dirichlet.dy", src_indices=[0])
model.link("node_src_line7_y.dy", "n7y_dirichlet.dy", src_indices=[-1])

# Add component for line 6
model.add_component(
    name="pde_line6",
    size=edge6.shape[0],
    comp_obj=truss_x,
)

model.add_component(
    name="node_src_line6_x",
    size=edge6.shape[0] + 1,
    comp_obj=node_src_truss_x,
)

model.add_component(
    name="n5x_dirichlet",
    size=1,
    comp_obj=node5x_dirichlet,
)

model.add_component(
    name="n6x_dirichlet",
    size=1,
    comp_obj=node6x_dirichlet,
)

for i in range(edge6.shape[0]):
    model.link(f"pde_line6.x_coord[{i}]", f"node_src_line6_x.x_coord[{i}:{i+2}]")
    model.link(f"pde_line6.dx[{i}]", f"node_src_line6_x.dx[[{i},{i+1}]]")

model.link("node_src_line6_x.dx", "n5x_dirichlet.dx", src_indices=[0])
model.link("node_src_line6_x.dx", "n6x_dirichlet.dx", src_indices=[-1])


# Add component for line 5
model.add_component(
    name="pde_line5",
    size=edge5.shape[0],
    comp_obj=truss_y,
)

model.add_component(
    name="node_src_line5_y",
    size=edge5.shape[0] + 1,  # Number of nodes on edge 8
    comp_obj=node_src_truss_y,
)
model.add_component(
    name="n4y_dirichlet",
    size=1,
    comp_obj=node4y_dirichlet,
)
model.add_component(
    name="n5y_dirichlet",
    size=1,
    comp_obj=node5y_dirichlet,
)

for i in range(edge5.shape[0]):
    model.link(f"pde_line5.y_coord[{i}]", f"node_src_line5_y.y_coord[{i}:{i+2}]")
    model.link(f"pde_line5.dy[{i}]", f"node_src_line5_y.dy[[{i},{i+1}]]")

model.link("node_src_line5_y.dy", "n4y_dirichlet.dy", src_indices=[0])
model.link("node_src_line5_y.dy", "n5y_dirichlet.dy", src_indices=[-1])

# Add component for line 8
model.add_component(
    name="pde_line8",
    size=edge8.shape[0],  # Number of elements on edge 8
    comp_obj=truss_x,
)

# Define the node source for line8 dx morph
model.add_component(
    name="node_src_line8_x",
    size=edge8.shape[0] + 1,  # Number of nodes on edge 8
    comp_obj=node_src_truss_x,
)

# Define dirichlet BC for line 8
model.add_component(
    name="n7x_dirichlet",
    size=1,
    comp_obj=node7x_dirichlet,
)

model.add_component(
    name="n4x_dirichlet",
    size=1,
    comp_obj=node4x_dirichlet,
)

# Link objects
for i in range(edge8.shape[0]):
    model.link(f"pde_line8.x_coord[{i}]", f"node_src_line8_x.x_coord[{i}:{i+2}]")
    model.link(f"pde_line8.dx[{i}]", f"node_src_line8_x.dx[[{i},{i+1}]]")

# Link BC
model.link("node_src_line8_x.dx", "n7x_dirichlet.dx", src_indices=[0])
model.link("node_src_line8_x.dx", "n4x_dirichlet.dx", src_indices=[-1])

# Build module
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

# Store the original x,y coords of the nodes
node7_coords = X[7]
node4_coords = X[4]
node5_coords = X[5]
node6_coords = X[6]

# Define new locations for the nodes
line5_x_val = 0.5
line7_x_val = -0.2
line6_y_val = 0.3
line8_y_val = -0.1

# Define the y offset for lines 8 amd 6
y_offset_line8 = line8_y_val + 1.0
y_offset_line6 = line6_y_val - 1.0

# Define the x offset for lines 5 amd 7
x_offset_line5 = line5_x_val - 1.0
x_offset_line7 = line7_x_val + 1.0

# Set the problem data
data = model.get_data_vector()
data["node_src_line8_x.x_coord"] = np.unique(X[edge8, 0], sorted=False)
data["node_src_line5_y.y_coord"] = np.unique(X[edge5, 1], sorted=False)
data["node_src_line6_x.x_coord"] = np.unique(X[edge6, 0], sorted=False)
data["node_src_line7_y.y_coord"] = np.unique(X[edge7, 1], sorted=False)

data["n7x_dirichlet.val"] = line7_x_val
data["n4x_dirichlet.val"] = line5_x_val

data["n4y_dirichlet.val"] = line8_y_val
data["n5y_dirichlet.val"] = line6_y_val

data["n5x_dirichlet.val"] = line5_x_val
data["n6x_dirichlet.val"] = line7_x_val

data["n6y_dirichlet.val"] = line6_y_val
data["n7y_dirichlet.val"] = line8_y_val

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
# print(csr_mat.todense())

# Plot matrix
# plot_matrix(csr_mat.todense())
# plt.show()

# Solve the problem
ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
line8_x_vals = ans_local.get_array()[model.get_indices("node_src_line8_x.dx")]
line5_y_vals = ans_local.get_array()[model.get_indices("node_src_line5_y.dy")]
line6_x_vals = ans_local.get_array()[model.get_indices("node_src_line6_x.dx")]
line7_y_vals = ans_local.get_array()[model.get_indices("node_src_line7_y.dy")]


# Plot before and after
fig, ax = plt.subplots()

line8_xcoords = X[edge8.flatten(), 0]
line8_ycoords = X[edge8.flatten(), 1]
line5_xcoords = X[edge5.flatten(), 0]
line5_ycoords = X[edge5.flatten(), 1]
line6_xcoords = X[edge6.flatten(), 0]
line6_ycoords = X[edge6.flatten(), 1]
line7_xcoords = X[edge7.flatten(), 0]
line7_ycoords = X[edge7.flatten(), 1]
ax.plot(line8_xcoords, line8_ycoords, "ko--", label="Line 8")
ax.plot(line5_xcoords, line5_ycoords, "ko--", label="Line 5")
ax.plot(line6_xcoords, line6_ycoords, "ko--", label="Line 6")
ax.plot(line7_xcoords, line7_ycoords, "ko--", label="Line 7")

# Overwrite the coordinates that have a constant value
# This enables plotting the solution field
line8_ycoords = X[np.unique(edge8.flatten()), 1]
line5_xcoords = X[np.unique(edge5.flatten()), 0]
line6_ycoords = X[np.unique(edge6.flatten()), 1]
line7_xcoords = X[np.unique(edge7.flatten()), 0]
ax.plot(
    line8_x_vals,
    line8_ycoords + y_offset_line8,
    "ro-",
    label="Line 8 (new)",
)
ax.plot(
    line5_xcoords + x_offset_line5,
    line5_y_vals,
    "ro-",
    label="Line 5 (new)",
)
ax.plot(
    line6_x_vals,
    line6_ycoords + y_offset_line6,
    "ro-",
    label="Line 6 (new)",
)
ax.plot(
    line7_xcoords + x_offset_line7,
    line7_y_vals,
    "ro-",
    label="Line 7 (new)",
)

# ax.legend()
plt.savefig("demo.jpg", dpi=500)
# plt.show()
