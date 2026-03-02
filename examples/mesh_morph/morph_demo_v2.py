import fea_comps
from parser import InpParser
import numpy as np
import argparse
import amigo as am
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import examples.mesh_morph.utils as utils

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

# Define physics classes for the truss
planar_truss = fea_comps.PlanarTruss()
node_src_planar_truss = fea_comps.NodeSourcePlanarTruss()
dirichlet_bc_planar_truss = fea_comps.DirichletBcPlanarTruss()

# Line 5
nelems_line5 = edge5.shape[0]
nnodes_line5 = edge5.shape[0] + 1

model.add_component(
    "truss_line_5",
    size=nelems_line5,
    comp_obj=planar_truss,
)

model.add_component(
    "node_src_line_5",
    size=nnodes_line5,
    comp_obj=node_src_planar_truss,
)

model.add_component(
    "dirichlet_line_5_x",
    size=nnodes_line5,
    comp_obj=dirichlet_bc_planar_truss,
)

model.add_component(
    "dirichlet_line_5_y",
    size=2,
    comp_obj=dirichlet_bc_planar_truss,
)

# Get the unique node tags on each line (remove duplicates and keep the original ordering)
line5_node_tags = np.array(list(dict.fromkeys(edge5.flatten())))

for i in range(edge5.shape[0]):
    model.link(f"truss_line_5.x_coord[{i}]", f"node_src_line_5.x_coord[{i}:{i+2}]")
    model.link(f"truss_line_5.u_truss[{i}]", f"node_src_line_5.u_truss[[{i},{i+1}]]")

    model.link(f"truss_line_5.y_coord[{i}]", f"node_src_line_5.y_coord[{i}:{i+2}]")
    model.link(f"truss_line_5.v_truss[{i}]", f"node_src_line_5.v_truss[[{i},{i+1}]]")

model.link("node_src_line_5.v_truss", "dirichlet_line_5_y.dof", src_indices=[0, -1])
model.link("node_src_line_5.u_truss", "dirichlet_line_5_x.dof")

# Line 7
nelems_line7 = edge7.shape[0]
nnodes_line7 = edge7.shape[0] + 1

model.add_component(
    "truss_line_7",
    size=nelems_line7,
    comp_obj=planar_truss,
)

model.add_component(
    "node_src_line_7",
    size=nnodes_line7,
    comp_obj=node_src_planar_truss,
)

model.add_component(
    "dirichlet_line_7_x",
    size=nnodes_line7,
    comp_obj=dirichlet_bc_planar_truss,
)

model.add_component(
    "dirichlet_line_7_y",
    size=2,
    comp_obj=dirichlet_bc_planar_truss,
)

# Get the unique node tags on each line (remove duplicates and keep the original ordering)
line7_node_tags = np.array(list(dict.fromkeys(edge7.flatten())))

for i in range(edge7.shape[0]):
    model.link(f"truss_line_7.x_coord[{i}]", f"node_src_line_7.x_coord[{i}:{i+2}]")
    model.link(f"truss_line_7.u_truss[{i}]", f"node_src_line_7.u_truss[[{i},{i+1}]]")

    model.link(f"truss_line_7.y_coord[{i}]", f"node_src_line_7.y_coord[{i}:{i+2}]")
    model.link(f"truss_line_7.v_truss[{i}]", f"node_src_line_7.v_truss[[{i},{i+1}]]")

model.link("node_src_line_7.v_truss", "dirichlet_line_7_y.dof", src_indices=[0, -1])
model.link("node_src_line_7.u_truss", "dirichlet_line_7_x.dof")

# Line 8
nelems_line8 = edge8.shape[0]
nnodes_line8 = edge8.shape[0] + 1

model.add_component(
    "truss_line_8",
    size=nelems_line8,
    comp_obj=planar_truss,
)

model.add_component(
    "node_src_line_8",
    size=nnodes_line8,
    comp_obj=node_src_planar_truss,
)

model.add_component(
    "dirichlet_line_8_x",
    size=2,
    comp_obj=dirichlet_bc_planar_truss,
)

model.add_component(
    "dirichlet_line_8_y",
    size=nnodes_line8,
    comp_obj=dirichlet_bc_planar_truss,
)

# Get the unique node tags on each line (remove duplicates and keep the original ordering)
line8_node_tags = np.array(list(dict.fromkeys(edge8.flatten())))

for i in range(edge8.shape[0]):
    model.link(f"truss_line_8.x_coord[{i}]", f"node_src_line_8.x_coord[{i}:{i+2}]")
    model.link(f"truss_line_8.u_truss[{i}]", f"node_src_line_8.u_truss[[{i},{i+1}]]")

    model.link(f"truss_line_8.y_coord[{i}]", f"node_src_line_8.y_coord[{i}:{i+2}]")
    model.link(f"truss_line_8.v_truss[{i}]", f"node_src_line_8.v_truss[[{i},{i+1}]]")

model.link("node_src_line_8.u_truss", "dirichlet_line_8_x.dof", src_indices=[0, -1])
model.link("node_src_line_8.v_truss", "dirichlet_line_8_y.dof")

# Line 6
nelems_line6 = edge6.shape[0]
nnodes_line6 = edge6.shape[0] + 1

model.add_component(
    "truss_line_6",
    size=nelems_line6,
    comp_obj=planar_truss,
)

model.add_component(
    "node_src_line_6",
    size=nnodes_line6,
    comp_obj=node_src_planar_truss,
)

model.add_component(
    "dirichlet_line_6_x",
    size=2,
    comp_obj=dirichlet_bc_planar_truss,
)

model.add_component(
    "dirichlet_line_6_y",
    size=nnodes_line6,
    comp_obj=dirichlet_bc_planar_truss,
)

# Get the unique node tags on each line (remove duplicates and keep the original ordering)
line6_node_tags = np.array(list(dict.fromkeys(edge6.flatten())))

for i in range(edge6.shape[0]):
    model.link(f"truss_line_6.x_coord[{i}]", f"node_src_line_6.x_coord[{i}:{i+2}]")
    model.link(f"truss_line_6.u_truss[{i}]", f"node_src_line_6.u_truss[[{i},{i+1}]]")

    model.link(f"truss_line_6.y_coord[{i}]", f"node_src_line_6.y_coord[{i}:{i+2}]")
    model.link(f"truss_line_6.v_truss[{i}]", f"node_src_line_6.v_truss[[{i},{i+1}]]")

model.link("node_src_line_6.u_truss", "dirichlet_line_6_x.dof", src_indices=[0, -1])
model.link("node_src_line_6.v_truss", "dirichlet_line_6_y.dof")

# Define classes for the plane stress model
ps = fea_comps.PlaneStress()
ps_node_src = fea_comps.NodeSourceMorph()
ps_dirichlet = fea_comps.DirichletBcMorph()

# Plane stress components and links
model.add_component("ps", nelems, ps)
model.add_component("ps_node_src", nnodes, ps_node_src)
model.add_component("ps_dirichlet", len(outer_bc_tags), ps_dirichlet)

model.link("ps.x_coord", "ps_node_src.x_coord", tgt_indices=conn)
model.link("ps.y_coord", "ps_node_src.y_coord", tgt_indices=conn)

model.link("ps.u", "ps_node_src.u", tgt_indices=conn)
model.link("ps.v", "ps_node_src.v", tgt_indices=conn)

# Zero dirichlet BC for the outter boundary
model.link("ps_node_src.u", "ps_dirichlet.dof", src_indices=outer_bc_tags)
model.link("ps_node_src.v", "ps_dirichlet.dof", src_indices=outer_bc_tags)

# Dirichlet for line 5
model.link("ps_node_src.u", "node_src_line_5.u_truss", src_indices=line5_node_tags)
model.link("ps_node_src.v", "node_src_line_5.v_truss", src_indices=line5_node_tags)

# Dirichlet for line 7
model.link("ps_node_src.u", "node_src_line_7.u_truss", src_indices=line7_node_tags)
model.link("ps_node_src.v", "node_src_line_7.v_truss", src_indices=line7_node_tags)

# Dirichlet for line 6 (Interior nodes only)
model.link(
    "ps_node_src.u",
    f"node_src_line_6.u_truss[{1}:{-1}]",
    src_indices=line6_node_tags[1:-1],
)
model.link(
    "ps_node_src.v",
    f"node_src_line_6.v_truss[{1}:{-1}]",
    src_indices=line6_node_tags[1:-1],
)

# Dirichlet for line 8 (Interior nodes only)
model.link(
    "ps_node_src.u",
    f"node_src_line_8.u_truss[{1}:{-1}]",
    src_indices=line8_node_tags[1:-1],
)
model.link(
    "ps_node_src.v",
    f"node_src_line_8.v_truss[{1}:{-1}]",
    src_indices=line8_node_tags[1:-1],
)

# Build module
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

# Define the data vector
data = model.get_data_vector()

# Define deltas you want in the inner mesh
dx5_dx4 = 0.8  # Right edge displacment in x
dy5_dy6 = -0.5  # Top edge displacement in y
dx6_dx7 = 1.0  # Left edge displacement in x
dy7_dy4 = -0.1  # Bottom edge displacxement in y

# Set the problem data for line 7
data["node_src_line_7.x_coord"] = X[line7_node_tags, 0]
data["node_src_line_7.y_coord"] = X[line7_node_tags, 1]
data["dirichlet_line_7_y.offset[0]"] = dy5_dy6  # node 6
data["dirichlet_line_7_y.offset[-1]"] = dy7_dy4  # node 7
data["dirichlet_line_7_x.offset[:]"] = dx6_dx7

# Set the problem data for line 5
data["node_src_line_5.x_coord"] = X[line5_node_tags, 0]
data["node_src_line_5.y_coord"] = X[line5_node_tags, 1]
data["dirichlet_line_5_y.offset[0]"] = dy7_dy4
data["dirichlet_line_5_y.offset[-1]"] = dy5_dy6
data["dirichlet_line_5_x.offset[:]"] = dx5_dx4

# Set the problem data for line 8
data["node_src_line_8.x_coord"] = X[line8_node_tags, 0]
data["node_src_line_8.y_coord"] = X[line8_node_tags, 1]
data["dirichlet_line_8_x.offset[0]"] = dx6_dx7
data["dirichlet_line_8_x.offset[-1]"] = dx5_dx4
data["dirichlet_line_8_y.offset[:]"] = dy7_dy4

# Set the problem data for line 6
data["node_src_line_6.x_coord"] = X[line6_node_tags, 0]
data["node_src_line_6.y_coord"] = X[line6_node_tags, 1]
data["dirichlet_line_6_x.offset[0]"] = dx5_dx4
data["dirichlet_line_6_x.offset[-1]"] = dx6_dx7
data["dirichlet_line_6_y.offset[:]"] = dy5_dy6

# Data for plane stress model
data["ps_node_src.x_coord"] = X[:, 0]
data["ps_node_src.y_coord"] = X[:, 1]

# Setup the problem
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
# utils.plot_matrix(csr_mat.todense())
# plt.show()

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans

# # Extract solutions
# line7_u = ans_local.get_array()[model.get_indices("node_src_line_7.u_truss")]
# line7_v = ans_local.get_array()[model.get_indices("node_src_line_7.v_truss")]

# line5_u = ans_local.get_array()[model.get_indices("node_src_line_5.u_truss")]
# line5_v = ans_local.get_array()[model.get_indices("node_src_line_5.v_truss")]

# line8_u = ans_local.get_array()[model.get_indices("node_src_line_8.u_truss")]
# line8_v = ans_local.get_array()[model.get_indices("node_src_line_8.v_truss")]

# line6_u = ans_local.get_array()[model.get_indices("node_src_line_6.u_truss")]
# line6_v = ans_local.get_array()[model.get_indices("node_src_line_6.v_truss")]

# # Iniital coordinates for each line
# line7_x0 = X[line7_node_tags, 0]
# line7_y0 = X[line7_node_tags, 1]

# line8_x0 = X[line8_node_tags, 0]
# line8_y0 = X[line8_node_tags, 1]

# line5_x0 = X[line5_node_tags, 0]
# line5_y0 = X[line5_node_tags, 1]

# line6_x0 = X[line6_node_tags, 0]
# line6_y0 = X[line6_node_tags, 1]

# # Plot
# fig, ax = plt.subplots()

# ax.plot(line7_x0, line7_y0, "k-")
# ax.plot(line8_x0, line8_y0, "k-")
# ax.plot(line5_x0, line5_y0, "k-")
# ax.plot(line6_x0, line6_y0, "k-")

# ax.plot(line7_x0 + line7_u, line7_y0 + line7_v, "r-")
# ax.plot(line8_x0 + line8_u, line8_y0 + line8_v, "r-")
# ax.plot(line5_x0 + line5_u, line5_y0 + line5_v, "r-")
# ax.plot(line6_x0 + line6_u, line6_y0 + line6_v, "r-")
# ax.set_aspect("equal")

# Plot the new and old mesh
fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 6))
# OLD mesh
X_old = X.copy()
X_old[:, 0] -= 10
utils.plot_region(ax2[0], X_old, [conn_surface1], color="#7BE7FF", label="Surface 1")
utils.plot_region(ax2[0], X_old, [conn_surface2], color="#BF7BFF", label="Surface 2")
ax2[0].set_title("Old Mesh")

# NEW mesh
u = ans_local.get_array()[model.get_indices("ps_node_src.u")]
v = ans_local.get_array()[model.get_indices("ps_node_src.v")]
X_new = X.copy()
X_new[:, 0] += u
X_new[:, 1] += v

utils.plot_region(ax2[1], X_new, [conn_surface1], color="#7BE7FF", label="Surface 1")
utils.plot_region(ax2[1], X_new, [conn_surface2], color="#BF7BFF", label="Surface 2")
ax2[1].set_title("New Mesh")

xmin_old = np.min(X_old[:, 0])
xmax_old = np.max(X_old[:, 0])
ymin_old = np.min(X_old[:, 1])
ymax_old = np.max(X_old[:, 1])

xmin_new = np.min(X_new[:, 0])
xmax_new = np.max(X_new[:, 0])
ymin_new = np.min(X_new[:, 1])
ymax_new = np.max(X_new[:, 1])

ax2[0].set_xlim(xmin_old, xmax_old)
ax2[0].set_ylim(ymin_old, ymax_old)
ax2[1].set_xlim(xmin_new, xmax_new)
ax2[1].set_ylim(ymin_new, ymax_new)
fig2.tight_layout()
plt.savefig("mesh_morph_demo.jpg", dpi=800)
plt.show()
