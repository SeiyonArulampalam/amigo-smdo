import pyvista as pv
import numpy as np

# -----------------------------
# Shaft parameters
# -----------------------------
outer_radius = 1.0
inner_radius = 0.4
shaft_height = 0.1

# Base hollow cylinder
outer_shaft = pv.Cylinder(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    radius=outer_radius,
    height=shaft_height,
    resolution=200,
)
inner_shaft = pv.Cylinder(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    radius=inner_radius,
    height=shaft_height * 1.1,
    resolution=200,
)
shaft = outer_shaft.triangulate().boolean_difference(inner_shaft.triangulate())

# -----------------------------
# Magnet parameters
# -----------------------------
n_magnet = 10  # number of magnets
gap_fraction = 0.35  # 10% of each segment is empty
theta_span = 2 * np.pi / n_magnet
tooth_span = theta_span * (1 - gap_fraction)
magnet_height = 0.1
n_pts = 50  # points per arc

# -----------------------------
# Build all magnet
# -----------------------------
all_magnet = []

for i in range(n_magnet):
    # Offset angle for this tooth
    theta = np.linspace(-tooth_span / 2, tooth_span / 2, n_pts) + i * theta_span

    # Bottom arcs
    x_outer_bot = outer_radius * np.cos(theta)
    y_outer_bot = outer_radius * np.sin(theta)
    x_inner_bot = inner_radius * np.cos(theta)
    y_inner_bot = inner_radius * np.sin(theta)
    z_bot = np.full_like(theta, shaft_height / 2)

    # Top arcs
    x_outer_top = x_outer_bot.copy()
    y_outer_top = y_outer_bot.copy()
    x_inner_top = x_inner_bot.copy()
    y_inner_top = y_inner_bot.copy()
    z_top = np.full_like(theta, shaft_height / 2 + magnet_height)

    # Stack points: outer bottom, inner bottom, outer top, inner top
    points = np.column_stack(
        [
            np.concatenate([x_outer_bot, x_inner_bot, x_outer_top, x_inner_top]),
            np.concatenate([y_outer_bot, y_inner_bot, y_outer_top, y_inner_top]),
            np.concatenate([z_bot, z_bot, z_top, z_top]),
        ]
    )

    # Build quad faces
    faces = []

    # Outer wall
    for j in range(n_pts - 1):
        faces.extend([4, j, j + 1, n_pts * 2 + j + 1, n_pts * 2 + j])
    # Inner wall
    for j in range(n_pts - 1):
        faces.extend([4, n_pts + j, n_pts + j + 1, n_pts * 3 + j + 1, n_pts * 3 + j])
    # Side walls
    faces.extend([4, 0, n_pts, n_pts * 3, n_pts * 2])
    faces.extend([4, n_pts - 1, n_pts * 2 - 1, n_pts * 4 - 1, n_pts * 3 - 1])
    # Top and bottom
    for j in range(n_pts - 1):
        faces.extend(
            [4, n_pts * 2 + j, n_pts * 2 + j + 1, n_pts * 3 + j + 1, n_pts * 3 + j]
        )
        faces.extend([4, j, j + 1, n_pts + j + 1, n_pts + j])

    faces = np.array(faces)
    tooth = pv.PolyData(points, faces)
    all_magnet.append(tooth)

# -----------------------------
# Stator coil parameters
# -----------------------------
n_coil = 12  # number of coils (independent of magnets)
coil_thickness = 0.08  # physical thickness (shaft_height) of each layer
layer_spacing = 0.0  # gap between coil layers
stator_offset = 0.4  # gap between the rotor and coil
layer_colors = ["#878787", "orange", "#878787"]

stator_coils = []

# Use first magnet as template
ref_tooth = all_magnet[0]
ref_points = ref_tooth.points
ref_faces = ref_tooth.faces

z_top_magnet = ref_points[:, 2].max()
z_bottom_magnet = ref_points[:, 2].min()

for i in range(n_coil):
    angle = i * 2 * np.pi / n_coil

    # 2x2 rotation matrix for x-y
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    for j, color in enumerate(layer_colors):
        # Start each layer above the magnet + spacing between layers
        z_start = (
            z_top_magnet + 0.002 + j * (coil_thickness + layer_spacing) + stator_offset
        )

        # Duplicate and transform
        new_points = ref_points.copy()
        new_points[:, :2] = (R @ new_points[:, :2].T).T

        # Normalize coil shaft_height and scale to desired thickness
        z_local = new_points[:, 2] - z_bottom_magnet
        z_local = z_local / (z_top_magnet - z_bottom_magnet) * coil_thickness
        new_points[:, 2] = z_start + z_local

        coil_layer = pv.PolyData(new_points, ref_faces)
        stator_coils.append((coil_layer, color))

# -----------------------------
# Mirror only shaft and magnets, then move up by +1 in Z
# -----------------------------
mirror_offset = (
    shaft_height + magnet_height + coil_thickness * 3 + 2 * stator_offset
)  # amount to move the mirrored geometry upward

# Mirror and shift shaft
mirrored_shell = shaft.copy()
mirrored_shell.points[:, 2] *= -1
mirrored_shell.points[:, 2] += mirror_offset

# Mirror and shift magnets
mirrored_magnets = []
for tooth in all_magnet:
    mirrored = tooth.copy()
    mirrored.points[:, 2] *= -1
    mirrored.points[:, 2] += mirror_offset
    mirrored_magnets.append(mirrored)

# -----------------------------
# Slice parameters
# -----------------------------
slice_ro = 0.68
slice_ri = 0.67
z1 = -0.2
z2 = 2 * shaft_height + magnet_height + coil_thickness * 3 + 2 * stator_offset
height = z2 - z1
center_z = (z1 + z2) / 2

# Outer and inner cylinders
outer_slice = pv.Cylinder(
    center=(0, 0, center_z),  # centered between z1 and z2
    direction=(0, 0, 1),
    radius=slice_ro,
    height=height,
    resolution=200,
)

inner_slice = pv.Cylinder(
    center=(0, 0, center_z),
    direction=(0, 0, 1),
    radius=slice_ri,
    height=height * 1.1,
    resolution=200,
)

slice = outer_slice.triangulate().boolean_difference(inner_slice.triangulate())

# Clip the cylinder at z1 and z2 to limit height
slice = slice.clip(normal=[0, 0, -1], origin=[0, 0, z1])
slice = slice.clip(normal=[0, 0, 1], origin=[0, 0, z2])

# -----------------------------
# Plot
# -----------------------------
plotter = pv.Plotter()
plotter.add_mesh(shaft, color="grey", show_edges=False, opacity=1.0)

colors = ["blue", "red"]
for i, tooth in enumerate(all_magnet):
    plotter.add_mesh(tooth, color=colors[i % 2], show_edges=False)

# Mirrored shaft + magnets
colors = ["red", "blue"]
plotter.add_mesh(mirrored_shell, color="grey", show_edges=False, opacity=1.0)
for i, tooth in enumerate(mirrored_magnets):
    plotter.add_mesh(tooth, color=colors[i % 2], show_edges=False)

# Coils
for coil, color in stator_coils:
    plotter.add_mesh(coil, color=color, show_edges=False, smooth_shading=True)

plotter.add_mesh(slice, color="#6AFC2C", opacity=0.4, smooth_shading=True)

light = pv.Light(position=(-1, 1, 1), color="white")
light.positional = True
plotter.add_light(light)
plotter.show()
