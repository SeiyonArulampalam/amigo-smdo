import fem, basis
import amigo as am
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt


class ContinuityBc(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("u1", value=1.0)
        self.add_input("u2", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        self.objective["obj"] = (self.inputs["u1"] - self.inputs["u2"]) * self.inputs[
            "lam"
        ]
        return


def phase_currents(alpha, num_mag, Jz_peak):
    """
    Compute the current in phase 1 of the electrical winding.

    Parameters
    ----------
    alpha : Electrical rotation of shaft (rad)
    num_mag : Total number of magnets
    Jz_peak : Maximum current density of a single strand (A/m2)

    Returns
    -------
    phase1 : Current in phase 1
    phase2 : Current in phase 2
    phase3 : Current in phase 3
    """
    k = np.pi / (2 * np.pi / num_mag)
    offset120 = 120 * (np.pi / 180)
    offset240 = 240 * (np.pi / 180)
    phase1 = Jz_peak * np.cos(k * alpha)
    phase2 = Jz_peak * np.cos(k * alpha + offset120)
    phase3 = Jz_peak * np.cos(k * alpha + offset240)
    return phase1, phase2, phase3


# Geometry
airgap = 1.0e-3
tooth_tip_thickness = 1.0e-3
copper_slot_height = 35e-3
magnet_thickness = 10e-3
back_iron_thickness = 20.0e-3
y_offset1 = 0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap
y_offset2 = back_iron_thickness + 0.5 * airgap + magnet_thickness
y_offset = y_offset1 + y_offset2

# Fixed Values
npts_airgap = 240
num_mag = 10
slide_number = 0

# Compute current in each winding phase
Jz_peak = 0.0
phase1, phase2, phase3 = phase_currents(
    alpha=(2 * np.pi / npts_airgap) * slide_number,
    num_mag=num_mag,
    Jz_peak=Jz_peak,
)

dir = "/Users/seiyonarulampalam/git/amigo-smdo/examples/axial_flux_motor/"
meshes = {
    "outer_rotor": fem.Mesh(dir + "outter_rotor.inp"),
    "stator": fem.Mesh(dir + "stator.inp"),
    "inner_rotor": fem.Mesh(dir + "inner_rotor.inp"),
}

dirichlet_bcs = {
    "outer_rotor": {
        "dirichlet_outer": {
            "type": "dirichlet",
            "target": "LINE56",
            "input": ["u"],
            "start": False,
            "end": False,
        }
    },
    "inner_rotor": {
        "dirichlet_inner": {
            "type": "dirichlet",
            "target": "LINE56",
            "input": ["u"],
            "start": False,
            "end": False,
        }
    },
    "stator": {},
}

symm_bcs = {
    "outer_rotor": {
        "symm_outer_1": {
            "input": ["u"],
            "start": True,
            "end": True,
            "target": ["LINE55", "LINE57"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
        "symm_outer_2": {
            "input": ["u"],
            "start": False,
            "end": False,
            "target": ["LINE54", "LINE52"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
        "symm_outer_3": {
            "input": ["u"],
            "start": True,
            "end": True,
            "target": ["LINE69", "LINE70"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
    },
    "inner_rotor": {
        "symm_inner_1": {
            "input": ["u"],
            "start": True,
            "end": True,
            "target": ["LINE55", "LINE57"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
        "symm_inner_2": {
            "input": ["u"],
            "start": False,
            "end": False,
            "target": ["LINE54", "LINE52"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
        "symm_inner_3": {
            "input": ["u"],
            "start": True,
            "end": True,
            "target": ["LINE69", "LINE70"],
            "flip": [False, True],
            "scale": [-1.0, 1.0],
        },
    },
    "stator": {
        "symm_stator_1": {
            "input": ["u"],
            "start": True,
            "end": False,
            "target": ["LINE211", "LINE210"],
            "flip": [False, False],
            "scale": [-1.0, 1.0],
        },
        "symm_stator_2": {
            "input": ["u"],
            "start": False,
            "end": False,
            "target": ["LINE97", "LINE122"],
            "flip": [False, False],
            "scale": [-1.0, 1.0],
        },
        "symm_stator_3": {
            "input": ["u"],
            "start": False,
            "end": True,
            "target": ["LINE212", "LINE214"],
            "flip": [False, False],
            "scale": [-1.0, 1.0],
        },
    },
}

weakforms_outer_rotor = {
    "name": "outer_rotor_wfs",
    "SURFACE1": fem.weakform_back_iron,
    "SURFACE2": fem.weakform_SN_Magnet,
    "SURFACE3": fem.weakform_NS_Magnet,
    "SURFACE4": fem.weakform_SN_Magnet,
    "SURFACE5": fem.weakform_NS_Magnet,
    "SURFACE6": fem.weakform_SN_Magnet,
    "SURFACE7": fem.weakform_NS_Magnet,
    "SURFACE8": fem.weakform_SN_Magnet,
    "SURFACE9": fem.weakform_NS_Magnet,
    "SURFACE10": fem.weakform_SN_Magnet,
    "SURFACE11": fem.weakform_NS_Magnet,
    "SURFACE12": fem.weakform_air,
    "SURFACE13": fem.weakform_air,
    "SURFACE14": fem.weakform_air,
    "SURFACE15": fem.weakform_air,
    "SURFACE16": fem.weakform_air,
    "SURFACE17": fem.weakform_air,
    "SURFACE18": fem.weakform_air,
    "SURFACE19": fem.weakform_air,
    "SURFACE20": fem.weakform_air,
    "SURFACE21": fem.weakform_air,
    "SURFACE22": fem.weakform_air,
    "SURFACE23": fem.weakform_air,
}

weakforms_inner_rotor = {
    "name": "inner_rotor_wfs",
    "SURFACE1": fem.weakform_back_iron,
    "SURFACE2": fem.weakform_NS_Magnet,
    "SURFACE3": fem.weakform_SN_Magnet,
    "SURFACE4": fem.weakform_NS_Magnet,
    "SURFACE5": fem.weakform_SN_Magnet,
    "SURFACE6": fem.weakform_NS_Magnet,
    "SURFACE7": fem.weakform_SN_Magnet,
    "SURFACE8": fem.weakform_NS_Magnet,
    "SURFACE9": fem.weakform_SN_Magnet,
    "SURFACE10": fem.weakform_NS_Magnet,
    "SURFACE11": fem.weakform_SN_Magnet,
    "SURFACE12": fem.weakform_air,
    "SURFACE13": fem.weakform_air,
    "SURFACE14": fem.weakform_air,
    "SURFACE15": fem.weakform_air,
    "SURFACE16": fem.weakform_air,
    "SURFACE17": fem.weakform_air,
    "SURFACE18": fem.weakform_air,
    "SURFACE19": fem.weakform_air,
    "SURFACE20": fem.weakform_air,
    "SURFACE21": fem.weakform_air,
    "SURFACE22": fem.weakform_air,
    "SURFACE23": fem.weakform_air,
}

weakforms_stator = {
    "name": "stator_wfs",
    "SURFACE1": fem.weakform_coils,
    "SURFACE2": fem.weakform_coils,
    "SURFACE3": fem.weakform_coils,
    "SURFACE4": fem.weakform_coils,
    "SURFACE5": fem.weakform_coils,
    "SURFACE6": fem.weakform_coils,
    "SURFACE7": fem.weakform_coils,
    "SURFACE8": fem.weakform_coils,
    "SURFACE9": fem.weakform_coils,
    "SURFACE10": fem.weakform_coils,
    "SURFACE11": fem.weakform_coils,
    "SURFACE12": fem.weakform_coils,
    "SURFACE13": fem.weakform_coils,
    "SURFACE14": fem.weakform_coils,
    "SURFACE15": fem.weakform_coils,
    "SURFACE16": fem.weakform_coils,
    "SURFACE17": fem.weakform_coils,
    "SURFACE18": fem.weakform_coils,
    "SURFACE19": fem.weakform_coils,
    "SURFACE20": fem.weakform_coils,
    "SURFACE21": fem.weakform_coils,
    "SURFACE22": fem.weakform_coils,
    "SURFACE23": fem.weakform_coils,
    "SURFACE24": fem.weakform_coils,
    "SURFACE25": fem.weakform_stator_iron,
    "SURFACE26": fem.weakform_stator_iron,
    "SURFACE27": fem.weakform_stator_iron,
    "SURFACE28": fem.weakform_stator_iron,
    "SURFACE29": fem.weakform_stator_iron,
    "SURFACE30": fem.weakform_stator_iron,
    "SURFACE31": fem.weakform_stator_iron,
    "SURFACE32": fem.weakform_stator_iron,
    "SURFACE33": fem.weakform_stator_iron,
    "SURFACE34": fem.weakform_stator_iron,
    "SURFACE35": fem.weakform_stator_iron,
    "SURFACE36": fem.weakform_stator_iron,
    "SURFACE37": fem.weakform_stator_iron,
    "SURFACE38": fem.weakform_air,
    "SURFACE39": fem.weakform_air,
    "SURFACE40": fem.weakform_air,
    "SURFACE41": fem.weakform_air,
    "SURFACE42": fem.weakform_air,
    "SURFACE43": fem.weakform_air,
    "SURFACE44": fem.weakform_air,
    "SURFACE45": fem.weakform_air,
    "SURFACE46": fem.weakform_air,
    "SURFACE47": fem.weakform_air,
    "SURFACE48": fem.weakform_air,
    "SURFACE49": fem.weakform_air,
    "SURFACE50": fem.weakform_air,
    "SURFACE51": fem.weakform_air,
    "SURFACE52": fem.weakform_air,
    "SURFACE53": fem.weakform_air,
    "SURFACE54": fem.weakform_air,
    "SURFACE55": fem.weakform_air,
    "SURFACE56": fem.weakform_air,
    "SURFACE57": fem.weakform_air,
    "SURFACE58": fem.weakform_air,
    "SURFACE59": fem.weakform_air,
    "SURFACE60": fem.weakform_air,
    "SURFACE61": fem.weakform_air,
    "SURFACE62": fem.weakform_air,
    "SURFACE63": fem.weakform_air,
}

weakforms = {
    "outer_rotor": weakforms_outer_rotor,
    "stator": weakforms_stator,
    "inner_rotor": weakforms_inner_rotor,
}

# Initialize the spaces (same for all domains)
soln_space = basis.SolutionSpace({"u": "H1"})
data_space = basis.SolutionSpace({"Jz": "const"})
geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

# Define the global amigo model
main = am.Model("main")

# Create an amigo model for each mesh
for mesh_name, mesh in meshes.items():
    problem = fem.Problem(
        mesh,
        soln_space,
        weakforms[mesh_name],
        data_space=data_space,
        geo_space=geo_space,
        dirichlet_bc_map=dirichlet_bcs[mesh_name],
        sym_bc_map=symm_bcs[mesh_name],
        ndim=2,
    )
    model = problem.create_model(mesh_name)
    main.add_model(mesh_name, model)

# Extract the shared edges between all 3 meshes
outer_rotor_mesh = meshes.get("outer_rotor")
stator_mesh = meshes.get("stator")
inner_rotor_mesh = meshes.get("inner_rotor")

inner_rotor_line_53 = inner_rotor_mesh.get_bc_nodes("LINE53", "T3D2", flip=True)
stator_line_213 = stator_mesh.get_bc_nodes("LINE213", "T3D2", flip=False)
stator_line_209 = stator_mesh.get_bc_nodes("LINE209", "T3D2", flip=False)
outer_rotor_line_53 = outer_rotor_mesh.get_bc_nodes("LINE53", "T3D2", flip=True)

# Link the main model nodes
shared_outer_53 = outer_rotor_line_53[1:-1]
shared_stator_209 = stator_line_209[1:-1]
shared_stator_213 = stator_line_213[1:-1]
shared_inner_53 = inner_rotor_line_53[1:-1]

main.add_component("stator_outer_ag", len(shared_outer_53), ContinuityBc())
main.add_component("stator_inner_ag", len(shared_inner_53), ContinuityBc())

for i in range(len(shared_outer_53)):
    main.link(
        f"outer_rotor.src_soln.u[{shared_outer_53[i]}]",
        f"stator_outer_ag.u1[{i}]",
    )
    main.link(
        f"stator.src_soln.u[{shared_stator_209[i]}]",
        f"stator_outer_ag.u2[{i}]",
    )

for i in range(len(shared_inner_53)):
    main.link(
        f"inner_rotor.src_soln.u[{shared_inner_53[i]}]",
        f"stator_inner_ag.u1[{i}]",
    )
    main.link(
        f"stator.src_soln.u[{shared_stator_213[i]}]",
        f"stator_inner_ag.u2[{i}]",
    )
# Build the model
main.build_module()
main.initialize()
p = main.get_problem()

# Set the problem data
data = main.get_data_vector()
data["outer_rotor.src_geo.x"] = meshes["outer_rotor"].X[:, 0]
data["outer_rotor.src_geo.y"] = meshes["outer_rotor"].X[:, 1]
data["stator.src_geo.x"] = meshes["stator"].X[:, 0]
data["stator.src_geo.y"] = meshes["stator"].X[:, 1]
data["inner_rotor.src_geo.x"] = meshes["inner_rotor"].X[:, 0]
data["inner_rotor.src_geo.y"] = meshes["inner_rotor"].X[:, 1]


phase1_pos_surfaces = ["SURFACE1", "SURFACE2", "SURFACE12", "SURFACE15"]
phase1_neg_surfaces = ["SURFACE3", "SURFACE13", "SURFACE14", "SURFACE24"]

phase2_pos_surfaces = ["SURFACE4", "SURFACE7", "SURFACE17", "SURFACE18"]
phase2_neg_surfaces = ["SURFACE5", "SURFACE6", "SURFACE16", "SURFACE19"]

phase3_pos_surfaces = ["SURFACE9", "SURFACE10", "SURFACE20", "SURFACE23"]
phase3_neg_surfaces = ["SURFACE8", "SURFACE11", "SURFACE21", "SURFACE22"]

for s in phase1_pos_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = phase1

for s in phase1_neg_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = -phase1

for s in phase2_pos_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = phase2

for s in phase2_neg_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = -phase2

for s in phase3_pos_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = phase3

for s in phase3_neg_surfaces:
    idx = int(s.replace("SURFACE", "")) - 1
    data[f"stator.src_data.Jz[{idx}]"] = -phase3

mat = p.create_matrix()
alpha = 1.0
x = p.create_vector()
ans = p.create_vector()
g = p.create_vector()
rhs = p.create_vector()
p.hessian(alpha, x, mat)
p.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
u_outer_rotor = ans_local.get_array()[main.get_indices("outer_rotor.src_soln.u")]
u_stator = ans_local.get_array()[main.get_indices("stator.src_soln.u")]
u_inner_rotor = ans_local.get_array()[main.get_indices("inner_rotor.src_soln.u")]

# Plot results

combined = np.concatenate((u_outer_rotor, u_stator, u_inner_rotor))
min_val = combined.min()
max_val = combined.max()
print(min_val, max_val)

fig, ax = plt.subplots()
meshes["outer_rotor"].plot(
    u_outer_rotor,
    x_offset=0.0,
    y_offset=y_offset,
    ax=ax,
    min_level=min_val,
    max_level=max_val,
)
meshes["stator"].plot(
    u_stator,
    x_offset=0.0,
    y_offset=0.0,
    ax=ax,
    min_level=min_val,
    max_level=max_val,
)
meshes["inner_rotor"].plot(
    u_inner_rotor,
    x_offset=0.0,
    y_offset=-y_offset,
    ax=ax,
    min_level=min_val,
    max_level=max_val,
)
plt.show()
