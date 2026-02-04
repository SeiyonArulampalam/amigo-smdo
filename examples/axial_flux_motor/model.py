import numpy as np
import matplotlib.pylab as plt
import motor_controller
import utils  # Misc helper funcs
from parser import InpParser  # mesh parser
import create_mesh  # gmsh class
import plot_motor  # Plot motor geometry
import argparse
import amigo as am
from scipy.sparse.linalg import spsolve
from comps import (
    Maxwell,
    Coil,
    Magnets,
    DirichletBc,
    AntiSymmBc,
    SymmBc,
    NodeSource,
    MaterialSource,
)
import compute_magnetic_field
import compute_forces as compute_forces
import compute_losses
import matplotlib.ticker as ticker
import compute_mass

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None

# Define parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--radial_slice",
    default=None,
    type=float,
    help="Define slice ratio [0,1]",
)

parser.add_argument(
    "--radial_slice_mean_diam",
    action="store_true",
    help="Run model at the mean diameter",
)

parser.add_argument(
    "--slide_number",
    default=0,
    type=int,
    help="Select slide number to define the shaft rotation",
)
parser.add_argument(
    "--plot_potential",
    action="store_true",
    help="Plot the solution field and save the figure",
)
parser.add_argument(
    "--plot_geometry",
    action="store_true",
    help="Plot the motor geometry",
)

parser.add_argument(
    "--plot_flux",
    action="store_true",
    help="Plot the magentic flux",
)

parser.add_argument(
    "--plot_force_calc",
    action="store_true",
    help="Plot the data in the force calculation",
)

parser.add_argument(
    "--plot_ag_flux",
    action="store_true",
    help="Plot the magentic flux in the airgap only",
)

parser.add_argument(
    "--plot_loss",
    action="store_true",
    help="Plot the magentic flux in the airgap only",
)


parser.add_argument(
    "--mesh",
    action="store_true",
    help="Generate new GMSH .inp files for the motor geometry",
)
parser.add_argument(
    "--gmsh_popup",
    action="store_true",
    default=False,
    help="pop up the gmsh interactive window",
)
parser.add_argument(
    "--build",
    dest="build",
    action="store_true",
    default=False,
    help="Enable building",
)
parser.add_argument(
    "--with-openmp",
    dest="use_openmp",
    action="store_true",
    default=False,
    help="Enable OpenMP",
)
parser.add_argument(
    "--order-type",
    choices=["amd", "nd", "natural"],
    default="nd",
    help="Ordering strategy to use (default: amd)",
)
parser.add_argument(
    "--order-for-block",
    dest="order_for_block",
    action="store_true",
    default=False,
    help="Order for 2x2 block KKT matrix",
)
parser.add_argument(
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
parser.add_argument(
    "--with-debug",
    dest="use_debug",
    action="store_true",
    default=False,
    help="Enable debug compiler flags",
)
args = parser.parse_args()

############################
# Define MotorCAD Parameters
############################
stator_lam_dia = 100e-3
stator_bore = 50e-3
rotor_lam_di = 100e-3
shaft_diam = 25e-3

###########################
# Define the motor geometry
###########################
num_mag = 10  #! Fixed
mesh_refinement = 3e-3
npts_airgap = 240
slide_number = args.slide_number

airgap = 1.0e-3
tooth_tip_thickness = 1.0e-3
copper_slot_height = 35e-3
magnet_thickness = 10e-3
back_iron_thickness = 20.0e-3

Di = stator_bore
Do = stator_lam_dia
Ri = 0.5 * Di  # inner radius
Ro = 0.5 * Do  # outer radius

stack_length = (
    2 * airgap
    + 2 * magnet_thickness
    + 2 * back_iron_thickness
    + 2 * tooth_tip_thickness
    + copper_slot_height
)

if args.radial_slice is not None:
    # Custom defined radial slice
    R = (Ro - Ri) * args.radial_slice + Ri  # radius of analysis

elif args.radial_slice_mean_diam:
    # Radial slice defined by the mean diameter
    D_avg = (Di + Do) * 0.5
    R = 0.5 * D_avg

else:
    raise Exception("Err: Radial slice was not defined")

theta_mi = np.deg2rad(30)  # magnet inner span
theta_mo = np.deg2rad(30)  # magnet outer span
theta_bi = np.deg2rad(28)  # bell inner span
theta_bo = np.deg2rad(28)  # bell outer span
theta_ti = np.deg2rad(14.6)  # tooth inner span
theta_to = np.deg2rad(14.6)  # tooth outer span

# Functions for computing the span given a radial location
func_theta_m = lambda r: theta_mi + (theta_mo - theta_mi) * r
func_theta_b = lambda r: theta_bi + (theta_bo - theta_bi) * r
func_theta_t = lambda r: theta_ti + (theta_to - theta_ti) * r

theta_m = func_theta_m(R)  # magnet span
theta_b = func_theta_b(R)  # bell span
theta_t = func_theta_t(R)  # tooth span
total_length = 2 * np.pi * R  # total slice length

bell_width = R * theta_b  # arc length of the bell
tooth_width = R * theta_t  # arc length of the tooth
magnet_length = R * theta_m  # arc length of the magnet

# Raise errors for incorrect geometry
if R > Ro or R < Ri:
    print(f"R: {R}, Ro:{Ro}, Ri:{Ri}")
    raise Exception("ERROR: R is not defined correctly")

if theta_m > theta_mo or theta_m < theta_mi:
    print(f"theta_m: {theta_m}, theta_mo:{theta_mo}, theta_mi:{theta_mi}")
    raise Exception("ERROR: theta_m is not defined correctly")

if theta_b > theta_bo or theta_b < theta_bi:
    print(f"theta_b: {theta_b}, theta_bo:{theta_bo}, theta_bi:{theta_bi}")
    raise Exception("ERROR: theta_b is not defined correctly")

if theta_t > theta_to or theta_t < theta_ti:
    print(f"theta_t: {theta_t}, theta_to:{theta_to}, theta_ti:{theta_ti}")
    raise Exception("ERROR: theta_t is not defined correctly")

magnet_gap = (np.deg2rad(360) - theta_m * num_mag) / num_mag
if magnet_gap <= np.deg2rad(1):
    print(np.rad2deg(theta_m))
    print(np.rad2deg(theta_m * num_mag))
    print(f"magnet_gap: {magnet_gap} >= {np.deg2rad(1)}")
    raise Exception(f"ERROR: Gap between magnets")

bell_gap = (np.deg2rad(360) - theta_b * 12) / 12
if bell_gap <= np.deg2rad(1):
    print(f"bell_gap: {bell_gap} >= {np.deg2rad(1)}")
    raise Exception(f"ERROR: Gap between bell")

tooth_gap = (np.deg2rad(360) - theta_t * 12) / 12
if tooth_gap <= np.deg2rad(1):
    print(f"tooth_gap: {tooth_gap} >= {np.deg2rad(1)}")
    raise Exception(f"ERROR: Gap between tooth")

if theta_b <= theta_t:
    raise Exception("ERROR: Bell span is less than the tooth span")

########################
# Define operation point
########################
num_windings = 50
rpm = 5000.0  # RPM of motor
omega = rpm * np.pi / 30.0  # Angular vel (rad/s)
I_current = 20  # Current in single strand
I_rms = I_current / np.sqrt(2)  # RMS current
I_peak = I_current * num_windings  # Total current in slot
I_peak_rms = I_rms * num_windings  # Total RMS current in slot
fund_freq = rpm * num_mag / 120.0  # Hz
print(f"I_current:{I_current:.4f}")

# Multiply by 0.5 because a slot is half the area of the region
slot_fill_factor = 0.4
slot_width = (total_length - 12 * R * theta_ti) / 12
slot_area = (slot_width * copper_slot_height * 0.5) * slot_fill_factor
print(f"Slot Area: {slot_area:2e}")

# Peak current desnity in the slot region
Jz_peak = I_peak / slot_area
Jz_peak_rms = I_peak_rms / slot_area
print(f"Jz_peak: {Jz_peak:.2e}")
print(f"Jz_peak_rms: {Jz_peak_rms:.2e}")

# Compute the area
r_strand = 0.5 * (2.203e-3)
A_strand = np.pi * (r_strand**2) * num_windings
ao = Ro * theta_to * num_windings * A_strand
ai = Ri * theta_ti * num_windings * A_strand
A_cu = A_strand * 24  # + 12 * (ao + ai)
print(f"A_cu: {A_cu:2e}")

# Define the strength of the magnet
My = 1.0e6

#####################
# Material Properties
#####################
rho_iron = 7500.0
rho_magnet = 7500.0
mu_r_magnet = 1.05
mu_r_air = 1.0
mu_r_iron = 7000.0

###################
# Generate the mesh
###################
if args.mesh:
    mesh = create_mesh.AFPM_Mesh_12S5PP(
        total_length=total_length,
        airgap=airgap,
        copper_slot_height=copper_slot_height,
        tooth_tip_thickness=tooth_tip_thickness,
        bell_width=bell_width,
        tooth_width=tooth_width,
        magnet_length=magnet_length,
        magnet_thickness=magnet_thickness,
        back_iron_thickness=back_iron_thickness,
        mesh_refinement=mesh_refinement,
        npts_airgap=npts_airgap,
        gmsh_popup=args.gmsh_popup,
    )
    mesh.stator()  # stator.inp
    mesh.outter_rotor()  # outter_rotor.inp
    mesh.inner_rotor()  # inner_rotor.inp

###################
# Extract mesh conn
###################
# Define the offset for the y coordinates for each mesh
y_offset1 = 0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap
y_offset2 = back_iron_thickness + 0.5 * airgap + magnet_thickness
y_offset = y_offset1 + y_offset2

# Define the offset to slide the stator mesh to the right
if slide_number > npts_airgap:
    raise Exception("Slide number excedes npts in airgap")
slide_offset = (total_length / (npts_airgap - 1)) * slide_number

# Define the filenames for the mesh
fname_stator = "stator.inp"
fname_inner_rotor = "inner_rotor.inp"
fname_outter_rotor = "outter_rotor.inp"

# Instance of InpParser for stator, inner rotor, and outter rotor
parser_stator = InpParser()
parser_stator.parse_inp(fname_stator)

parser_inner_rotor = InpParser()
parser_inner_rotor.parse_inp(fname_inner_rotor)

parser_outter_rotor = InpParser()
parser_outter_rotor.parse_inp(fname_outter_rotor)

# Extract node coords for each mesh
X_stator = parser_stator.get_nodes()
X_inner_rotor = parser_inner_rotor.get_nodes()
X_outter_rotor = parser_outter_rotor.get_nodes()

# Update the node coordinates for the offset of the outter and inner rotors
X_inner_rotor[:, 1] -= y_offset
X_outter_rotor[:, 1] += y_offset

# Update the node coordinates of the stator to slide to the right
X_stator[:, 0] += slide_offset

#################################
# Stator connectivity information
#################################
# Get the connectivity for the slot regions
# conn_s1 = connectivity of slot 1 surface
stator_conn_s1 = parser_stator.get_conn("SURFACE1", "CPS3")
stator_conn_s2 = parser_stator.get_conn("SURFACE2", "CPS3")
stator_conn_s3 = parser_stator.get_conn("SURFACE3", "CPS3")
stator_conn_s4 = parser_stator.get_conn("SURFACE4", "CPS3")
stator_conn_s5 = parser_stator.get_conn("SURFACE5", "CPS3")
stator_conn_s6 = parser_stator.get_conn("SURFACE6", "CPS3")
stator_conn_s7 = parser_stator.get_conn("SURFACE7", "CPS3")
stator_conn_s8 = parser_stator.get_conn("SURFACE8", "CPS3")
stator_conn_s9 = parser_stator.get_conn("SURFACE9", "CPS3")
stator_conn_s10 = parser_stator.get_conn("SURFACE10", "CPS3")
stator_conn_s11 = parser_stator.get_conn("SURFACE11", "CPS3")
stator_conn_s12 = parser_stator.get_conn("SURFACE12", "CPS3")
stator_conn_s13 = parser_stator.get_conn("SURFACE13", "CPS3")
stator_conn_s14 = parser_stator.get_conn("SURFACE14", "CPS3")
stator_conn_s15 = parser_stator.get_conn("SURFACE15", "CPS3")
stator_conn_s16 = parser_stator.get_conn("SURFACE16", "CPS3")
stator_conn_s17 = parser_stator.get_conn("SURFACE17", "CPS3")
stator_conn_s18 = parser_stator.get_conn("SURFACE18", "CPS3")
stator_conn_s19 = parser_stator.get_conn("SURFACE19", "CPS3")
stator_conn_s20 = parser_stator.get_conn("SURFACE20", "CPS3")
stator_conn_s21 = parser_stator.get_conn("SURFACE21", "CPS3")
stator_conn_s22 = parser_stator.get_conn("SURFACE22", "CPS3")
stator_conn_s23 = parser_stator.get_conn("SURFACE23", "CPS3")
stator_conn_s24 = parser_stator.get_conn("SURFACE24", "CPS3")

# Stator teeth connectivity
# conn_t1 = connectivity for tooth 1 surface
stator_conn_t1 = parser_stator.get_conn("SURFACE25", "CPS3")
stator_conn_t2 = parser_stator.get_conn("SURFACE26", "CPS3")
stator_conn_t3 = parser_stator.get_conn("SURFACE27", "CPS3")
stator_conn_t4 = parser_stator.get_conn("SURFACE28", "CPS3")
stator_conn_t5 = parser_stator.get_conn("SURFACE29", "CPS3")
stator_conn_t6 = parser_stator.get_conn("SURFACE30", "CPS3")
stator_conn_t7 = parser_stator.get_conn("SURFACE31", "CPS3")
stator_conn_t8 = parser_stator.get_conn("SURFACE32", "CPS3")
stator_conn_t9 = parser_stator.get_conn("SURFACE33", "CPS3")
stator_conn_t10 = parser_stator.get_conn("SURFACE34", "CPS3")
stator_conn_t11 = parser_stator.get_conn("SURFACE35", "CPS3")
stator_conn_t12 = parser_stator.get_conn("SURFACE36", "CPS3")
stator_conn_t13 = parser_stator.get_conn("SURFACE37", "CPS3")

# Airgap connectivity
stator_conn_ag_outter = parser_stator.get_conn("SURFACE38", "CPS3")
stator_conn_ag_inner = parser_stator.get_conn("SURFACE39", "CPS3")

# Airgap connectivity between the stator teeth (outter)
stator_conn_ag_1 = parser_stator.get_conn("SURFACE40", "CPS3")
stator_conn_ag_2 = parser_stator.get_conn("SURFACE41", "CPS3")
stator_conn_ag_3 = parser_stator.get_conn("SURFACE42", "CPS3")
stator_conn_ag_4 = parser_stator.get_conn("SURFACE43", "CPS3")
stator_conn_ag_5 = parser_stator.get_conn("SURFACE44", "CPS3")
stator_conn_ag_6 = parser_stator.get_conn("SURFACE45", "CPS3")
stator_conn_ag_7 = parser_stator.get_conn("SURFACE46", "CPS3")
stator_conn_ag_8 = parser_stator.get_conn("SURFACE47", "CPS3")
stator_conn_ag_9 = parser_stator.get_conn("SURFACE48", "CPS3")
stator_conn_ag_10 = parser_stator.get_conn("SURFACE49", "CPS3")
stator_conn_ag_11 = parser_stator.get_conn("SURFACE50", "CPS3")
stator_conn_ag_12 = parser_stator.get_conn("SURFACE51", "CPS3")
stator_conn_ag_13 = parser_stator.get_conn("SURFACE52", "CPS3")
stator_conn_ag_14 = parser_stator.get_conn("SURFACE53", "CPS3")
stator_conn_ag_15 = parser_stator.get_conn("SURFACE54", "CPS3")
stator_conn_ag_16 = parser_stator.get_conn("SURFACE55", "CPS3")
stator_conn_ag_17 = parser_stator.get_conn("SURFACE56", "CPS3")
stator_conn_ag_18 = parser_stator.get_conn("SURFACE57", "CPS3")
stator_conn_ag_19 = parser_stator.get_conn("SURFACE58", "CPS3")
stator_conn_ag_20 = parser_stator.get_conn("SURFACE59", "CPS3")
stator_conn_ag_21 = parser_stator.get_conn("SURFACE60", "CPS3")
stator_conn_ag_22 = parser_stator.get_conn("SURFACE61", "CPS3")
stator_conn_ag_23 = parser_stator.get_conn("SURFACE62", "CPS3")
stator_conn_ag_24 = parser_stator.get_conn("SURFACE63", "CPS3")


# Line on the left edge of the stator (goes from bottom to top)
stator_conn_left_edge1 = parser_stator.get_conn("LINE212", "T3D2")
stator_conn_left_edge2 = parser_stator.get_conn("LINE97", "T3D2")
stator_conn_left_edge3 = parser_stator.get_conn("LINE211", "T3D2")
stator_conn_pbc_left = np.concatenate(
    (
        stator_conn_left_edge1.flatten(),
        stator_conn_left_edge2.flatten(),
        stator_conn_left_edge3.flatten(),
    ),
    axis=None,
)
# Turn into a single unique list of nodes preserving the order from bottom to top
stator_pbc_nodes_left = np.array(list(dict.fromkeys(stator_conn_pbc_left)))

# Line on the right edge of the stator (goes from bottom to top)
stator_conn_right_edge1 = parser_stator.get_conn("LINE214", "T3D2")
stator_conn_right_edge2 = parser_stator.get_conn("LINE122", "T3D2")
stator_conn_right_edge3 = parser_stator.get_conn("LINE210", "T3D2")
stator_conn_pbc_right = np.concatenate(
    (
        stator_conn_right_edge1.flatten(),
        stator_conn_right_edge2.flatten(),
        stator_conn_right_edge3.flatten(),
    ),
    axis=None,
)
# Turn into a single unique list of nodes preserving the order from bottom to top
stator_pbc_nodes_right = np.array(list(dict.fromkeys(stator_conn_pbc_right)))

# Line on the top edge of the stator (oriented from left to right)
stator_conn_top_edge = parser_stator.get_conn("LINE209", "T3D2")
stator_pbc_nodes_top = np.array(list(dict.fromkeys(stator_conn_top_edge.flatten())))

# Line on the bottom edge of the stator
stator_conn_bottom_edge = parser_stator.get_conn("LINE213", "T3D2")
stator_pbc_nodes_bottom = np.array(
    list(dict.fromkeys(stator_conn_bottom_edge.flatten()))
)

# Total number of elements in the stator
# Collect all connectivity arrays in a list
stator_conns = [
    stator_conn_s1,
    stator_conn_s2,
    stator_conn_s3,
    stator_conn_s4,
    stator_conn_s5,
    stator_conn_s6,
    stator_conn_s7,
    stator_conn_s8,
    stator_conn_s9,
    stator_conn_s10,
    stator_conn_s11,
    stator_conn_s12,
    stator_conn_s13,
    stator_conn_s14,
    stator_conn_s15,
    stator_conn_s16,
    stator_conn_s17,
    stator_conn_s18,
    stator_conn_s19,
    stator_conn_s20,
    stator_conn_s21,
    stator_conn_s22,
    stator_conn_s23,
    stator_conn_s24,
    stator_conn_t1,
    stator_conn_t2,
    stator_conn_t3,
    stator_conn_t4,
    stator_conn_t5,
    stator_conn_t6,
    stator_conn_t7,
    stator_conn_t8,
    stator_conn_t9,
    stator_conn_t10,
    stator_conn_t11,
    stator_conn_t12,
    stator_conn_t13,
    stator_conn_ag_inner,
    stator_conn_ag_outter,
    stator_conn_ag_1,
    stator_conn_ag_2,
    stator_conn_ag_3,
    stator_conn_ag_4,
    stator_conn_ag_5,
    stator_conn_ag_6,
    stator_conn_ag_7,
    stator_conn_ag_8,
    stator_conn_ag_9,
    stator_conn_ag_10,
    stator_conn_ag_11,
    stator_conn_ag_12,
    stator_conn_ag_13,
    stator_conn_ag_14,
    stator_conn_ag_15,
    stator_conn_ag_16,
    stator_conn_ag_17,
    stator_conn_ag_18,
    stator_conn_ag_19,
    stator_conn_ag_20,
    stator_conn_ag_21,
    stator_conn_ag_22,
    stator_conn_ag_23,
    stator_conn_ag_24,
]

# Compute total number of elements and elements
nelems_stator = sum(conn.shape[0] for conn in stator_conns)
nnodes_stator = X_stator.shape[0]
print("\nNELEMS stator (COMPUTED):", nelems_stator)


#######################################
# Outter rotor connectivity information
#######################################
# Get the connectivity for each of the magnets
# conn_m1 = connectivity for magnet 1 (starting from far left)
outter_rotor_conn_m1 = parser_outter_rotor.get_conn("SURFACE11", "CPS3")
outter_rotor_conn_m2 = parser_outter_rotor.get_conn("SURFACE10", "CPS3")
outter_rotor_conn_m3 = parser_outter_rotor.get_conn("SURFACE9", "CPS3")
outter_rotor_conn_m4 = parser_outter_rotor.get_conn("SURFACE8", "CPS3")
outter_rotor_conn_m5 = parser_outter_rotor.get_conn("SURFACE7", "CPS3")
outter_rotor_conn_m6 = parser_outter_rotor.get_conn("SURFACE6", "CPS3")
outter_rotor_conn_m7 = parser_outter_rotor.get_conn("SURFACE5", "CPS3")
outter_rotor_conn_m8 = parser_outter_rotor.get_conn("SURFACE4", "CPS3")
outter_rotor_conn_m9 = parser_outter_rotor.get_conn("SURFACE3", "CPS3")
outter_rotor_conn_m10 = parser_outter_rotor.get_conn("SURFACE2", "CPS3")

# Get the back iron connectivity
outter_rotor_conn_back_iron = parser_outter_rotor.get_conn("SURFACE1", "CPS3")

# Get the airgap connectivity
outter_rotor_conn_airgap = parser_outter_rotor.get_conn("SURFACE23", "CPS3")

# Get the connectivity of the airgap between the magnets
outter_rotor_conn_ag_mag_1 = parser_outter_rotor.get_conn("SURFACE12", "CPS3")
outter_rotor_conn_ag_mag_2 = parser_outter_rotor.get_conn("SURFACE13", "CPS3")
outter_rotor_conn_ag_mag_3 = parser_outter_rotor.get_conn("SURFACE14", "CPS3")
outter_rotor_conn_ag_mag_4 = parser_outter_rotor.get_conn("SURFACE15", "CPS3")
outter_rotor_conn_ag_mag_5 = parser_outter_rotor.get_conn("SURFACE16", "CPS3")
outter_rotor_conn_ag_mag_6 = parser_outter_rotor.get_conn("SURFACE17", "CPS3")
outter_rotor_conn_ag_mag_7 = parser_outter_rotor.get_conn("SURFACE18", "CPS3")
outter_rotor_conn_ag_mag_8 = parser_outter_rotor.get_conn("SURFACE19", "CPS3")
outter_rotor_conn_ag_mag_9 = parser_outter_rotor.get_conn("SURFACE20", "CPS3")
outter_rotor_conn_ag_mag_10 = parser_outter_rotor.get_conn("SURFACE21", "CPS3")
outter_rotor_conn_ag_mag_11 = parser_outter_rotor.get_conn("SURFACE22", "CPS3")

# Line on the left edge
outter_rotor_left_edge0 = parser_outter_rotor.get_conn("LINE69", "T3D2")
outter_rotor_left_edge1 = parser_outter_rotor.get_conn("LINE54", "T3D2")
outter_rotor_left_edge2 = parser_outter_rotor.get_conn("LINE55", "T3D2")
outter_rotor_conn_pbc_left = np.concatenate(
    (
        outter_rotor_left_edge0.flatten(),
        outter_rotor_left_edge1.flatten(),
        outter_rotor_left_edge2.flatten(),
    ),
    axis=0,
)

# Turn into a single unique list of nodes preserving the order from bottom to top
# Oriented from bottom to top
outter_rotor_pbc_nodes_left = np.array(list(dict.fromkeys(outter_rotor_conn_pbc_left)))

# Line of the right edge
outter_rotor_right_edge0 = parser_outter_rotor.get_conn("LINE57", "T3D2")
outter_rotor_right_edge2 = parser_outter_rotor.get_conn("LINE52", "T3D2")
outter_rotor_right_edge1 = parser_outter_rotor.get_conn("LINE70", "T3D2")
outter_rotor_conn_pbc_right = np.concatenate(
    (
        outter_rotor_right_edge0.flatten(),
        outter_rotor_right_edge1.flatten(),
        outter_rotor_right_edge2.flatten(),
    ),
    axis=0,
)

# Turn into a single unique list of nodes preserving the order from top to bottom
# Oriented from bottom to top
outter_rotor_pbc_nodes_right = np.array(
    list(dict.fromkeys(outter_rotor_conn_pbc_right))
)

# Need to flip to make the orientation from bottom to top
outter_rotor_pbc_nodes_right = np.flip(outter_rotor_pbc_nodes_right)

# Line on the top and bottom edges
outter_rotor_top_edge = parser_outter_rotor.get_conn("LINE56", "T3D2")
outter_rotor_bottom_edge = parser_outter_rotor.get_conn("LINE53", "T3D2")
outter_rotor_dirichlet_nodes = np.array(
    list(dict.fromkeys(outter_rotor_top_edge.flatten()))
)  # Left to right
outter_rotor_pbc_nodes_bottom = np.array(
    list(dict.fromkeys(outter_rotor_bottom_edge.flatten()))
)
outter_rotor_pbc_nodes_bottom = np.flip(outter_rotor_pbc_nodes_bottom)

# Total number of elements in the outter rotor
outter_rotor_conns = [
    outter_rotor_conn_m1,
    outter_rotor_conn_m2,
    outter_rotor_conn_m3,
    outter_rotor_conn_m4,
    outter_rotor_conn_m5,
    outter_rotor_conn_m6,
    outter_rotor_conn_m7,
    outter_rotor_conn_m8,
    outter_rotor_conn_m9,
    outter_rotor_conn_m10,
    outter_rotor_conn_back_iron,
    outter_rotor_conn_airgap,
    outter_rotor_conn_ag_mag_1,
    outter_rotor_conn_ag_mag_2,
    outter_rotor_conn_ag_mag_3,
    outter_rotor_conn_ag_mag_4,
    outter_rotor_conn_ag_mag_5,
    outter_rotor_conn_ag_mag_6,
    outter_rotor_conn_ag_mag_7,
    outter_rotor_conn_ag_mag_8,
    outter_rotor_conn_ag_mag_9,
    outter_rotor_conn_ag_mag_10,
    outter_rotor_conn_ag_mag_11,
]

# Compute total number of elements and nodes
nelems_outter_rotor = sum(conn.shape[0] for conn in outter_rotor_conns)
nnodes_outter_rotor = X_outter_rotor.shape[0]
print("\nNELEMS outter rotor (COMPUTED):", nelems_outter_rotor)

######################################
# Inner rotor connectivity information
######################################
# Get the connectivity for each of the magnets
# conn_m1 = connectivity for magnet 1 (starting from far left)
inner_rotor_conn_m1 = parser_inner_rotor.get_conn("SURFACE11", "CPS3")
inner_rotor_conn_m2 = parser_inner_rotor.get_conn("SURFACE10", "CPS3")
inner_rotor_conn_m3 = parser_inner_rotor.get_conn("SURFACE9", "CPS3")
inner_rotor_conn_m4 = parser_inner_rotor.get_conn("SURFACE8", "CPS3")
inner_rotor_conn_m5 = parser_inner_rotor.get_conn("SURFACE7", "CPS3")
inner_rotor_conn_m6 = parser_inner_rotor.get_conn("SURFACE6", "CPS3")
inner_rotor_conn_m7 = parser_inner_rotor.get_conn("SURFACE5", "CPS3")
inner_rotor_conn_m8 = parser_inner_rotor.get_conn("SURFACE4", "CPS3")
inner_rotor_conn_m9 = parser_inner_rotor.get_conn("SURFACE3", "CPS3")
inner_rotor_conn_m10 = parser_inner_rotor.get_conn("SURFACE2", "CPS3")

# Get the back iron connectivity
inner_rotor_conn_back_iron = parser_inner_rotor.get_conn("SURFACE1", "CPS3")

# Get the airgap connectivity
inner_rotor_conn_airgap = parser_inner_rotor.get_conn("SURFACE23", "CPS3")

# Get the connectivity of the airgap between the magnets
inner_rotor_conn_ag_mag_1 = parser_inner_rotor.get_conn("SURFACE12", "CPS3")
inner_rotor_conn_ag_mag_2 = parser_inner_rotor.get_conn("SURFACE13", "CPS3")
inner_rotor_conn_ag_mag_3 = parser_inner_rotor.get_conn("SURFACE14", "CPS3")
inner_rotor_conn_ag_mag_4 = parser_inner_rotor.get_conn("SURFACE15", "CPS3")
inner_rotor_conn_ag_mag_5 = parser_inner_rotor.get_conn("SURFACE16", "CPS3")
inner_rotor_conn_ag_mag_6 = parser_inner_rotor.get_conn("SURFACE17", "CPS3")
inner_rotor_conn_ag_mag_7 = parser_inner_rotor.get_conn("SURFACE18", "CPS3")
inner_rotor_conn_ag_mag_8 = parser_inner_rotor.get_conn("SURFACE19", "CPS3")
inner_rotor_conn_ag_mag_9 = parser_inner_rotor.get_conn("SURFACE20", "CPS3")
inner_rotor_conn_ag_mag_10 = parser_inner_rotor.get_conn("SURFACE21", "CPS3")
inner_rotor_conn_ag_mag_11 = parser_inner_rotor.get_conn("SURFACE22", "CPS3")

# Line on the left edge
inner_rotor_left_edge0 = parser_inner_rotor.get_conn("LINE69", "T3D2")
inner_rotor_left_edge1 = parser_inner_rotor.get_conn("LINE54", "T3D2")
inner_rotor_left_edge2 = parser_inner_rotor.get_conn("LINE55", "T3D2")
inner_rotor_conn_pbc_left = np.concatenate(
    (
        inner_rotor_left_edge0.flatten(),
        inner_rotor_left_edge1.flatten(),
        inner_rotor_left_edge2.flatten(),
    ),
    axis=0,
)

# Turn into a single unique list of nodes preserving the order from bottom to top
# Oriented from bottom to top
inner_rotor_pbc_nodes_left = np.array(list(dict.fromkeys(inner_rotor_conn_pbc_left)))
inner_rotor_pbc_nodes_left = np.flip(inner_rotor_pbc_nodes_left)

# Line of the right edge
inner_rotor_right_edge0 = parser_inner_rotor.get_conn("LINE57", "T3D2")
inner_rotor_right_edge1 = parser_inner_rotor.get_conn("LINE52", "T3D2")
inner_rotor_right_edge2 = parser_inner_rotor.get_conn("LINE70", "T3D2")
inner_rotor_conn_pbc_right = np.concatenate(
    (
        inner_rotor_right_edge0.flatten(),
        inner_rotor_right_edge1.flatten(),
        inner_rotor_right_edge2.flatten(),
    ),
    axis=0,
)

# Turn into a single unique list of nodes preserving the order from bottom to top
# Oriented from bottom to top
inner_rotor_pbc_nodes_right = np.array(list(dict.fromkeys(inner_rotor_conn_pbc_right)))

# Line on the top and bottom edges
inner_rotor_bottom_edge = parser_inner_rotor.get_conn("LINE56", "T3D2")
inner_rotor_top_edge = parser_inner_rotor.get_conn("LINE53", "T3D2")
inner_rotor_dirichlet_nodes = np.array(
    list(dict.fromkeys(inner_rotor_bottom_edge.flatten()))
)  # Left to right
inner_rotor_pbc_nodes_top = np.array(
    list(dict.fromkeys(inner_rotor_top_edge.flatten()))
)

inner_rotor_pbc_nodes_top = np.flip(inner_rotor_pbc_nodes_top)

# Total number of elements in the inner rotor
inner_rotor_conns = [
    inner_rotor_conn_m1,
    inner_rotor_conn_m2,
    inner_rotor_conn_m3,
    inner_rotor_conn_m4,
    inner_rotor_conn_m5,
    inner_rotor_conn_m6,
    inner_rotor_conn_m7,
    inner_rotor_conn_m8,
    inner_rotor_conn_m9,
    inner_rotor_conn_m10,
    inner_rotor_conn_back_iron,
    inner_rotor_conn_airgap,
    inner_rotor_conn_ag_mag_1,
    inner_rotor_conn_ag_mag_2,
    inner_rotor_conn_ag_mag_3,
    inner_rotor_conn_ag_mag_4,
    inner_rotor_conn_ag_mag_5,
    inner_rotor_conn_ag_mag_6,
    inner_rotor_conn_ag_mag_7,
    inner_rotor_conn_ag_mag_8,
    inner_rotor_conn_ag_mag_9,
    inner_rotor_conn_ag_mag_10,
    inner_rotor_conn_ag_mag_11,
]

# Compute total number of elements and nodes
nelems_inner_rotor = sum(conn.shape[0] for conn in inner_rotor_conns)
nnodes_inner_rotor = X_inner_rotor.shape[0]
print("\nNELEMS inner rotor (COMPUTED):", nelems_inner_rotor)

#####################
# Plot motor geometry
#####################
if args.plot_geometry:
    plot_motor.plot(
        slide_number,
        total_length,
        copper_slot_height,
        tooth_tip_thickness,
        airgap,
        magnet_thickness,
        back_iron_thickness,
        X_stator,
        X_inner_rotor,
        X_outter_rotor,
        stator_conn_s1,
        stator_conn_s2,
        stator_conn_s3,
        stator_conn_s4,
        stator_conn_s5,
        stator_conn_s6,
        stator_conn_s7,
        stator_conn_s8,
        stator_conn_s9,
        stator_conn_s10,
        stator_conn_s11,
        stator_conn_s12,
        stator_conn_s13,
        stator_conn_s14,
        stator_conn_s15,
        stator_conn_s16,
        stator_conn_s17,
        stator_conn_s18,
        stator_conn_s19,
        stator_conn_s20,
        stator_conn_s21,
        stator_conn_s22,
        stator_conn_s23,
        stator_conn_s24,
        stator_conn_t1,
        stator_conn_t2,
        stator_conn_t3,
        stator_conn_t4,
        stator_conn_t5,
        stator_conn_t6,
        stator_conn_t7,
        stator_conn_t8,
        stator_conn_t9,
        stator_conn_t10,
        stator_conn_t11,
        stator_conn_t12,
        stator_conn_t13,
        stator_conn_ag_inner,
        stator_conn_ag_outter,
        stator_pbc_nodes_left,
        stator_pbc_nodes_right,
        stator_pbc_nodes_bottom,
        stator_pbc_nodes_top,
        stator_conn_ag_1,
        stator_conn_ag_2,
        stator_conn_ag_3,
        stator_conn_ag_4,
        stator_conn_ag_5,
        stator_conn_ag_6,
        stator_conn_ag_7,
        stator_conn_ag_8,
        stator_conn_ag_9,
        stator_conn_ag_10,
        stator_conn_ag_11,
        stator_conn_ag_12,
        stator_conn_ag_13,
        stator_conn_ag_14,
        stator_conn_ag_15,
        stator_conn_ag_16,
        stator_conn_ag_17,
        stator_conn_ag_18,
        stator_conn_ag_19,
        stator_conn_ag_20,
        stator_conn_ag_21,
        stator_conn_ag_22,
        stator_conn_ag_23,
        stator_conn_ag_24,
        outter_rotor_conn_m1,
        outter_rotor_conn_m2,
        outter_rotor_conn_m3,
        outter_rotor_conn_m4,
        outter_rotor_conn_m5,
        outter_rotor_conn_m6,
        outter_rotor_conn_m7,
        outter_rotor_conn_m8,
        outter_rotor_conn_m9,
        outter_rotor_conn_m10,
        outter_rotor_conn_back_iron,
        outter_rotor_conn_airgap,
        outter_rotor_pbc_nodes_left,
        outter_rotor_pbc_nodes_right,
        outter_rotor_dirichlet_nodes,
        outter_rotor_pbc_nodes_bottom,
        outter_rotor_conn_ag_mag_1,
        outter_rotor_conn_ag_mag_2,
        outter_rotor_conn_ag_mag_3,
        outter_rotor_conn_ag_mag_4,
        outter_rotor_conn_ag_mag_5,
        outter_rotor_conn_ag_mag_6,
        outter_rotor_conn_ag_mag_7,
        outter_rotor_conn_ag_mag_8,
        outter_rotor_conn_ag_mag_9,
        outter_rotor_conn_ag_mag_10,
        outter_rotor_conn_ag_mag_11,
        inner_rotor_conn_m1,
        inner_rotor_conn_m2,
        inner_rotor_conn_m3,
        inner_rotor_conn_m4,
        inner_rotor_conn_m5,
        inner_rotor_conn_m6,
        inner_rotor_conn_m7,
        inner_rotor_conn_m8,
        inner_rotor_conn_m9,
        inner_rotor_conn_m10,
        inner_rotor_conn_back_iron,
        inner_rotor_conn_airgap,
        inner_rotor_pbc_nodes_left,
        inner_rotor_pbc_nodes_right,
        inner_rotor_dirichlet_nodes,
        inner_rotor_pbc_nodes_top,
        inner_rotor_conn_ag_mag_1,
        inner_rotor_conn_ag_mag_2,
        inner_rotor_conn_ag_mag_3,
        inner_rotor_conn_ag_mag_4,
        inner_rotor_conn_ag_mag_5,
        inner_rotor_conn_ag_mag_6,
        inner_rotor_conn_ag_mag_7,
        inner_rotor_conn_ag_mag_8,
        inner_rotor_conn_ag_mag_9,
        inner_rotor_conn_ag_mag_10,
        inner_rotor_conn_ag_mag_11,
    )

################################
# Global element indices mapping
################################
# Stator Region
stator_surfaces = {
    "s1": stator_conn_s1,
    "s2": stator_conn_s2,
    "s3": stator_conn_s3,
    "s4": stator_conn_s4,
    "s5": stator_conn_s5,
    "s6": stator_conn_s6,
    "s7": stator_conn_s7,
    "s8": stator_conn_s8,
    "s9": stator_conn_s9,
    "s10": stator_conn_s10,
    "s11": stator_conn_s11,
    "s12": stator_conn_s12,
    "s13": stator_conn_s13,
    "s14": stator_conn_s14,
    "s15": stator_conn_s15,
    "s16": stator_conn_s16,
    "s17": stator_conn_s17,
    "s18": stator_conn_s18,
    "s19": stator_conn_s19,
    "s20": stator_conn_s20,
    "s21": stator_conn_s21,
    "s22": stator_conn_s22,
    "s23": stator_conn_s23,
    "s24": stator_conn_s24,
    "t1": stator_conn_t1,
    "t2": stator_conn_t2,
    "t3": stator_conn_t3,
    "t4": stator_conn_t4,
    "t5": stator_conn_t5,
    "t6": stator_conn_t6,
    "t7": stator_conn_t7,
    "t8": stator_conn_t8,
    "t9": stator_conn_t9,
    "t10": stator_conn_t10,
    "t11": stator_conn_t11,
    "t12": stator_conn_t12,
    "t13": stator_conn_t13,
    "ag_outer": stator_conn_ag_outter,
    "ag_inner": stator_conn_ag_inner,
    "ag1": stator_conn_ag_1,
    "ag2": stator_conn_ag_2,
    "ag3": stator_conn_ag_3,
    "ag4": stator_conn_ag_4,
    "ag5": stator_conn_ag_5,
    "ag6": stator_conn_ag_6,
    "ag7": stator_conn_ag_7,
    "ag8": stator_conn_ag_8,
    "ag9": stator_conn_ag_9,
    "ag10": stator_conn_ag_10,
    "ag11": stator_conn_ag_11,
    "ag12": stator_conn_ag_12,
    "ag13": stator_conn_ag_13,
    "ag14": stator_conn_ag_14,
    "ag15": stator_conn_ag_15,
    "ag16": stator_conn_ag_16,
    "ag17": stator_conn_ag_17,
    "ag18": stator_conn_ag_18,
    "ag19": stator_conn_ag_19,
    "ag20": stator_conn_ag_20,
    "ag21": stator_conn_ag_21,
    "ag22": stator_conn_ag_22,
    "ag23": stator_conn_ag_23,
    "ag24": stator_conn_ag_24,
}
stator_elem_indices_map, stator_glob_conn = utils.build_global_mapping(stator_surfaces)

# Store the global element indices for the stator teeth
stator_teeth_indices = np.hstack(
    [
        stator_elem_indices_map["t1"],
        stator_elem_indices_map["t2"],
        stator_elem_indices_map["t3"],
        stator_elem_indices_map["t4"],
        stator_elem_indices_map["t5"],
        stator_elem_indices_map["t6"],
        stator_elem_indices_map["t7"],
        stator_elem_indices_map["t8"],
        stator_elem_indices_map["t9"],
        stator_elem_indices_map["t10"],
        stator_elem_indices_map["t11"],
        stator_elem_indices_map["t12"],
        stator_elem_indices_map["t13"],
    ]
)

# Store the global element indices for the stator airgap
stator_ag_indices = np.hstack(
    [
        stator_elem_indices_map["ag_outer"],
        stator_elem_indices_map["ag_inner"],
        stator_elem_indices_map["ag1"],
        stator_elem_indices_map["ag2"],
        stator_elem_indices_map["ag3"],
        stator_elem_indices_map["ag4"],
        stator_elem_indices_map["ag5"],
        stator_elem_indices_map["ag6"],
        stator_elem_indices_map["ag7"],
        stator_elem_indices_map["ag8"],
        stator_elem_indices_map["ag9"],
        stator_elem_indices_map["ag10"],
        stator_elem_indices_map["ag11"],
        stator_elem_indices_map["ag12"],
        stator_elem_indices_map["ag13"],
        stator_elem_indices_map["ag14"],
        stator_elem_indices_map["ag15"],
        stator_elem_indices_map["ag16"],
        stator_elem_indices_map["ag17"],
        stator_elem_indices_map["ag18"],
        stator_elem_indices_map["ag19"],
        stator_elem_indices_map["ag20"],
        stator_elem_indices_map["ag21"],
        stator_elem_indices_map["ag22"],
        stator_elem_indices_map["ag23"],
        stator_elem_indices_map["ag24"],
    ]
)

# Store the global element indices for the stator copper
stator_copper_indices = np.hstack(
    [
        stator_elem_indices_map["s1"],
        stator_elem_indices_map["s2"],
        stator_elem_indices_map["s3"],
        stator_elem_indices_map["s4"],
        stator_elem_indices_map["s5"],
        stator_elem_indices_map["s6"],
        stator_elem_indices_map["s7"],
        stator_elem_indices_map["s8"],
        stator_elem_indices_map["s9"],
        stator_elem_indices_map["s10"],
        stator_elem_indices_map["s11"],
        stator_elem_indices_map["s12"],
        stator_elem_indices_map["s13"],
        stator_elem_indices_map["s14"],
        stator_elem_indices_map["s15"],
        stator_elem_indices_map["s16"],
        stator_elem_indices_map["s17"],
        stator_elem_indices_map["s18"],
        stator_elem_indices_map["s19"],
        stator_elem_indices_map["s20"],
        stator_elem_indices_map["s21"],
        stator_elem_indices_map["s22"],
        stator_elem_indices_map["s23"],
        stator_elem_indices_map["s24"],
    ]
)

stator_phase1_pos_indices = np.hstack(
    [
        stator_elem_indices_map["s1"],
        stator_elem_indices_map["s2"],
        stator_elem_indices_map["s12"],
        stator_elem_indices_map["s15"],
    ]
)
stator_phase1_neg_indices = np.hstack(
    [
        stator_elem_indices_map["s3"],
        stator_elem_indices_map["s13"],
        stator_elem_indices_map["s14"],
        stator_elem_indices_map["s24"],
    ]
)
stator_phase2_pos_indices = np.hstack(
    [
        stator_elem_indices_map["s4"],
        stator_elem_indices_map["s7"],
        stator_elem_indices_map["s17"],
        stator_elem_indices_map["s18"],
    ]
)
stator_phase2_neg_indices = np.hstack(
    [
        stator_elem_indices_map["s5"],
        stator_elem_indices_map["s6"],
        stator_elem_indices_map["s16"],
        stator_elem_indices_map["s19"],
    ]
)
stator_phase3_pos_indices = np.hstack(
    [
        stator_elem_indices_map["s9"],
        stator_elem_indices_map["s10"],
        stator_elem_indices_map["s20"],
        stator_elem_indices_map["s23"],
    ]
)
stator_phase3_neg_indices = np.hstack(
    [
        stator_elem_indices_map["s8"],
        stator_elem_indices_map["s11"],
        stator_elem_indices_map["s21"],
        stator_elem_indices_map["s22"],
    ]
)


# Outter Rotor connectivity
outter_rotor_surfaces = {
    "back_iron": outter_rotor_conn_back_iron,
    "m10": outter_rotor_conn_m10,
    "m9": outter_rotor_conn_m9,
    "m8": outter_rotor_conn_m8,
    "m7": outter_rotor_conn_m7,
    "m6": outter_rotor_conn_m6,
    "m5": outter_rotor_conn_m5,
    "m4": outter_rotor_conn_m4,
    "m3": outter_rotor_conn_m3,
    "m2": outter_rotor_conn_m2,
    "m1": outter_rotor_conn_m1,
    "ag": outter_rotor_conn_airgap,
    "ag1": outter_rotor_conn_ag_mag_1,
    "ag2": outter_rotor_conn_ag_mag_2,
    "ag3": outter_rotor_conn_ag_mag_3,
    "ag4": outter_rotor_conn_ag_mag_4,
    "ag5": outter_rotor_conn_ag_mag_5,
    "ag6": outter_rotor_conn_ag_mag_6,
    "ag7": outter_rotor_conn_ag_mag_7,
    "ag8": outter_rotor_conn_ag_mag_8,
    "ag9": outter_rotor_conn_ag_mag_9,
    "ag10": outter_rotor_conn_ag_mag_10,
    "ag11": outter_rotor_conn_ag_mag_11,
}
outter_rotor_elem_indices_map, outter_rotor_glob_conn = utils.build_global_mapping(
    outter_rotor_surfaces
)

# Store the global element indices for the airgap
outter_rotor_ag_inidices = np.hstack(
    [
        outter_rotor_elem_indices_map["ag"],
        outter_rotor_elem_indices_map["ag1"],
        outter_rotor_elem_indices_map["ag2"],
        outter_rotor_elem_indices_map["ag3"],
        outter_rotor_elem_indices_map["ag4"],
        outter_rotor_elem_indices_map["ag5"],
        outter_rotor_elem_indices_map["ag6"],
        outter_rotor_elem_indices_map["ag7"],
        outter_rotor_elem_indices_map["ag8"],
        outter_rotor_elem_indices_map["ag9"],
        outter_rotor_elem_indices_map["ag10"],
        outter_rotor_elem_indices_map["ag11"],
    ]
)

# Store global element indices for the magnets
# NS = North-South pole
outter_rotor_magnet_NS_indices = np.hstack(
    [
        outter_rotor_elem_indices_map["m9"],
        outter_rotor_elem_indices_map["m7"],
        outter_rotor_elem_indices_map["m5"],
        outter_rotor_elem_indices_map["m3"],
        outter_rotor_elem_indices_map["m1"],
    ]
)
# SN = South-North pole
outter_rotor_magnet_SN_indices = np.hstack(
    [
        outter_rotor_elem_indices_map["m10"],
        outter_rotor_elem_indices_map["m8"],
        outter_rotor_elem_indices_map["m6"],
        outter_rotor_elem_indices_map["m4"],
        outter_rotor_elem_indices_map["m2"],
    ]
)

# Inner Rotor connectivity
inner_rotor_surfaces = {
    "back_iron": inner_rotor_conn_back_iron,
    "m10": inner_rotor_conn_m10,
    "m9": inner_rotor_conn_m9,
    "m8": inner_rotor_conn_m8,
    "m7": inner_rotor_conn_m7,
    "m6": inner_rotor_conn_m6,
    "m5": inner_rotor_conn_m5,
    "m4": inner_rotor_conn_m4,
    "m3": inner_rotor_conn_m3,
    "m2": inner_rotor_conn_m2,
    "m1": inner_rotor_conn_m1,
    "ag": inner_rotor_conn_airgap,
    "ag1": inner_rotor_conn_ag_mag_1,
    "ag2": inner_rotor_conn_ag_mag_2,
    "ag3": inner_rotor_conn_ag_mag_3,
    "ag4": inner_rotor_conn_ag_mag_4,
    "ag5": inner_rotor_conn_ag_mag_5,
    "ag6": inner_rotor_conn_ag_mag_6,
    "ag7": inner_rotor_conn_ag_mag_7,
    "ag8": inner_rotor_conn_ag_mag_8,
    "ag9": inner_rotor_conn_ag_mag_9,
    "ag10": inner_rotor_conn_ag_mag_10,
    "ag11": inner_rotor_conn_ag_mag_11,
}
inner_rotor_elem_indices_map, inner_rotor_glob_conn = utils.build_global_mapping(
    inner_rotor_surfaces
)

# Store the global element indices for the airgap
inner_rotor_ag_inidices = np.hstack(
    [
        inner_rotor_elem_indices_map["ag"],
        inner_rotor_elem_indices_map["ag1"],
        inner_rotor_elem_indices_map["ag2"],
        inner_rotor_elem_indices_map["ag3"],
        inner_rotor_elem_indices_map["ag4"],
        inner_rotor_elem_indices_map["ag5"],
        inner_rotor_elem_indices_map["ag6"],
        inner_rotor_elem_indices_map["ag7"],
        inner_rotor_elem_indices_map["ag8"],
        inner_rotor_elem_indices_map["ag9"],
        inner_rotor_elem_indices_map["ag10"],
        inner_rotor_elem_indices_map["ag11"],
    ]
)

# Store global element indices for the magnets
# SN = South-North pole
inner_rotor_magnet_SN_indices = np.hstack(
    [
        inner_rotor_elem_indices_map["m9"],
        inner_rotor_elem_indices_map["m7"],
        inner_rotor_elem_indices_map["m5"],
        inner_rotor_elem_indices_map["m3"],
        inner_rotor_elem_indices_map["m1"],
    ]
)

# NS = North-South pole
inner_rotor_magnet_NS_indices = np.hstack(
    [
        inner_rotor_elem_indices_map["m10"],
        inner_rotor_elem_indices_map["m8"],
        inner_rotor_elem_indices_map["m6"],
        inner_rotor_elem_indices_map["m4"],
        inner_rotor_elem_indices_map["m2"],
    ]
)

#############
# Amigo Model
#############
model = am.Model("afpm")

# Define components
maxwell = Maxwell()
coils = Coil()
magnets = Magnets()
node_src = NodeSource()
materials = MaterialSource()
dirichlet = DirichletBc()
symm = SymmBc()

# Materials components
#   - alpha[0] = iron
#   - alpha[1] = copper
#   - alpha[2] = air
#   - alpha[3] = magnets
#   - Jz_phase1_pos[4] = phase 1 out
#   - Jz_phase2_pos[5] = phase 2 out
#   - Jz_phase3_pos[6] = phase 3 out
#   - Jz_phase1_neg[7] = phase 1 into
#   - Jz_phase2_neg[8] = phase 2 into
#   - Jz_phase3_neg[9] = phase 3 into
#   - Mx_NS[10] = 0.0
#   - My_NS[11] = +My
#   - Mx_NS[12] = 0.0
#   - My_NS[13] = -My
model.add_component("material_src", 14, materials)

# Stator components
model.add_component("stator_maxwell", nelems_stator, maxwell)
model.add_component("stator_node_src", nnodes_stator, node_src)
model.add_component("coils", nelems_stator, coils)
model.add_component("stator_pbc", len(stator_pbc_nodes_left), symm)

# Outter rotor components
model.add_component("outter_rotor_maxwell", nelems_outter_rotor, maxwell)
model.add_component("outter_rotor_magnets", nelems_outter_rotor, magnets)
model.add_component("outter_rotor_node_src", nnodes_outter_rotor, node_src)
model.add_component(
    "outter_rotor_pbc",
    len(
        outter_rotor_pbc_nodes_left[:-1]
        if slide_number > 0
        else outter_rotor_pbc_nodes_left[1:-1]
    ),
    symm,
)
model.add_component(
    "outter_rotor_dirichlet", len(outter_rotor_dirichlet_nodes), dirichlet
)

# Inner rotor components
model.add_component("inner_rotor_maxwell", nelems_inner_rotor, maxwell)
model.add_component("inner_rotor_magnets", nelems_inner_rotor, magnets)
model.add_component("inner_rotor_node_src", nnodes_inner_rotor, node_src)
model.add_component(
    "inner_rotor_pbc",
    len(
        inner_rotor_pbc_nodes_left[1:]
        if slide_number > 0
        else inner_rotor_pbc_nodes_left[1:-1]
    ),
    symm,
)
model.add_component(
    "inner_rotor_dirichlet", len(inner_rotor_dirichlet_nodes), dirichlet
)

# Airgap continuity bc components
model.add_component(
    "stator_outter_airgap_continuity_bc",
    len(stator_pbc_nodes_top[1 : -slide_number - 1]),
    symm,
)
model.add_component(
    "stator_inner_airgap_continuity_bc",
    len(stator_pbc_nodes_bottom[1 : -slide_number - 1]),
    symm,
)

# Enforced slide_number > 1 because if the slide number equals 1,
# the list for airgap pbc nodes will be empty.
if 1 < slide_number:
    # Airgap pbc components
    model.add_component(
        "stator_outter_airgap_pbc",
        len(outter_rotor_pbc_nodes_bottom[1:slide_number]),
        symm,
    )
    model.add_component(
        "stator_inner_airgap_pbc",
        len(inner_rotor_pbc_nodes_top[1:slide_number]),
        symm,
    )


##############
# Stator Model
##############
# Link node coordinates to components
# Ex: stator_maxwell.y_coord = src.y_coord[conn]
model.link(
    "stator_maxwell.x_coord",
    "stator_node_src.x_coord",
    tgt_indices=stator_glob_conn,
)
model.link(
    "stator_maxwell.y_coord",
    "stator_node_src.y_coord",
    tgt_indices=stator_glob_conn,
)
model.link(
    "coils.x_coord",
    "stator_node_src.x_coord",
    tgt_indices=stator_glob_conn,
)

model.link(
    "coils.y_coord",
    "stator_node_src.y_coord",
    tgt_indices=stator_glob_conn,
)

# Link material source
model.link(
    "material_src.alpha[0]",  # Iron
    "stator_maxwell.alpha",
    tgt_indices=stator_teeth_indices,
)
model.link(
    "material_src.alpha[1]",  # Copper
    "stator_maxwell.alpha",
    tgt_indices=stator_copper_indices,
)
model.link(
    "material_src.alpha[2]",  # Air
    "stator_maxwell.alpha",
    tgt_indices=stator_ag_indices,
)
model.link(
    "material_src.Jz_phase1_pos[4]",  # Phase 1 Current (pos)
    "coils.Jz",
    tgt_indices=stator_phase1_pos_indices,
)
model.link(
    "material_src.Jz_phase2_pos[5]",  # Phase 2 Current (pos)
    "coils.Jz",
    tgt_indices=stator_phase2_pos_indices,
)
model.link(
    "material_src.Jz_phase3_pos[6]",  # Phase 3 Current (pos)
    "coils.Jz",
    tgt_indices=stator_phase3_pos_indices,
)
model.link(
    "material_src.Jz_phase1_neg[7]",  # Phase 1 Current (neg)
    "coils.Jz",
    tgt_indices=stator_phase1_neg_indices,
)
model.link(
    "material_src.Jz_phase2_neg[8]",  # Phase 2 Current (neg)
    "coils.Jz",
    tgt_indices=stator_phase2_neg_indices,
)
model.link(
    "material_src.Jz_phase3_neg[9]",  # Phase 3 Current (neg)
    "coils.Jz",
    tgt_indices=stator_phase3_neg_indices,
)

# Link solution vectors
model.link(
    "coils.u",
    "stator_node_src.u",
    tgt_indices=stator_glob_conn,
)
model.link(
    "stator_maxwell.u",
    "stator_node_src.u",
    tgt_indices=stator_glob_conn,
)

# Link the periodic nodes
model.link("stator_node_src.u", "stator_pbc.u1", src_indices=stator_pbc_nodes_left)
model.link("stator_node_src.u", "stator_pbc.u2", src_indices=stator_pbc_nodes_right)

####################
# Outter Rotor Model
####################
# Link node coordinates to components
# Ex: outter_rotor_maxwell.y_coord = src.y_coord[conn]
model.link(
    "outter_rotor_maxwell.x_coord",
    "outter_rotor_node_src.x_coord",
    tgt_indices=outter_rotor_glob_conn,
)
model.link(
    "outter_rotor_maxwell.y_coord",
    "outter_rotor_node_src.y_coord",
    tgt_indices=outter_rotor_glob_conn,
)
model.link(
    "outter_rotor_magnets.x_coord",
    "outter_rotor_node_src.x_coord",
    tgt_indices=outter_rotor_glob_conn,
)
model.link(
    "outter_rotor_magnets.y_coord",
    "outter_rotor_node_src.y_coord",
    tgt_indices=outter_rotor_glob_conn,
)

# Link material source
model.link(
    "material_src.alpha[0]",  # Iron
    "outter_rotor_maxwell.alpha",
    tgt_indices=outter_rotor_elem_indices_map["back_iron"],
)
model.link(
    "material_src.alpha[2]",  # Air
    "outter_rotor_maxwell.alpha",
    tgt_indices=outter_rotor_ag_inidices,
)
model.link(
    "material_src.alpha[3]",  # Magnets
    "outter_rotor_maxwell.alpha",
    tgt_indices=outter_rotor_magnet_NS_indices,
)
model.link(
    "material_src.alpha[3]",  # Magnets
    "outter_rotor_maxwell.alpha",
    tgt_indices=outter_rotor_magnet_SN_indices,
)
model.link(
    "material_src.Mx_NS[10]",  # Mx for NS magnet
    "outter_rotor_magnets.Mx",
    tgt_indices=outter_rotor_magnet_NS_indices,
)
model.link(
    "material_src.My_NS[11]",  # My for NS magnet
    "outter_rotor_magnets.My",
    tgt_indices=outter_rotor_magnet_NS_indices,
)
model.link(
    "material_src.Mx_SN[12]",  # Mx for SN magnet
    "outter_rotor_magnets.Mx",
    tgt_indices=outter_rotor_magnet_SN_indices,
)
model.link(
    "material_src.My_NS[13]",  # My for SN magnet
    "outter_rotor_magnets.My",
    tgt_indices=outter_rotor_magnet_SN_indices,
)

# Link the periodic nodes
model.link(
    "outter_rotor_node_src.u",
    "outter_rotor_pbc.u1",
    src_indices=(
        outter_rotor_pbc_nodes_left[:-1]
        if slide_number > 0
        else outter_rotor_pbc_nodes_left[1:-1]
    ),
)
model.link(
    "outter_rotor_node_src.u",
    "outter_rotor_pbc.u2",
    src_indices=(
        outter_rotor_pbc_nodes_right[:-1]
        if slide_number > 0
        else outter_rotor_pbc_nodes_right[1:-1]
    ),
)

# Link the dirichlet nodes
model.link(
    "outter_rotor_node_src.u",
    "outter_rotor_dirichlet.u",
    src_indices=outter_rotor_dirichlet_nodes,
)

# Link the solution vectors
model.link(
    "outter_rotor_maxwell.u",
    "outter_rotor_node_src.u",
    tgt_indices=outter_rotor_glob_conn,
)
model.link(
    "outter_rotor_magnets.u",
    "outter_rotor_node_src.u",
    tgt_indices=outter_rotor_glob_conn,
)

###################
# Inner Rotor Model
###################
# Link node coordinates to components
# Ex: inner_rotor_maxwell.y_coord = src.y_coord[conn]
model.link(
    "inner_rotor_maxwell.x_coord",
    "inner_rotor_node_src.x_coord",
    tgt_indices=inner_rotor_glob_conn,
)
model.link(
    "inner_rotor_maxwell.y_coord",
    "inner_rotor_node_src.y_coord",
    tgt_indices=inner_rotor_glob_conn,
)
model.link(
    "inner_rotor_magnets.x_coord",
    "inner_rotor_node_src.x_coord",
    tgt_indices=inner_rotor_glob_conn,
)
model.link(
    "inner_rotor_magnets.y_coord",
    "inner_rotor_node_src.y_coord",
    tgt_indices=inner_rotor_glob_conn,
)

# Link material source
model.link(
    "material_src.alpha[0]",  # Iron
    "inner_rotor_maxwell.alpha",
    tgt_indices=inner_rotor_elem_indices_map["back_iron"],
)
model.link(
    "material_src.alpha[2]",  # Air
    "inner_rotor_maxwell.alpha",
    tgt_indices=inner_rotor_ag_inidices,
)
model.link(
    "material_src.alpha[3]",  # Magnets
    "inner_rotor_maxwell.alpha",
    tgt_indices=inner_rotor_magnet_NS_indices,
)
model.link(
    "material_src.alpha[3]",  # Magnets
    "inner_rotor_maxwell.alpha",
    tgt_indices=inner_rotor_magnet_SN_indices,
)
model.link(
    "material_src.Mx_NS[10]",  # Mx for NS magnet
    "inner_rotor_magnets.Mx",
    tgt_indices=inner_rotor_magnet_NS_indices,
)
model.link(
    "material_src.My_NS[11]",  # My for NS magnet
    "inner_rotor_magnets.My",
    tgt_indices=inner_rotor_magnet_NS_indices,
)
model.link(
    "material_src.Mx_SN[12]",  # Mx for SN magnet
    "inner_rotor_magnets.Mx",
    tgt_indices=inner_rotor_magnet_SN_indices,
)
model.link(
    "material_src.My_NS[13]",  # My for SN magnet
    "inner_rotor_magnets.My",
    tgt_indices=inner_rotor_magnet_SN_indices,
)

# Link the periodic nodes
model.link(
    "inner_rotor_node_src.u",
    "inner_rotor_pbc.u1",
    src_indices=(
        inner_rotor_pbc_nodes_left[1:]
        if slide_number > 0
        else inner_rotor_pbc_nodes_left[1:-1]
    ),
)
model.link(
    "inner_rotor_node_src.u",
    "inner_rotor_pbc.u2",
    src_indices=(
        inner_rotor_pbc_nodes_right[1:]
        if slide_number > 0
        else inner_rotor_pbc_nodes_right[1:-1]
    ),
)

# Link the dirichlet nodes
model.link(
    "inner_rotor_node_src.u",
    "inner_rotor_dirichlet.u",
    src_indices=inner_rotor_dirichlet_nodes,
)

# Link the solution vectors
model.link(
    "inner_rotor_maxwell.u",
    "inner_rotor_node_src.u",
    tgt_indices=inner_rotor_glob_conn,
)
model.link(
    "inner_rotor_magnets.u",
    "inner_rotor_node_src.u",
    tgt_indices=inner_rotor_glob_conn,
)

###################
# Airgap Continuity
###################
model.link(
    "stator_node_src.u",
    "stator_outter_airgap_continuity_bc.u1",
    src_indices=stator_pbc_nodes_top[1 : -slide_number - 1],
)
model.link(
    "outter_rotor_node_src.u",
    "stator_outter_airgap_continuity_bc.u2",
    src_indices=outter_rotor_pbc_nodes_bottom[slide_number + 1 : -1],
)
model.link(
    "stator_node_src.u",
    "stator_inner_airgap_continuity_bc.u1",
    src_indices=stator_pbc_nodes_bottom[1 : -slide_number - 1],
)
model.link(
    "inner_rotor_node_src.u",
    "stator_inner_airgap_continuity_bc.u2",
    src_indices=inner_rotor_pbc_nodes_top[slide_number + 1 : -1],
)

############
# Airgap PBC
############
if 1 < slide_number:
    model.link(
        "outter_rotor_node_src.u",
        "stator_outter_airgap_pbc.u1",
        src_indices=outter_rotor_pbc_nodes_bottom[1:slide_number],
    )

    model.link(
        "stator_node_src.u",
        "stator_outter_airgap_pbc.u2",
        src_indices=stator_pbc_nodes_top[-slide_number:-1],
    )
    model.link(
        "inner_rotor_node_src.u",
        "stator_inner_airgap_pbc.u1",
        src_indices=inner_rotor_pbc_nodes_top[1:slide_number],
    )

    model.link(
        "stator_node_src.u",
        "stator_inner_airgap_pbc.u2",
        src_indices=stator_pbc_nodes_bottom[-slide_number:-1],
    )

#####################
# Build and Run Model
#####################
# Build module
if args.build:
    compile_args = []
    link_args = ["-lblas", "-llapack"]
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args += ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module()

# Initialize the model
model.initialize()

# Compute current in each winding phase
phase1, phase2, phase3 = motor_controller.phase_currents(
    alpha=(2 * np.pi / npts_airgap) * slide_number,
    num_mag=num_mag,
    Jz_peak=Jz_peak,
)

# Set the problem data
data = model.get_data_vector()
data["stator_node_src.x_coord"] = X_stator[:, 0]
data["stator_node_src.y_coord"] = X_stator[:, 1]
data["outter_rotor_node_src.x_coord"] = X_outter_rotor[:, 0]
data["outter_rotor_node_src.y_coord"] = X_outter_rotor[:, 1]
data["inner_rotor_node_src.x_coord"] = X_inner_rotor[:, 0]
data["inner_rotor_node_src.y_coord"] = X_inner_rotor[:, 1]
data["material_src.alpha[0]"] = 1 / 7000  # Iron region alpha
data["material_src.alpha[1]"] = 1.0  # Copper region alpha
data["material_src.alpha[2]"] = 1.0  # Air region alpha
data["material_src.alpha[3]"] = 1.0 / 1.05  # Magnet region alpha
data["material_src.Jz_phase1_pos[4]"] = phase1
data["material_src.Jz_phase2_pos[5]"] = phase2
data["material_src.Jz_phase3_pos[6]"] = phase3
data["material_src.Jz_phase1_neg[7]"] = -phase1
data["material_src.Jz_phase2_neg[8]"] = -phase2
data["material_src.Jz_phase3_neg[9]"] = -phase3
data["material_src.Mx_NS[10]"] = 0.0  # Magnetization in x
data["material_src.My_NS[11]"] = My  # Magnetization in y
data["material_src.Mx_SN[12]"] = 0.0  # Magnetization in x
data["material_src.My_SN[13]"] = -My  # Magnetization in y
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

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans

# Extract solution for each mesh
vals_stator = ans_local.get_array()[model.get_indices("stator_node_src.u")]
vals_outter_rotor = ans_local.get_array()[model.get_indices("outter_rotor_node_src.u")]
vals_inner_rotor = ans_local.get_array()[model.get_indices("inner_rotor_node_src.u")]
vals = np.concatenate([vals_stator, vals_outter_rotor, vals_inner_rotor])


#####################
# Print out bc values
#####################
# print("\nStator left and right pbc:")
# print(vals_stator[stator_pbc_nodes_left])
# print(vals_stator[stator_pbc_nodes_right])
# print(vals_stator[stator_pbc_nodes_left] - vals_stator[stator_pbc_nodes_right])

# print("\nOuter rotor left and right pbc")
# print(vals_outter_rotor[outter_rotor_pbc_nodes_left])
# print(vals_outter_rotor[outter_rotor_pbc_nodes_right])
# print(
#     vals_outter_rotor[outter_rotor_pbc_nodes_left]
#     - vals_outter_rotor[outter_rotor_pbc_nodes_right]
# )

# print("\nInner rotor left and right pbc")
# print(vals_inner_rotor[inner_rotor_pbc_nodes_left])
# print(vals_inner_rotor[inner_rotor_pbc_nodes_right])
# print(
#     vals_inner_rotor[inner_rotor_pbc_nodes_left]
#     - vals_inner_rotor[inner_rotor_pbc_nodes_right]
# )

# if slide_number > 1:
#     print("\nOutter rotor, stator airgap pbc")
#     print(
#         vals_outter_rotor[outter_rotor_pbc_nodes_bottom[1:slide_number]],
#         vals_stator[stator_pbc_nodes_top[-slide_number:-1]],
#     )
#     print(
#         vals_outter_rotor[outter_rotor_pbc_nodes_bottom[1:slide_number]]
#         - vals_stator[stator_pbc_nodes_top[-slide_number:-1]],
#     )

#     print("\nInner rotor, stator airgap pbc")
#     print(
#         vals_inner_rotor[inner_rotor_pbc_nodes_top[1:slide_number]],
#         vals_stator[stator_pbc_nodes_bottom[-slide_number:-1]],
#     )
#     print(
#         vals_inner_rotor[inner_rotor_pbc_nodes_top[1:slide_number]]
#         - vals_stator[stator_pbc_nodes_bottom[-slide_number:-1]]
#     )

# print("\nDirichlet Outter Rotor")
# print(vals_outter_rotor[outter_rotor_dirichlet_nodes])

# print("\nDirichlet Inner Rotor")
# print(vals_inner_rotor[inner_rotor_dirichlet_nodes])


#####################
# Plot solution field
#####################
if args.plot_potential:
    fig_soln, ax_soln = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    min_level = np.min(vals)
    max_level = np.max(vals)
    levels = np.linspace(min_level, max_level, 20)
    cntr_stator = utils.plot_solution(
        ax_soln, X_stator, stator_glob_conn, vals_stator, levels
    )
    cntr_outter_rotor = utils.plot_solution(
        ax_soln, X_outter_rotor, outter_rotor_glob_conn, vals_outter_rotor, levels
    )
    cntr_inner_rotor = utils.plot_solution(
        ax_soln, X_inner_rotor, inner_rotor_glob_conn, vals_inner_rotor, levels
    )
    xlim = 2 * total_length
    ylim = (
        0.5 * copper_slot_height
        + tooth_tip_thickness
        + airgap
        + magnet_thickness
        + back_iron_thickness
    ) + 1e-3
    ax_soln.plot([xlim, xlim, -1e-3, -1e-3], [-ylim, ylim, ylim, -ylim], color="white")

    # Add colorbar at the top
    ticks = [min_level, 0.5 * (max_level + min_level), max_level]
    cbar = fig_soln.colorbar(
        cntr_stator,
        ax=ax_soln,
        # orientation="horizontal",
        location="left",
        pad=-0.03,
        fraction=0.005,  # adjust color bar height
    )
    cbar.set_ticks(ticks)
    cbar.formatter = ticker.FormatStrFormatter("%.2e")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=4)

    ax_soln.set_aspect("equal")
    ax_soln.axis("off")
    fig_soln.tight_layout()
    plt.savefig(
        f"Figures/Potential/{slide_number}.png",
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.01,
    )

##########################################
# Compute the magnetic field for each mesh
##########################################
Bx_stator, By_stator = compute_magnetic_field.compute(
    conn=stator_glob_conn,
    nodeCoords=X_stator,
    u=vals_stator,
)

Bx_inner_rotor, By_inner_rotor = compute_magnetic_field.compute(
    conn=inner_rotor_glob_conn,
    nodeCoords=X_inner_rotor,
    u=vals_inner_rotor,
)

Bx_outer_rotor, By_outer_rotor = compute_magnetic_field.compute(
    conn=outter_rotor_glob_conn,
    nodeCoords=X_outter_rotor,
    u=vals_outter_rotor,
)

Bmag_func = lambda x, y: np.sqrt(x**2 + y**2)

# Flux magnitude in the stator and rotors.
B_stator = Bmag_func(Bx_stator, By_stator)
B_inner_rotor = Bmag_func(Bx_inner_rotor, By_inner_rotor)
B_outer_rotor = Bmag_func(Bx_outer_rotor, By_outer_rotor)

################################
# Plot the magentic field (full)
################################
if args.plot_flux:
    B_stator = Bmag_func(Bx_stator, By_stator)
    B_inner_rotor = Bmag_func(Bx_inner_rotor, By_inner_rotor)
    B_outer_rotor = Bmag_func(Bx_outer_rotor, By_outer_rotor)
    min_Bmag = np.min(np.concatenate((B_stator, B_inner_rotor, B_outer_rotor)))
    max_Bmag = np.max(np.concatenate((B_stator, B_inner_rotor, B_outer_rotor)))
    fig_flux, ax_flux = plt.subplots()
    cntr_stator_flux = utils.plot_flux(
        ax_flux,
        X_stator,
        stator_glob_conn,
        B_stator,
        min_Bmag,
        max_Bmag,
    )
    cntr_inner_rotor_flux = utils.plot_flux(
        ax_flux,
        X_inner_rotor,
        inner_rotor_glob_conn,
        B_inner_rotor,
        min_Bmag,
        max_Bmag,
    )
    cntr_outter_rotor_flux = utils.plot_flux(
        ax_flux,
        X_outter_rotor,
        outter_rotor_glob_conn,
        B_outer_rotor,
        min_Bmag,
        max_Bmag,
    )

    xlim = 2 * total_length
    ylim = (
        0.5 * copper_slot_height
        + tooth_tip_thickness
        + airgap
        + magnet_thickness
        + back_iron_thickness
    ) + 1e-3
    ax_flux.plot([xlim, xlim, -1e-3, -1e-3], [-ylim, ylim, ylim, -ylim], color="white")
    ax_flux.set_aspect("equal")
    ax_flux.axis("off")

    # Add colorbar
    ticks = [min_Bmag, 0.5 * (max_Bmag + min_Bmag), max_Bmag]
    cbar = fig_flux.colorbar(
        cntr_stator_flux,
        ax=ax_flux,
        # orientation="horizontal",
        location="left",
        pad=-0.03,
        fraction=0.005,  # make the colorbar thinner
    )
    cbar.set_ticks(ticks)
    cbar.formatter = ticker.FormatStrFormatter("%.2e")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=4)

    fig_flux.tight_layout()
    plt.savefig(
        f"Figures/Flux/{slide_number}.png",
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.01,
    )

##########################################
# Compute the magnetic field in the airgap
##########################################
# Outer rotor Bfield (airgap)
Bx_outer_rotor_ag, By_outer_rotor_ag = compute_magnetic_field.compute(
    conn=outter_rotor_conn_airgap,
    u=vals_outter_rotor,
    nodeCoords=X_outter_rotor,
)
B_outer_rotor_ag = Bmag_func(Bx_outer_rotor_ag, By_outer_rotor_ag)

# Inner rotor (airgap)
Bx_inner_rotor_ag, By_inner_rotor_ag = compute_magnetic_field.compute(
    conn=inner_rotor_conn_airgap,
    u=vals_inner_rotor,
    nodeCoords=X_inner_rotor,
)
B_inner_rotor_ag = Bmag_func(Bx_inner_rotor_ag, By_inner_rotor_ag)

# Stator inner (airgap)
Bx_inner_stator_ag, By_inner_stator_ag = compute_magnetic_field.compute(
    conn=stator_conn_ag_inner,
    u=vals_stator,
    nodeCoords=X_stator,
)
B_inner_stator_ag = Bmag_func(Bx_inner_stator_ag, By_inner_stator_ag)

# Stator outer (airgap)
Bx_outer_stator_ag, By_outer_stator_ag = compute_magnetic_field.compute(
    conn=stator_conn_ag_outter,
    u=vals_stator,
    nodeCoords=X_stator,
)
B_outer_stator_ag = Bmag_func(Bx_outer_stator_ag, By_outer_stator_ag)


##################################
# Plot the magentic field (airgap)
##################################
if args.plot_ag_flux:
    min_val_ag = np.min(
        np.concatenate(
            (
                B_outer_rotor_ag,
                B_inner_rotor_ag,
                B_inner_stator_ag,
                B_outer_stator_ag,
            )
        )
    )
    max_val_ag = np.max(
        np.concatenate(
            (
                B_outer_rotor_ag,
                B_inner_rotor_ag,
                B_inner_stator_ag,
                B_outer_stator_ag,
            )
        )
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    utils.plot_airgap_flux(
        ax=ax,
        xyz_nodeCoords=X_outter_rotor,
        conn=outter_rotor_glob_conn,
        # z_conn=outter_rotor_conn_airgap,
        elem_indices_map=outter_rotor_elem_indices_map["ag"],
        z=B_outer_rotor_ag,
        min_val=min_val_ag,
        max_val=max_val_ag,
    )
    utils.plot_airgap_flux(
        ax=ax,
        xyz_nodeCoords=X_inner_rotor,
        conn=inner_rotor_glob_conn,
        # z_conn=inner_rotor_conn_airgap,
        elem_indices_map=inner_rotor_elem_indices_map["ag"],
        z=B_inner_rotor_ag,
        min_val=min_val_ag,
        max_val=max_val_ag,
    )
    utils.plot_airgap_flux(
        ax=ax,
        xyz_nodeCoords=X_stator,
        conn=stator_glob_conn,
        # z_conn=stator_conn_ag_inner,
        elem_indices_map=stator_elem_indices_map["ag_inner"],
        z=B_inner_stator_ag,
        min_val=min_val_ag,
        max_val=max_val_ag,
    )
    utils.plot_airgap_flux(
        ax=ax,
        xyz_nodeCoords=X_stator,
        conn=stator_glob_conn,
        # z_conn=stator_conn_ag_outter,
        elem_indices_map=stator_elem_indices_map["ag_outer"],
        z=B_outer_stator_ag,
        min_val=min_val_ag,
        max_val=max_val_ag,
    )
    fig.tight_layout()


##########################
# Compute the force output
##########################
if args.plot_force_calc:
    # Plot the flux in the airgap used for the force calculation
    fig_force, ax_force = plt.subplots()
else:
    fig_force = None
    ax_force = None

# L = Ro - Ri  # Arrkio method stack length
L = Ro - Ri  # Arrkio method stack length
# Outer rotor
force_outer_rotor = compute_forces.compute(
    z0=0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap,
    z1=0.5 * copper_slot_height + tooth_tip_thickness + 1.0 * airgap,
    Bphi=Bx_outer_rotor_ag,
    Bz=By_outer_rotor_ag,
    conn=outter_rotor_conn_airgap,
    nodeCoords=X_outter_rotor,
    L=L,
    ax=ax_force,
    fig=fig_force,
)

force_inner_rotor = compute_forces.compute(
    z1=-(0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap),
    z0=-(0.5 * copper_slot_height + tooth_tip_thickness + 1.0 * airgap),
    Bphi=Bx_inner_rotor_ag,
    Bz=By_inner_rotor_ag,
    conn=inner_rotor_conn_airgap,
    nodeCoords=X_inner_rotor,
    L=L,
    ax=ax_force,
    fig=fig_force,
)

# Stator outer
force_stator_outer = compute_forces.compute(
    z0=0.5 * copper_slot_height + tooth_tip_thickness,
    z1=0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap,
    Bphi=Bx_outer_stator_ag,
    Bz=By_outer_stator_ag,
    conn=stator_conn_ag_outter,
    nodeCoords=X_stator,
    L=L,
    ax=ax_force,
    fig=fig_force,
)

# Stator inner
force_stator_inner = compute_forces.compute(
    z1=-(0.5 * copper_slot_height + tooth_tip_thickness),
    z0=-(0.5 * copper_slot_height + tooth_tip_thickness + 0.5 * airgap),
    Bphi=Bx_inner_stator_ag,
    Bz=By_inner_stator_ag,
    conn=stator_conn_ag_inner,
    nodeCoords=X_stator,
    L=L,
    ax=ax_force,
    fig=fig_force,
)

# Total force output for the radial slice
force = force_outer_rotor + force_inner_rotor + force_stator_outer + force_stator_inner
print()
print(f"Force (N): {force:.4f}")

###############
# Torque Output
###############
torque = force * (Ro - Ri)
print(f"Torque (N.m): {torque:.4f}")

######################################
# Compute the iron loss density (W/kg)
#####################################
# Compute mass/length (kg/m)
stator_iron_mass_per_length = compute_losses.mass_per_unit_length(
    X_stator,
    stator_glob_conn[stator_teeth_indices],  # ! slice elements for teeth
    rho_iron,
)

outer_iron_rotor_mass_per_length = compute_losses.mass_per_unit_length(
    X_outter_rotor,
    outter_rotor_glob_conn[outter_rotor_elem_indices_map["back_iron"]],  # ! slice?
    rho_iron,
)

inner_iron_rotor_mass_per_length = compute_losses.mass_per_unit_length(
    X_inner_rotor,
    inner_rotor_glob_conn[inner_rotor_elem_indices_map["back_iron"]],  # ! slice?
    rho_iron,
)

outer_NS_mass_per_length = compute_losses.mass_per_unit_length(
    X_outter_rotor,
    outter_rotor_glob_conn[outter_rotor_magnet_NS_indices],
    rho_magnet,
)

outer_SN_mass_per_length = compute_losses.mass_per_unit_length(
    X_outter_rotor,
    outter_rotor_glob_conn[outter_rotor_magnet_SN_indices],
    rho_magnet,
)

inner_NS_mass_per_length = compute_losses.mass_per_unit_length(
    X_inner_rotor,
    inner_rotor_glob_conn[inner_rotor_magnet_NS_indices],
    rho_magnet,
)

inner_SN_mass_per_length = compute_losses.mass_per_unit_length(
    X_inner_rotor,
    inner_rotor_glob_conn[inner_rotor_magnet_SN_indices],
    rho_magnet,
)


# Slice the flux in the stator teeth
# Compute the stator iron loss density
stator_iron_flux = B_stator[stator_teeth_indices]
stator_iron_loss_density = compute_losses.iron(
    stator_iron_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)

# Outer rotor loss density
outer_rotor_iron_flux = B_outer_rotor[outter_rotor_elem_indices_map["back_iron"]]
outer_rotor_iron_loss_density = compute_losses.iron(
    outer_rotor_iron_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)

# Inner rotor loss density
inner_rotor_iron_flux = B_inner_rotor[inner_rotor_elem_indices_map["back_iron"]]
inner_rotor_iron_loss_density = compute_losses.iron(
    inner_rotor_iron_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)


# Outer rotor NS magnet loss density
outer_NS_magnet_flux = B_outer_rotor[outter_rotor_magnet_NS_indices]
outer_NS_iron_loss_density = compute_losses.iron(
    outer_NS_magnet_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)


# Outer rotor SN magnet loss density
outer_SN_magnet_flux = B_outer_rotor[outter_rotor_magnet_SN_indices]
outer_SN_iron_loss_density = compute_losses.iron(
    outer_SN_magnet_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)

# Inner rotor NS magnet loss density
inner_NS_magnet_flux = B_inner_rotor[inner_rotor_magnet_NS_indices]
inner_NS_iron_loss_density = compute_losses.iron(
    inner_NS_magnet_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)

# Inner rotor SN magnet loss density
inner_SN_magnet_flux = B_inner_rotor[inner_rotor_magnet_SN_indices]
inner_SN_iron_loss_density = compute_losses.iron(
    inner_SN_magnet_flux,
    fund_freq=fund_freq,
    alpha=2.0,
    beta=0.0,
)

# Total iron loss density in the slice (W/kg)
total_iron_loss_density = (
    np.sum(stator_iron_loss_density)
    + np.sum(outer_rotor_iron_loss_density)
    + np.sum(inner_rotor_iron_loss_density)
    + np.sum(outer_NS_iron_loss_density)
    + np.sum(outer_SN_iron_loss_density)
    + np.sum(inner_NS_iron_loss_density)
    + np.sum(inner_SN_iron_loss_density)
)
print(f"Total Iron Loss (W/kg): {total_iron_loss_density:.6f}")

# Total iron loss density in the slice (W/m)
total_iron_loss_per_length = (
    np.dot(stator_iron_loss_density, stator_iron_mass_per_length)
    + np.dot(outer_rotor_iron_loss_density, outer_iron_rotor_mass_per_length)
    + np.dot(inner_rotor_iron_loss_density, inner_iron_rotor_mass_per_length)
    + np.dot(outer_NS_iron_loss_density, outer_NS_mass_per_length)
    + np.dot(outer_SN_iron_loss_density, outer_SN_mass_per_length)
    + np.dot(inner_NS_iron_loss_density, inner_NS_mass_per_length)
    + np.dot(inner_SN_iron_loss_density, inner_SN_mass_per_length)
)
print(f"Total Iron Loss (W/m): {total_iron_loss_per_length:.6f}")

iron_loss = total_iron_loss_per_length * (Ro - Ri)
print(f"Total Iron Loss (W): {iron_loss:.6f}")

if args.plot_loss:
    # Plot the loss denisty
    fig_iron_loss, ax_iron_loss = plt.subplots()

    stator_iron_loss = np.full(nelems_stator, np.nan)
    stator_iron_loss[stator_teeth_indices] = stator_iron_loss_density

    outer_rotor_iron_loss = np.full(nelems_outter_rotor, np.nan)
    outer_rotor_iron_loss[outter_rotor_elem_indices_map["back_iron"]] = (
        outer_rotor_iron_loss_density
    )
    outer_rotor_iron_loss[outter_rotor_magnet_NS_indices] = outer_NS_iron_loss_density
    outer_rotor_iron_loss[outter_rotor_magnet_SN_indices] = outer_SN_iron_loss_density

    inner_rotor_iron_loss = np.full(nelems_inner_rotor, np.nan)
    inner_rotor_iron_loss[inner_rotor_elem_indices_map["back_iron"]] = (
        inner_rotor_iron_loss_density
    )
    inner_rotor_iron_loss[inner_rotor_magnet_NS_indices] = inner_NS_iron_loss_density
    inner_rotor_iron_loss[inner_rotor_magnet_SN_indices] = inner_SN_iron_loss_density

    min_loss = np.min(
        np.concatenate(
            (
                stator_iron_loss_density,
                outer_rotor_iron_loss_density,
                inner_rotor_iron_loss_density,
            )
        )
    )
    max_loss = np.max(
        np.concatenate(
            (
                stator_iron_loss_density,
                outer_rotor_iron_loss_density,
                inner_rotor_iron_loss_density,
            )
        )
    )

    cntr_stator_loss = utils.plot_flux(
        ax_iron_loss,
        X_stator,
        stator_glob_conn,
        stator_iron_loss,
        min_loss,
        max_loss,
        cmap="magma",
    )

    cntr_inner_loss = utils.plot_flux(
        ax_iron_loss,
        X_outter_rotor,
        outter_rotor_glob_conn,
        outer_rotor_iron_loss,
        min_loss,
        max_loss,
        cmap="magma",
    )

    cntr_outter_loss = utils.plot_flux(
        ax_iron_loss,
        X_inner_rotor,
        inner_rotor_glob_conn,
        inner_rotor_iron_loss,
        min_loss,
        max_loss,
        cmap="magma",
    )

    xlim = 2 * total_length
    ylim = (
        0.5 * copper_slot_height
        + tooth_tip_thickness
        + airgap
        + magnet_thickness
        + back_iron_thickness
    ) + 1e-3
    ax_iron_loss.plot(
        [xlim, xlim, -1e-3, -1e-3], [-ylim, ylim, ylim, -ylim], color="white"
    )
    ax_iron_loss.set_aspect("equal")
    ax_iron_loss.axis("off")

    # Add colorbar
    ticks = [min_loss, 0.5 * (max_loss + min_loss), max_loss]
    cbar = fig_iron_loss.colorbar(
        cntr_stator_loss,
        ax=ax_iron_loss,
        location="left",
        pad=-0.03,
        fraction=0.005,
    )
    cbar.set_ticks(ticks)
    cbar.formatter = ticker.FormatStrFormatter("%.2e")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=4)

    fig_iron_loss.tight_layout()
    plt.savefig(
        f"Figures/Loss_Density/{slide_number}.png",
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.01,
    )

#########################
# Compute Copper Loss (W)
#########################
copper_loss_per_unit_length = compute_losses.copper(
    Jrms=Jz_peak_rms,
    rho_cu=1.7e-8,
    A_cu=A_cu,
)
total_copper_loss_per_unit_length = copper_loss_per_unit_length
copper_loss = total_copper_loss_per_unit_length * (Ro - Ri)
print(f"Copper Loss (W/m): {total_copper_loss_per_unit_length:.6f}")
print(f"Copper Loss (W): {copper_loss:.6f}")

total_load_loss = copper_loss + iron_loss
print(f"Total Load Loss (W): {total_load_loss}")

##################
# Motor Efficiency
##################
P_out = np.abs(torque) * omega
P_total = P_out + copper_loss + iron_loss
efficiency = (P_out / P_total) * 100
print(f"Output Power (W): {P_out}")
print(f"Efficiency (%): {efficiency:.4f}")


############
# Motor Mass
############
mass = compute_mass.mass(
    Ro=Ro,
    Ri=Ri,
    tooth_tip_thickness=tooth_tip_thickness,
    copper_slot_height=copper_slot_height,
    magnet_thickness=magnet_thickness,
    back_iron_thickness=back_iron_thickness,
    theta_mi=theta_mi,
    theta_mo=theta_mo,
    theta_bi=theta_bi,
    theta_bo=theta_bo,
    theta_ti=theta_ti,
    theta_to=theta_to,
    A_strand=A_strand,
)
print(f"Mass (kg): {mass:.2f}")

# plt.show()
