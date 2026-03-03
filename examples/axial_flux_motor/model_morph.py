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
    PlanarTruss,
    NodeSourcePlanarTruss,
    DirichletBcPlanarTruss,
    PlaneStress,
    NodeSourcePlaneStress,
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
    "--plot_geometry",
    action="store_true",
    help="Plot the motor geometry",
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
mesh_refinement = 2e-3
npts_airgap = 340
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

################
# Define Classes
################
planar_truss = PlanarTruss()
node_src_truss = NodeSourcePlanarTruss()
dirichlet_bc_truss = DirichletBcPlanarTruss()

ps = PlaneStress()
ps_node_src = NodeSourcePlaneStress()

#############
# Amigo Model
#############
model = am.Model("morph")


###################
# Outer Rotor Morph
###################
"""
DV's:
{
    airgap, 
    magnet_thickness,
    back_iron_thickness,
    Di,
    Do,
    theta_mi,
    theta_mo,
}


Upper and lower parts of magnet boundary
DV's: theta_m, magnet_thickness, back_iron_thickness:
{
    2,   4,  6,  8, 10, 12, 14, 16, 18, 20, (left -> right node order)
    50, 47, 44, 41, 38, 35, 32, 29, 26, 23  (right -> left node order)
}

Space Between the Magnets
DV's: theta_m, theta_m, total_length:
{
    1,   3,  5,  7,  9, 11, 13, 15, 17, 19, 21 (left -> right node order)
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68 (left -> right node order)
    
}

Vertical magnet edges
{
    51, 48, 45, 42, 39, 36, 33, 30, 27, 24 (down -> up)
    49, 46, 43, 40, 37, 34, 31, 28, 25, 22 (up -> down)

}

Airgap Left and Right Edges
{
   69, (down -> up node order)
   70  (up -> down node order)
}

Left Back Iron
{
    55 (down -> up node order)
}

Left Magnet Airgap Edge
{
    54 (down -> up node order)
}

Right Back Iron Edge
{
    57 (up -> down node order)
}

Right Magnet Airgap Edge
{
    52 (up -> down node order)
}

Top and Bottom Edges
{
    56, (left -> right node order)
    53, (right -> left node order)
}

"""
line_set_upper_magnet_boundary = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
line_set_lower_magnet_boundary = [50, 47, 44, 41, 38, 35, 32, 29, 26, 23]

line_set_upper_space_between_magnet = [3, 5, 7, 9, 11, 13, 15, 17, 19]
line_set_lower_space_between_magnet = [59, 60, 61, 62, 63, 64, 65, 66, 67]

line_set_left_upper_half_space_between_magnet = [1]
line_set_right_upper_half_space_between_magnet = [21]
line_set_left_lower_half_space_between_magnet = [58]
line_set_right_lower_half_space_between_magnet = [68]

line_set_vert_magnet_left = [51, 48, 45, 42, 39, 36, 33, 30, 27, 24]
line_set_vert_magnet_right = [49, 46, 43, 40, 37, 34, 31, 28, 25, 22]

line_set_ag_left = [69]
line_set_ag_right = [70]

line_set_left_back_iron = [55]
line_set_left_magnet_ag = [54]
line_set_right_back_iron = [57]
line_set_right_magnet_ag = [52]

line_set_top_edges = [56]
line_set_bottom_edges = [53]

line_sets_plane_stress_bc = {
    "interior": [
        line_set_upper_magnet_boundary,
        line_set_lower_magnet_boundary,
        line_set_upper_space_between_magnet,
        line_set_lower_space_between_magnet,
        line_set_ag_left,
        line_set_ag_right,
        line_set_left_back_iron,
        line_set_right_back_iron,
        line_set_left_upper_half_space_between_magnet,
        line_set_right_upper_half_space_between_magnet,
        line_set_left_lower_half_space_between_magnet,
        line_set_right_lower_half_space_between_magnet,
    ],
    "all": [
        line_set_vert_magnet_left,
        line_set_vert_magnet_right,
        line_set_left_magnet_ag,
        line_set_right_magnet_ag,
        line_set_top_edges,
        line_set_bottom_edges,
    ],
}

line_sets_dict = {
    "horizontal": [
        line_set_upper_magnet_boundary,
        line_set_lower_magnet_boundary,
        line_set_upper_space_between_magnet,
        line_set_lower_space_between_magnet,
        line_set_top_edges,
        line_set_bottom_edges,
        line_set_left_upper_half_space_between_magnet,
        line_set_right_upper_half_space_between_magnet,
        line_set_left_lower_half_space_between_magnet,
        line_set_right_lower_half_space_between_magnet,
    ],
    "vertical": [
        line_set_ag_left,
        line_set_ag_right,
        line_set_left_back_iron,
        line_set_left_magnet_ag,
        line_set_right_back_iron,
        line_set_right_magnet_ag,
        line_set_vert_magnet_left,
        line_set_vert_magnet_right,
    ],
}

for key, val in line_sets_dict.items():
    # Loop through each line tag in the set
    for line_set in val:
        # Loop through each line set in the key
        for tag in line_set:
            # Get the connectivity for the line segment
            edge = parser_outter_rotor.get_conn(f"LINE{tag}", "T3D2")

            # Get the unique node tags on each line (remove duplicates and keep the original ordering)
            edge_node_tags = np.array(list(dict.fromkeys(edge.flatten())))

            # Compunte the toal number of elements and nodes on the this edge
            nelems_edge = edge.shape[0]
            nnodes_edge = edge.shape[0] + 1

            # Print out info
            print(f"Tag: {tag}, nelems: {nelems_edge}, nnodes: {nnodes_edge}")
            print("Edge Node Tags (Unique):")
            print(edge_node_tags)

            # Add the components and their respective sizes
            model.add_component(
                f"truss_line_{tag}",
                size=nelems_edge,
                comp_obj=planar_truss,
            )

            model.add_component(
                f"node_src_line_{tag}",
                size=nnodes_edge,
                comp_obj=node_src_truss,
            )

            if key == "horizontal":
                model.add_component(
                    f"dirichlet_line_{tag}_x",
                    size=2,
                    comp_obj=dirichlet_bc_truss,
                )

                model.add_component(
                    f"dirichlet_line_{tag}_y",
                    size=nnodes_edge,
                    comp_obj=dirichlet_bc_truss,
                )

            elif key == "vertical":
                model.add_component(
                    f"dirichlet_line_{tag}_x",
                    size=nnodes_edge,
                    comp_obj=dirichlet_bc_truss,
                )

                model.add_component(
                    f"dirichlet_line_{tag}_y",
                    size=2,
                    comp_obj=dirichlet_bc_truss,
                )
            else:
                raise ValueError("Failed to add components")

            # Link the data and inputs
            for i in range(nelems_edge):
                model.link(
                    f"truss_line_{tag}.x_coord[{i}]",
                    f"node_src_line_{tag}.x_coord[{i}:{i+2}]",
                )
                model.link(
                    f"truss_line_{tag}.u_truss[{i}]",
                    f"node_src_line_{tag}.u_truss[[{i},{i+1}]]",
                )

                model.link(
                    f"truss_line_{tag}.y_coord[{i}]",
                    f"node_src_line_{tag}.y_coord[{i}:{i+2}]",
                )
                model.link(
                    f"truss_line_{tag}.v_truss[{i}]",
                    f"node_src_line_{tag}.v_truss[[{i},{i+1}]]",
                )

            if key == "horizontal":
                model.link(
                    f"node_src_line_{tag}.u_truss",
                    f"dirichlet_line_{tag}_x.dof",
                    src_indices=[0, -1],
                )
                model.link(
                    f"node_src_line_{tag}.v_truss",
                    f"dirichlet_line_{tag}_y.dof",
                )

            elif key == "vertical":
                model.link(
                    f"node_src_line_{tag}.v_truss",
                    f"dirichlet_line_{tag}_y.dof",
                    src_indices=[0, -1],
                )
                model.link(
                    f"node_src_line_{tag}.u_truss",
                    f"dirichlet_line_{tag}_x.dof",
                )
            else:
                raise ValueError("Failed to link dirichlet to node src")

####################
# Plane Stress Model
####################
# Plane stress components and links
model.add_component("ps", nelems_outter_rotor, ps)
model.add_component("ps_node_src", nnodes_outter_rotor, ps_node_src)

model.link("ps.x_coord", "ps_node_src.x_coord", tgt_indices=outter_rotor_glob_conn)
model.link("ps.y_coord", "ps_node_src.y_coord", tgt_indices=outter_rotor_glob_conn)

model.link("ps.u", "ps_node_src.u", tgt_indices=outter_rotor_glob_conn)
model.link("ps.v", "ps_node_src.v", tgt_indices=outter_rotor_glob_conn)

for key, val in line_sets_plane_stress_bc.items():
    # Loop through each line tag in the set
    for line_set in val:
        # Loop through each line set in the key
        for tag in line_set:
            # Get the connectivity for the line segment
            edge = parser_outter_rotor.get_conn(f"LINE{tag}", "T3D2")

            # Get the unique node tags on each line (remove duplicates and keep the original ordering)
            edge_node_tags = np.array(list(dict.fromkeys(edge.flatten())))

            # Compunte the toal number of elements and nodes on the this edge
            nelems_edge = edge.shape[0]
            nnodes_edge = edge.shape[0] + 1

            # Print out info
            # print(f"Tag: {tag}, nelems: {nelems_edge}, nnodes: {nnodes_edge}")
            # print("Edge Node Tags (Unique):")
            # print(edge_node_tags)
            if key == "all":
                model.link(
                    "ps_node_src.u",
                    f"node_src_line_{tag}.u_truss",
                    src_indices=edge_node_tags,
                )
                model.link(
                    "ps_node_src.v",
                    f"node_src_line_{tag}.v_truss",
                    src_indices=edge_node_tags,
                )
            elif key == "interior":
                model.link(
                    "ps_node_src.u",
                    f"node_src_line_{tag}.u_truss[{1}:{-1}]",
                    src_indices=edge_node_tags[1:-1],
                )
                model.link(
                    "ps_node_src.v",
                    f"node_src_line_{tag}.v_truss[{1}:{-1}]",
                    src_indices=edge_node_tags[1:-1],
                )

##############
# Build Module
##############
# Build module
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

# Define the data vector
data = model.get_data_vector()

############################################
# Set the problem data (x_coord and y_coord)
############################################
for key, val in line_sets_dict.items():
    # Loop through each line tag in the set
    for line_set in val:
        # Loop through each line set in the key
        for tag in line_set:
            # Get the connectivity for the line segment
            edge = parser_outter_rotor.get_conn(f"LINE{tag}", "T3D2")

            # Get the unique node tags on each line (remove duplicates and keep the original ordering)
            edge_node_tags = np.array(list(dict.fromkeys(edge.flatten())))

            # Set the x and y coordinates for each node src
            data[f"node_src_line_{tag}.x_coord"] = X_outter_rotor[edge_node_tags, 0]
            data[f"node_src_line_{tag}.y_coord"] = X_outter_rotor[edge_node_tags, 1]

data["ps_node_src.x_coord"] = X_outter_rotor[:, 0]
data["ps_node_src.y_coord"] = X_outter_rotor[:, 1]

# Define the mesh morph dirichlet bc values
morph_total_length = 0e-3 #! Link to space between magnets
morph_magent_length = 1.5e-3 #! link to morph_space_between_magnets
morph_magnet_offset = 0.0e-3 #! Link to morph length
morph_space_between_magnets = morph_magent_length # ! Link the morph_magnet_length
morph_back_iron_thickness = -8e-3
morph_upper_magnet_thickness = 2e-3
morph_lower_magnet_thickness = -1e-3
morph_airgap_thickness = 1.0e-3 + morph_lower_magnet_thickness

# Interior
for tag in line_set_upper_magnet_boundary:
    data[f"dirichlet_line_{tag}_x.offset[0]"] = -morph_magent_length  # Left node
    data[f"dirichlet_line_{tag}_x.offset[-1]"] = morph_magent_length  # Right node
    data[f"dirichlet_line_{tag}_y.offset[:]"] = morph_upper_magnet_thickness

# Interior
for tag in line_set_lower_magnet_boundary:
    data[f"dirichlet_line_{tag}_x.offset[0]"] = morph_magent_length  # Right node
    data[f"dirichlet_line_{tag}_x.offset[-1]"] = -morph_magent_length  # Left node
    data[f"dirichlet_line_{tag}_y.offset[:]"] = -morph_lower_magnet_thickness

# Interior
for tag in line_set_upper_space_between_magnet:
    data[f"dirichlet_line_{tag}_x.offset[0]"] = morph_space_between_magnets  # Left node
    data[f"dirichlet_line_{tag}_x.offset[-1]"] = -morph_space_between_magnets  # Right node
    data[f"dirichlet_line_{tag}_y.offset[:]"] = morph_upper_magnet_thickness

# Interior
for tag in line_set_lower_space_between_magnet:
    data[f"dirichlet_line_{tag}_x.offset[0]"] = -morph_space_between_magnets  # Right node
    data[f"dirichlet_line_{tag}_x.offset[-1]"] = morph_space_between_magnets  # Left node
    data[f"dirichlet_line_{tag}_y.offset[:]"] = -morph_lower_magnet_thickness

# All
for tag in line_set_vert_magnet_left:
    data[f"dirichlet_line_{tag}_y.offset[0]"] = -morph_lower_magnet_thickness  # bottom node
    data[f"dirichlet_line_{tag}_y.offset[-1]"] = morph_upper_magnet_thickness  # top node
    data[f"dirichlet_line_{tag}_x.offset[:]"] = -morph_space_between_magnets

# All
for tag in line_set_vert_magnet_right:
    data[f"dirichlet_line_{tag}_y.offset[0]"] = morph_upper_magnet_thickness  # top node
    data[f"dirichlet_line_{tag}_y.offset[-1]"] = -morph_lower_magnet_thickness  # bottom node
    data[f"dirichlet_line_{tag}_x.offset[:]"] = morph_space_between_magnets

# All
data[f"dirichlet_line_56_x.offset[0]"] = -morph_total_length # Node 45
data[f"dirichlet_line_56_x.offset[-1]"] = morph_total_length # Node 46
data[f"dirichlet_line_56_y.offset[:]"] = morph_back_iron_thickness

# All
data[f"dirichlet_line_53_x.offset[0]"] = morph_total_length  # Node 43
data[f"dirichlet_line_53_x.offset[-1]"] = -morph_total_length  # Node 44
data[f"dirichlet_line_53_y.offset[:]"] = -morph_airgap_thickness

# Interior
data[f"dirichlet_line_55_y.offset[0]"] = morph_upper_magnet_thickness # Node 1
data[f"dirichlet_line_55_y.offset[-1]"] = morph_back_iron_thickness # Node 45
data[f"dirichlet_line_55_x.offset[:]"] = -morph_total_length

# All
data[f"dirichlet_line_54_y.offset[0]"] = -morph_lower_magnet_thickness # Node 47
data[f"dirichlet_line_54_y.offset[-1]"] = morph_upper_magnet_thickness # Node 1
data[f"dirichlet_line_54_x.offset[:]"] = -morph_total_length

# Interior
data[f"dirichlet_line_69_y.offset[0]"] = -morph_airgap_thickness  # Node 44
data[f"dirichlet_line_69_y.offset[-1]"] = morph_lower_magnet_thickness  # Node 57
data[f"dirichlet_line_69_x.offset[:]"] = -morph_total_length

# Interior
data[f"dirichlet_line_57_y.offset[0]"] = morph_back_iron_thickness # Node 46
data[f"dirichlet_line_57_y.offset[-1]"] = morph_upper_magnet_thickness # Node 22
data[f"dirichlet_line_57_x.offset[:]"] = morph_total_length

# All
data[f"dirichlet_line_52_y.offset[0]"] = morph_upper_magnet_thickness  # Node 22
data[f"dirichlet_line_52_y.offset[-1]"] = -morph_lower_magnet_thickness  # Node 48
data[f"dirichlet_line_52_x.offset[:]"] = morph_total_length

# Interior
data[f"dirichlet_line_70_y.offset[0]"] = -morph_lower_magnet_thickness  # Node 48
data[f"dirichlet_line_70_y.offset[-1]"] = -morph_airgap_thickness  # Node 43
data[f"dirichlet_line_70_x.offset[:]"] = morph_total_length

# Interior
data[f"dirichlet_line_1_x.offset[0]"] = -morph_total_length  # Node 1
data[f"dirichlet_line_1_x.offset[-1]"] = -morph_magent_length # Node 2
data[f"dirichlet_line_1_y.offset[:]"] = morph_upper_magnet_thickness

# Interior
data[f"dirichlet_line_21_x.offset[0]"] = morph_magent_length  # Node 21
data[f"dirichlet_line_21_x.offset[-1]"] = morph_total_length # Node 22
data[f"dirichlet_line_21_y.offset[:]"] = morph_upper_magnet_thickness

# Interior
data[f"dirichlet_line_58_x.offset[0]"] = -morph_total_length  # Node 47
data[f"dirichlet_line_58_x.offset[-1]"] = -morph_magent_length # Node 23
data[f"dirichlet_line_58_y.offset[:]"] = -morph_lower_magnet_thickness


# Interior
data[f"dirichlet_line_68_x.offset[0]"] =  morph_magent_length  # Node 42
data[f"dirichlet_line_68_x.offset[-1]"] = morph_total_length # Node 48
data[f"dirichlet_line_68_y.offset[:]"] = -morph_lower_magnet_thickness


#############################
# Setup and solve the problem
#############################
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

# Extract the solution
u_outer_rotor = ans_local.get_array()[model.get_indices("ps_node_src.u")]
v_outer_rotor = ans_local.get_array()[model.get_indices("ps_node_src.v")]

print(u_outer_rotor)
print(max(u_outer_rotor))

#########################
# Plot the original motor
#########################
fig, ax = plt.subplots(nrows=2, figsize=(8, 2))
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
    fig=fig,
    ax=ax[0],
)

# Morphed motor geometry on a new plot
X_outter_rotor_morph = X_outter_rotor.copy()
X_outter_rotor_morph[:, 0] += u_outer_rotor
X_outter_rotor_morph[:, 1] += v_outer_rotor
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
    X_outter_rotor_morph,
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
    fig=fig,
    ax=ax[1],
)

plt.show()
