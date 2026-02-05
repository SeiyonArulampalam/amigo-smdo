import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import time
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix  # For visualization
from parser import InpParser
from tabulate import tabulate
from scipy.sparse.linalg import spsolve
from linear_tri_elements import (
    eval_shape_funcs,
    dot,
    compute_detJ,
    compute_shape_derivs,
)

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


class Maxwell(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Material for each element
        self.add_data("alpha")

        # Define inputs to the problem
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract inputs
        u = self.inputs["u"]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea, Nx, Ny, detJ = compute_shape_derivs(xi, eta, X, Y)

        # Extract material
        alpha = self.data["alpha"]

        # Compute the local element residual
        K00 = qwts[n] * detJ * alpha * (Nx[0] * Nx[0] + Ny[0] * Ny[0])
        K01 = qwts[n] * detJ * alpha * (Nx[0] * Nx[1] + Ny[0] * Ny[1])
        K02 = qwts[n] * detJ * alpha * (Nx[0] * Nx[2] + Ny[0] * Ny[2])

        K10 = qwts[n] * detJ * alpha * (Nx[1] * Nx[0] + Ny[1] * Ny[0])
        K11 = qwts[n] * detJ * alpha * (Nx[1] * Nx[1] + Ny[1] * Ny[1])
        K12 = qwts[n] * detJ * alpha * (Nx[1] * Nx[2] + Ny[1] * Ny[2])

        K20 = qwts[n] * detJ * alpha * (Nx[2] * Nx[0] + Ny[2] * Ny[0])
        K21 = qwts[n] * detJ * alpha * (Nx[2] * Nx[1] + Ny[2] * Ny[1])
        K22 = qwts[n] * detJ * alpha * (Nx[2] * Nx[2] + Ny[2] * Ny[2])

        res = [
            K00 * u[0] + K01 * u[1] + K02 * u[2],
            K10 * u[0] + K11 * u[1] + K12 * u[2],
            K20 * u[0] + K21 * u[1] + K22 * u[2],
        ]

        #! Multiply by 0.5: 0.5 u^T K u
        self.objective["obj"] = (
            0.5 * u[0] * res[0] + 0.5 * u[1] * res[1] + 0.5 * u[2] * res[2]
        )
        return


class Coil(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))
        self.add_data("Jz")

        # Constants
        self.add_constant("mu0", value=4 * np.pi * 10**-7)

        # Add input
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        # Objective
        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Extract inputs
        u = self.inputs["u"]

        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea, Nx, Ny, detJ = compute_shape_derivs(xi, eta, X, Y)

        # Extract data
        Jz = self.data["Jz"]

        # Extract constants
        mu0 = self.constants["mu0"]
        res = [
            -mu0 * qwts[n] * detJ * Jz * N[0],
            -mu0 * qwts[n] * detJ * Jz * N[1],
            -mu0 * qwts[n] * detJ * Jz * N[2],
        ]
        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]
        return


class Magnets(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # Add input
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Magnetization data
        self.add_data("Mx")
        self.add_data("My")

        # Constants
        self.add_constant("mu0", value=4 * np.pi * 10**-7)

        # Objective
        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract inputs
        u = self.inputs["u"]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea, Nx, Ny, detJ = compute_shape_derivs(xi, eta, X, Y)

        # Extract data
        Mx = self.data["Mx"]
        My = self.data["My"]

        # Extract constants
        mu0 = self.constants["mu0"]

        # Constraint
        res = [
            -mu0 * (Mx * Ny[0] - My * Nx[0]) * detJ * qwts[n],
            -mu0 * (Mx * Ny[1] - My * Nx[1]) * detJ * qwts[n],
            -mu0 * (Mx * Ny[2] - My * Nx[2]) * detJ * qwts[n],
        ]
        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]
        return


class DirichletBc(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("u", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        self.objective["obj"] = self.inputs["u"] * self.inputs["lam"]
        return


class AntiSymmBc(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("u1", value=1.0)
        self.add_input("u2", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        self.objective["obj"] = (self.inputs["u1"] + self.inputs["u2"]) * self.inputs[
            "lam"
        ]
        return


class SymmBc(am.Component):
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


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")
        self.add_data("y_coord")

        # States
        self.add_input("u")
        return


class MaterialSource(am.Component):
    def __init__(self):
        super().__init__()
        self.add_data("alpha")
        self.add_data("Jz_phase1_pos")  # Out of the page
        self.add_data("Jz_phase2_pos")  # Out of the page
        self.add_data("Jz_phase3_pos")  # Out of the page
        self.add_data("Jz_phase1_neg")  # Into the page
        self.add_data("Jz_phase2_neg")  # Into the page
        self.add_data("Jz_phase3_neg")  # Into the page
        self.add_data("Mx_NS")
        self.add_data("My_NS")
        self.add_data("Mx_SN")
        self.add_data("My_SN")
        return


# class NodeSourceMorph(am.Component):
#     def __init__(self):
#         super().__init__()

#         # Mesh coordinates
#         self.add_data("x_coord")
#         self.add_data("y_coord")

#         # States
#         self.add_input("u")
#         self.add_input("v")
#         return


# class DirichletBcMorph(am.Component):
#     def __init__(self):
#         super().__init__()
#         self.add_input("dof", value=1.0)
#         self.add_input("lam", value=1.0)
#         self.add_objective("morph_obj")
#         return

#     def compute(self):
#         self.objective["morph_obj"] = self.inputs["dof"] * self.inputs["lam"]
#         return


# class DirichletBcMorphToothThicknessPos_dy(am.Component):
#     def __init__(self):
#         super().__init__()
#         self.add_input("dof", value=1.0)
#         self.add_input("lam", value=1.0)
#         self.add_objective("morph_obj")
#         return

#     def compute(self):

#         self.objective["morph_obj"] = (self.inputs["dof"] - 5e-3) * self.inputs["lam"]
#         return


# class DirichletBcMorphToothThicknessNeg_dy(am.Component):
#     def __init__(self):
#         super().__init__()
#         self.add_input("dof", value=1.0)
#         self.add_input("lam", value=1.0)
#         self.add_objective("morph_obj")
#         return

#     def compute(self):

#         self.objective["morph_obj"] = (self.inputs["dof"] + 5e-3) * self.inputs["lam"]
#         return


# class DirichletBcMorphAirGapPos_dy(am.Component):
#     def __init__(self):
#         super().__init__()
#         self.add_input("dof", value=1.0)
#         self.add_input("lam", value=1.0)
#         self.add_objective("morph_obj")
#         return

#     def compute(self):

#         self.objective["morph_obj"] = (self.inputs["dof"] + 2e-3) * self.inputs["lam"]
#         return


# class DirichletBcMorphAirGapNeg_dy(am.Component):
#     def __init__(self):
#         super().__init__()
#         self.add_input("dof", value=1.0)
#         self.add_input("lam", value=1.0)
#         self.add_objective("morph_obj")
#         return

#     def compute(self):

#         self.objective["morph_obj"] = (self.inputs["dof"] - 2e-3) * self.inputs["lam"]
#         return


# class PointForce(am.Component):
#     def __init__(self, force=1.0):
#         super().__init__()
#         self.force = force
#         self.add_input("dof")
#         self.add_objective("morph_obj")

#     def compute(self):
#         self.objective["morph_obj"] = -self.force * self.inputs["dof"]


# class PlaneStress(am.Component):
#     def __init__(self):
#         super().__init__()

#         # Add keyword arguments for the compute function
#         args = []
#         for n in range(3):
#             args.append({"n": n})
#         self.set_args(args)

#         # x/y coords for each node
#         self.add_data("x_coord", shape=(3,))
#         self.add_data("y_coord", shape=(3,))

#         # Material for each element
#         self.add_constant("E", value=1.0e3)  # Young's Modulus
#         self.add_constant("t", value=1.0)  # Thickness
#         self.add_constant("nu", value=0.3)  # Poisson's Ratio

#         # Define inputs to the problem (displacements)
#         self.add_input("u", shape=(3,), value=0.0)
#         self.add_input("v", shape=(3,), value=0.0)

#         self.add_objective("morph_obj")
#         return

#     def compute(self, n=None):
#         # Define gauss quad weights and points
#         qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
#         qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
#         xi, eta = qxi_qeta[n]

#         # Extract inputs
#         u = self.inputs["u"]

#         # Extract mesh data
#         X = self.data["x_coord"]
#         Y = self.data["y_coord"]
#         N, N_xi, N_ea, Nx, Ny, detJ = compute_shape_derivs(xi, eta, X, Y)

#         N1x = Nx[0]
#         N2x = Nx[1]
#         N3x = Nx[2]

#         N1y = Ny[0]
#         N2y = Ny[1]
#         N3y = Ny[2]

#         # Extract material constants
#         E = self.constants["E"]
#         t = self.constants["t"]
#         nu = self.constants["nu"]

#         # Extract inputs
#         u = self.inputs["u"]
#         v = self.inputs["v"]

#         u1 = u[0]
#         u2 = u[1]
#         u3 = u[2]

#         v1 = v[0]
#         v2 = v[1]
#         v3 = v[2]

#         # Compute local element stiffness matrix residual
#         # Plane-stress model coeff
#         coeff = qwts[n] * detJ * E * t / (1 - nu * nu)
#         alpha = 0.5 * (1 - nu)

#         # Compute B.T * D * B * u (SymPy math eqns)
#         # ! Multiply by coeff
#         res1 = coeff * (
#             N1x * N1y * v1 * (alpha + nu)
#             + u1 * (N1x**2 + N1y**2 * alpha)
#             + u2 * (N1x * N2x + N1y * N2y * alpha)
#             + u3 * (N1x * N3x + N1y * N3y * alpha)
#             + v2 * (N1x * N2y * nu + N1y * N2x * alpha)
#             + v3 * (N1x * N3y * nu + N1y * N3x * alpha)
#         )
#         res2 = coeff * (
#             N1x * N1y * u1 * (alpha + nu)
#             + u2 * (N1x * N2y * alpha + N1y * N2x * nu)
#             + u3 * (N1x * N3y * alpha + N1y * N3x * nu)
#             + v1 * (N1x**2 * alpha + N1y**2)
#             + v2 * (N1x * N2x * alpha + N1y * N2y)
#             + v3 * (N1x * N3x * alpha + N1y * N3y)
#         )
#         res3 = coeff * (
#             N2x * N2y * v2 * (alpha + nu)
#             + u1 * (N1x * N2x + N1y * N2y * alpha)
#             + u2 * (N2x**2 + N2y**2 * alpha)
#             + u3 * (N2x * N3x + N2y * N3y * alpha)
#             + v1 * (N1x * N2y * alpha + N1y * N2x * nu)
#             + v3 * (N2x * N3y * nu + N2y * N3x * alpha)
#         )
#         res4 = coeff * (
#             N2x * N2y * u2 * (alpha + nu)
#             + u1 * (N1x * N2y * nu + N1y * N2x * alpha)
#             + u3 * (N2x * N3y * alpha + N2y * N3x * nu)
#             + v1 * (N1x * N2x * alpha + N1y * N2y)
#             + v2 * (N2x**2 * alpha + N2y**2)
#             + v3 * (N2x * N3x * alpha + N2y * N3y)
#         )
#         res5 = coeff * (
#             N3x * N3y * v3 * (alpha + nu)
#             + u1 * (N1x * N3x + N1y * N3y * alpha)
#             + u2 * (N2x * N3x + N2y * N3y * alpha)
#             + u3 * (N3x**2 + N3y**2 * alpha)
#             + v1 * (N1x * N3y * alpha + N1y * N3x * nu)
#             + v2 * (N2x * N3y * alpha + N2y * N3x * nu)
#         )
#         res6 = coeff * (
#             N3x * N3y * u3 * (alpha + nu)
#             + u1 * (N1x * N3y * nu + N1y * N3x * alpha)
#             + u2 * (N2x * N3y * nu + N2y * N3x * alpha)
#             + v1 * (N1x * N3x * alpha + N1y * N3y)
#             + v2 * (N2x * N3x * alpha + N2y * N3y)
#             + v3 * (N3x**2 * alpha + N3y**2)
#         )

#         # res = K*u
#         res = [
#             res1,
#             res2,
#             res3,
#             res4,
#             res5,
#             res6,
#         ]

#         # Objective is 0.5 * uT * K * u = 0.5 * uT * res
#         self.objective["morph_obj"] = (
#             0.5 * res1 * u1
#             + 0.5 * res2 * v1
#             + 0.5 * res3 * u2
#             + 0.5 * res4 * v2
#             + 0.5 * res5 * u3
#             + 0.5 * res6 * v3
#         )
