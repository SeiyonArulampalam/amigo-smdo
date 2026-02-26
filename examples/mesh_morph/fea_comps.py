import amigo as am
import triangle_basis
import line_basis


class PDE1Dx(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(1):
            args.append({"n": n})
        self.set_args(args)

        # xcoords for each node
        self.add_data("x_coord", shape=(2,))

        # Material for each element
        self.add_constant("E", value=1.0)  # Young's Modulus
        self.add_constant("A", value=1.0)  # Area

        # Define inputs to the problem (displacements)
        self.add_input("dx", shape=(2,), value=0.0)

        self.add_objective("morph_obj")
        return

    def compute(self, n=None):
        # Define gauss quad weights and points
        qwts = [2.0]
        xi = 0.0

        # Extract inputs
        u = self.inputs["dx"]

        # Extract mesh data
        X = self.data["x_coord"]
        N, N_xi, Nx, detJ = line_basis.compute_shape_derivs(xi, X)

        # Extract material constants
        E = self.constants["E"]
        A = self.constants["A"]

        # Compute local element stiffness matrix residual
        coeff = qwts[n] * detJ * E * A

        # K matrix
        K00 = coeff * Nx[0] * Nx[0]
        K01 = coeff * Nx[0] * Nx[1]
        K10 = coeff * Nx[1] * Nx[0]
        K11 = coeff * Nx[1] * Nx[1]

        # R = Ku
        res = [
            K00 * u[0] + K01 * u[1],
            K10 * u[0] + K11 * u[1],
        ]

        # 0.5 u^T K u
        self.objective["morph_obj"] = 0.5 * u[0] * res[0] + 0.5 * u[1] * res[1]

        return


class PlaneStress(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Material for each element
        self.add_constant("E", value=1.0e3)  # Young's Modulus
        self.add_constant("t", value=1.0)  # Thickness
        self.add_constant("nu", value=0.3)  # Poisson's Ratio

        # Define inputs to the problem (displacements)
        self.add_input("u", shape=(3,), value=0.0)
        self.add_input("v", shape=(3,), value=0.0)

        self.add_objective("morph_obj")
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
        N, N_xi, N_ea, Nx, Ny, detJ = triangle_basis.compute_shape_derivs(xi, eta, X, Y)

        N1x = Nx[0]
        N2x = Nx[1]
        N3x = Nx[2]

        N1y = Ny[0]
        N2y = Ny[1]
        N3y = Ny[2]

        # Extract material constants
        E = self.constants["E"]
        t = self.constants["t"]
        nu = self.constants["nu"]

        # Extract inputs
        u = self.inputs["u"]
        v = self.inputs["v"]

        u1 = u[0]
        u2 = u[1]
        u3 = u[2]

        v1 = v[0]
        v2 = v[1]
        v3 = v[2]

        # Compute local element stiffness matrix residual
        # Plane-stress model coeff
        coeff = qwts[n] * detJ * E * t / (1 - nu * nu)
        alpha = 0.5 * (1 - nu)

        # Compute B.T * D * B * u (SymPy math eqns)
        # ! Multiply by coeff
        res1 = coeff * (
            N1x * N1y * v1 * (alpha + nu)
            + u1 * (N1x**2 + N1y**2 * alpha)
            + u2 * (N1x * N2x + N1y * N2y * alpha)
            + u3 * (N1x * N3x + N1y * N3y * alpha)
            + v2 * (N1x * N2y * nu + N1y * N2x * alpha)
            + v3 * (N1x * N3y * nu + N1y * N3x * alpha)
        )
        res2 = coeff * (
            N1x * N1y * u1 * (alpha + nu)
            + u2 * (N1x * N2y * alpha + N1y * N2x * nu)
            + u3 * (N1x * N3y * alpha + N1y * N3x * nu)
            + v1 * (N1x**2 * alpha + N1y**2)
            + v2 * (N1x * N2x * alpha + N1y * N2y)
            + v3 * (N1x * N3x * alpha + N1y * N3y)
        )
        res3 = coeff * (
            N2x * N2y * v2 * (alpha + nu)
            + u1 * (N1x * N2x + N1y * N2y * alpha)
            + u2 * (N2x**2 + N2y**2 * alpha)
            + u3 * (N2x * N3x + N2y * N3y * alpha)
            + v1 * (N1x * N2y * alpha + N1y * N2x * nu)
            + v3 * (N2x * N3y * nu + N2y * N3x * alpha)
        )
        res4 = coeff * (
            N2x * N2y * u2 * (alpha + nu)
            + u1 * (N1x * N2y * nu + N1y * N2x * alpha)
            + u3 * (N2x * N3y * alpha + N2y * N3x * nu)
            + v1 * (N1x * N2x * alpha + N1y * N2y)
            + v2 * (N2x**2 * alpha + N2y**2)
            + v3 * (N2x * N3x * alpha + N2y * N3y)
        )
        res5 = coeff * (
            N3x * N3y * v3 * (alpha + nu)
            + u1 * (N1x * N3x + N1y * N3y * alpha)
            + u2 * (N2x * N3x + N2y * N3y * alpha)
            + u3 * (N3x**2 + N3y**2 * alpha)
            + v1 * (N1x * N3y * alpha + N1y * N3x * nu)
            + v2 * (N2x * N3y * alpha + N2y * N3x * nu)
        )
        res6 = coeff * (
            N3x * N3y * u3 * (alpha + nu)
            + u1 * (N1x * N3y * nu + N1y * N3x * alpha)
            + u2 * (N2x * N3y * nu + N2y * N3x * alpha)
            + v1 * (N1x * N3x * alpha + N1y * N3y)
            + v2 * (N2x * N3x * alpha + N2y * N3y)
            + v3 * (N3x**2 * alpha + N3y**2)
        )

        # res = K*u
        res = [
            res1,
            res2,
            res3,
            res4,
            res5,
            res6,
        ]

        # Objective is 0.5 * uT * K * u = 0.5 * uT * res
        self.objective["morph_obj"] = (
            0.5 * res1 * u1
            + 0.5 * res2 * v1
            + 0.5 * res3 * u2
            + 0.5 * res4 * v2
            + 0.5 * res5 * u3
            + 0.5 * res6 * v3
        )


class NodeSourcePlaneStress(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")
        self.add_data("y_coord")

        # States
        self.add_input("u")
        self.add_input("v")
        return


class NodeSourceTruss_x(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")

        # States
        self.add_input("dx")
        return


class NodeSourceTruss_y(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("y_coord")

        # States
        self.add_input("dy")
        return


class DirichletBcNode7x(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("dx", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("morph_obj")
        return

    def compute(self):
        self.objective["morph_obj"] = (self.inputs["dx"] + 0.6) * self.inputs["lam"]
        return


class DirichletBcNode4x(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("dx", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("morph_obj")
        return

    def compute(self):
        self.objective["morph_obj"] = (self.inputs["dx"] - 0.8) * self.inputs["lam"]
        return
