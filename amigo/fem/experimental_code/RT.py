import numpy as np
import matplotlib.pyplot as plt


def triple_product(d, x, y, n=1):
    val = d[0] * x[0] * y[0]
    for i in range(1, n):
        val = val + d[i] * x[i] * y[i]
    return val


class RTBasis2D(Basis):
    def __init__(self, pts, dirs, uexps, vexps, names, kind="input"):
        super().__init__(names=names, nnodes=len(pts), kind=kind)
        self.pts = pts
        self.dirs = dirs
        self.uexps = uexps
        self.vexps = vexps

        self.vand = VecVandermonde2D(
            len(self.pts), self.uexps, self.vexps, self.pts, self.dirs
        )
        return

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(2, self.nnodes))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(2, self.nnodes))
        elif self.kind == "multiplier":
            for name in self.names:
                comp.add_constraint(f"res_{name}", shape=(2, self.nnodes))

        comp.add_data("directions", shape=(self.nnodes,))

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        d = self.data["directions"]
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            vx = d[0] * N[0, 0] * u[0, 0]
            vy = d[0] * N[1, 0] * u[1, 0]
            div = d[0] * (Nxi[0, 0] * u[0, 0] + Neta[1, 0] * u[0, 0])

            for i in range(1, self.n):
                vx = vx + d[i] * N[0, i] * u[0, i]
                vy = vy + d[i] * N[1, i] * u[1, i]
                div = div + d[i] * (Nxi[0, i] * u[0, i] + Neta[1, i] * u[0, i])

            soln[name] = {
                "vec": [vx, vy],
                "div": div,
            }

        return soln

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            vec = orig[name]["vec"]
            div = orig[name]["div"]
            vx = (J[0][0] * vec[0] + J[0][1] * vec[1]) / detJ
            vy = (J[1][0] * vec[0] + J[1][1] * vec[1]) / detJ
            soln[name] = {
                "vec": [vx, vy],
                "div": div / detJ,
            }

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"]["grad"]
        y_xi, y_eta = geo["y"]["grad"]

        detJ = x_xi * y_eta - x_eta * y_xi
        J = [[x_xi, x_eta], [y_xi, y_eta]]
        inv = 1.0 / detJ
        Jinv = [[y_eta * inv, -x_eta * inv], [-y_xi * inv, x_xi * inv]]

        return detJ, J, Jinv


class VecVandermonde2D:
    def __init__(self, n, uexps, vexps, pts, dirs):
        self.n = n
        self.uexps = uexps
        self.vexps = vexps
        self.pts = pts
        self.dirs = dirs
        self.C = self._build_vandermonde(self.n, self.pts, self.dirs)
        return

    def eval_basis(self, xi, eta):
        B = self._eval_polynomials(xi, eta)
        N = B @ self.C
        return N

    def eval_basis_grad(self, xi, eta):
        Bxi, Beta = self._eval_polynomial_grad(xi, eta)
        Nxi = Bxi @ self.C
        Neta = Beta @ self.C
        return Nxi, Neta

    def _eval_polynomials(self, xi, eta):
        """
        Return the basis function evaluated at point (xi, eta)
        shape = (2 , num basis functions)
        """
        B = np.zeros((2, self.n))
        for i, (u, v) in enumerate(zip(self.uexps, self.vexps)):
            if u[0] >= 0 and u[1] >= 0:
                B[0, i] = xi ** u[0] * eta ** u[1]
            if v[0] >= 0 and v[1] >= 0:
                B[1, i] = xi ** v[0] * eta ** v[1]

        return B

    def _eval_polynomial_grad(self, xi, eta):
        Bxi = np.zeros((2, self.n))
        Beta = np.zeros((2, self.n))
        for i, (u, v) in enumerate(zip(self.uexps, self.vexps)):
            if u[0] >= 0 and u[1] >= 0:
                # B[0, i] = xi ** u[0] * eta ** u[1]
                if u[0] >= 1:
                    Bxi[0, i] = u[0] * xi ** (u[0] - 1) * eta ** u[1]
                if u[1] >= 1:
                    Beta[0, i] = xi ** u[0] * u[1] * eta ** (u[1] - 1)

            if v[0] >= 0 and v[1] >= 0:
                # B[1, i] = xi ** v[0] * eta ** v[1]
                if v[0] >= 1:
                    Bxi[1, i] = v[0] * xi ** (v[0] - 1) * eta ** v[1]
                if v[1] >= 1:
                    Beta[1, i] = xi ** v[0] * v[1] * eta ** (v[1] - 1)

        return Bxi, Beta

    def _build_vandermonde(self, n, pts, dirs):
        V = np.zeros((n, n), dtype=float)
        for a, (xi, eta) in enumerate(pts):
            basis = self._eval_polynomials(xi, eta)
            V[a, :] = dirs[a] @ basis

        # Compute C = V^{-1}
        I = np.eye(n, dtype=float)
        return np.linalg.solve(V, I)

    def visualize(self, analytic=None):
        # Create the figure
        fig, ax = plt.subplots(ncols=self.n)

        # Define xi and eta within a unit right triangle
        rng = np.random.default_rng()

        # Evaluate the basis functions
        for i in range(500):
            a1 = rng.random()
            a2 = rng.random()

            if a1 + a2 > 1.0:
                # Flip about the hypotnuse
                a1, a2 = 1.0 - a1, 1.0 - a2

            # Compute the sample point
            sample = a1 * np.array([1, 0]) + a2 * np.array([0, 1])

            # Extract components
            xi = sample[0]
            eta = sample[1]

            # Evaluate the basis
            N = self.eval_basis(xi, eta)

            # Plot each basis function seperately
            for i in range(self.n):
                ax[i].quiver(xi, eta, N[0, i], N[1, i])

            if analytic is not None:
                N_exact = analytic(xi, eta)
                for i in range(self.n):
                    ax[i].quiver(
                        xi,
                        eta,
                        N_exact[0, i],
                        N_exact[1, i],
                        edgecolor="red",
                        facecolor="none",
                        linewidth=0.8,
                    )
        return


class QuadRTBasis(RTBasis2D):

    def __init__(self, p, names, kind="input"):
        if p == 1:
            pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]
            uexps = [(0, 0), (-1, -1), (1, 0), (0, 0)]
            vexps = [(-1, -1), (0, 0), (-1, -1), (0, 1)]
        elif p == 2:
            pts = []
            dirs = []
            uexps = []
            vexps = []
        else:
            raise ValueError(f"Degree {p} must be <= 2")

        super().__init__(pts, dirs, uexps, vexps, names, kind=kind)
        return


class TriangleRTBasis(RTBasis2D):
    def __init__(self, p, names, kind="input"):
        if p == 1:
            pts = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
            dirs = [[0, -1], [1, 1], [-1, 0]]
            uexps = [(0, 0), (-1, -1), (1, 0)]
            vexps = [(-1, -1), (0, 0), (0, 1)]
        elif p == 2:
            pts = [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1 / 3, 1 / 3],
                [1 / 3, 1 / 3],
            ]
            dirs = [
                [-1, 0],
                [0, -1],
                [0, -1],
                [1, 1],
                [1, 1],
                [-1, 0],
                [1, 0],
                [0, 1],
            ]
            uexps = [
                (0, 0),
                (-1, -1),
                (1, 0),
                (-1, -1),
                (0, 1),
                (-1, -1),
                (2, 0),
                (1, 1),
            ]
            vexps = [
                (-1, -1),
                (0, 0),
                (-1, -1),
                (1, 0),
                (-1, -1),
                (0, 1),
                (1, 1),
                (0, 2),
            ]
        else:
            raise ValueError(f"Degree {p} must be <= 2")

        super().__init__(pts, dirs, uexps, vexps, names, kind=kind)
        return


if __name__ == "__main__":
    # RT0
    # pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    # dirs = [[0, -1], [1 / np.sqrt(2), 1 / np.sqrt(2)], [-1, 0]]
    # uexps = [(0, 0), (-1, -1), (1, 0)]
    # vexps = [(-1, -1), (0, 0), (0, 1)]
    # vand = VecVandermonde2D(len(pts), uexps, vexps, pts, dirs)
    # vand.visualize()

    # RT1
    pts = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1 / 3, 1 / 3],
        [1 / 3, 1 / 3],
    ]
    dirs = [
        [-1, 0],
        [0, -1],
        [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1, 0],
        [1, 0],
        [0, 1],
    ]
    uexps = [(0, 0), (-1, -1), (1, 0), (-1, -1), (0, 1), (-1, -1), (2, 0), (1, 1)]
    vexps = [(-1, -1), (0, 0), (-1, -1), (1, 0), (-1, -1), (0, 1), (1, 1), (0, 2)]
    vand = VecVandermonde2D(len(pts), uexps, vexps, pts, dirs)
    vand.visualize()

    plt.show()
