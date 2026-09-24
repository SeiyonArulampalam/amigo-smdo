import amigo as am
import numpy as np
from .fem_space import Space, SolutionSpace
from .cell_types import CellType


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def curl_2d(x, y, n=1):
    return x[0] * y[1] - x[1] * y[0]


def mat_vec(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


def mat_vec_transpose(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[1][0] * x[1], A[0][1] * x[0] + A[1][1] * x[1]]


def eval_1d_monomials(p, xi):
    pows = np.ones(p + 1)
    for i in range(1, p + 1):
        pows[i] = pows[i - 1] * xi
    return pows


def eval_1d_monomial_grad(p, xi):
    pows = np.ones(p + 1)
    for i in range(1, p + 1):
        pows[i] = pows[i - 1] * xi

    grad = np.zeros(p + 1)
    for i in range(1, p + 1):
        grad[i] = i * pows[i - 1]
    return grad


def build_1d_lagrange_vandermonde(p, pts):
    n = p + 1
    V = np.zeros((n, n), dtype=float)
    for a, xi in enumerate(pts):
        V[a, :] = eval_1d_monomials(p, xi)

    # Compute C = V^{-1}
    I = np.eye(n, dtype=float)
    return np.linalg.solve(V, I)


def eval_2d_monomials(p, xi, eta, exps):
    """out[k] = xi^i * eta^j for each (i,j)."""
    xi_pows = np.ones(p + 1)
    eta_pows = np.ones(p + 1)

    for a in range(1, p + 1):
        xi_pows[a] = xi_pows[a - 1] * xi
    for b in range(1, p + 1):
        eta_pows[b] = eta_pows[b - 1] * eta

    out = np.empty(len(exps), dtype=float)
    for k, (i, j) in enumerate(exps):
        out[k] = xi_pows[i] * eta_pows[j]

    return out


def eval_2d_monomial_grad(p, xi, eta, exps):
    """grads[k] = [i * xi^{i-1} * eta^{j}, j * xi^{i} * eta^{j-1}]"""
    xi_pows = np.ones(p + 1)
    eta_pows = np.ones(p + 1)

    for a in range(1, p + 1):
        xi_pows[a] = xi_pows[a - 1] * xi
    for b in range(1, p + 1):
        eta_pows[b] = eta_pows[b - 1] * eta

    grad = np.zeros((len(exps), 2), dtype=float)
    for k, (i, j) in enumerate(exps):
        if i > 0:
            grad[k, 0] = i * xi_pows[i - 1] * eta_pows[j]
        if j > 0:
            grad[k, 1] = j * xi_pows[i] * eta_pows[j - 1]

    return grad


def build_2d_lagrange_vandermonde(n, p, pts, exps):
    # Build Vandermonde: V[a,k] = m_k(node_a)
    V = np.zeros((n, n), dtype=float)
    for a, (xi, eta) in enumerate(pts):
        V[a, :] = eval_2d_monomials(p, xi, eta, exps)

    # Compute C = V^{-1}
    I = np.eye(n, dtype=float)
    return np.linalg.solve(V, I)


class Basis:
    def __init__(self, names, nnodes=1, kind="input"):
        if isinstance(names, (list, tuple)):
            self.names = names
        elif isinstance(names, str):
            self.names = [names]
        self.nnodes = nnodes
        self.kind = kind

        if not (
            self.kind == "input" or self.kind == "data" or self.kind == "multiplier"
        ):
            raise ValueError(f"{self.kind} not recognized")

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(self.nnodes,))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(self.nnodes,))
        elif self.kind == "multiplier":
            for name in self.names:
                comp.add_constraint(f"res_{name}", shape=(self.nnodes,))


class ConstantBasis(Basis):
    def __init__(self, names, nnodes=1, kind="input"):
        super().__init__(names, nnodes=nnodes, kind=kind)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name]["value"]
            soln[name] = {
                "value": value,
            }
        return soln

    def eval(self, comp, pt):
        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = {
                "value": u[0],
            }
        return soln


class LagrangeBasis1D(Basis):
    def __init__(self, p, names, kind="input"):
        self.p = p
        nnodes = p + 1
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = np.linspace(-1, 1, nnodes)
        self.C = build_1d_lagrange_vandermonde(self.p, self.pts)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name]["value"]
            grad = orig[name]["grad"]
            soln[name] = {
                "value": value,
                "grad": [Jinv * grad[0]],
            }
        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi = geo["x"]["grad"][0]
        y_xi = geo["y"]["grad"][0]

        detJ = am.sqrt(x_xi**2 + y_xi**2)
        Jinv = 1.0 / detJ
        J = detJ
        return detJ, J, Jinv

    def eval(self, comp, pt):
        xi = pt[0]

        # Evaluate the monomials
        m = eval_1d_monomials(self.p, xi)
        N = m @ self.C

        mgrad = eval_1d_monomial_grad(self.p, xi)
        Nx = mgrad @ self.C
        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [dot_product(u, Nx, n=self.nnodes)],
            }

        return soln


class LagrangeBasis2D(Basis):
    def __init__(self, names, nnodes=1, kind="input"):
        super().__init__(names, nnodes=nnodes, kind=kind)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name]["value"]
            grad = orig[name]["grad"]
            soln[name] = {
                "value": value,
                "grad": mat_vec_transpose(Jinv, grad, n=2, m=2),
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


class TriangleLagrangeBasis(LagrangeBasis2D):
    def __init__(self, p, names, kind="input"):
        if p < 0:
            raise ValueError(f"Degree {p} must be >= 0")

        self.p = p
        nnodes = (p + 1) * (p + 2) // 2
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = self._get_tri_nodes(self.p)
        self.exps = self._get_monomial_exponents(self.p)
        self.C = build_2d_lagrange_vandermonde(self.nnodes, self.p, self.pts, self.exps)

        return

    def _get_tri_nodes(self, p):
        """Get the node locations"""
        pts = []

        if p == 0:
            return [[1 / 3, 1 / 3]]
        else:
            # Set the vertices
            pts = [[0, 0], [1, 0], [0, 1]]

            # At points on the edges
            edges = [[0, 1], [1, 2], [2, 0]]
            for a, b in edges:
                for i in range(1, p):
                    t = i / p
                    xi = (1 - t) * pts[a][0] + t * pts[b][0]
                    eta = (1 - t) * pts[a][1] + t * pts[b][1]
                    pts.append([xi, eta])

            # Add the remaining points in the interior
            for i in range(1, p):
                for j in range(1, p - i):
                    k = p - i - j
                    xi = j / p
                    eta = k / p
                    pts.append((xi, eta))

        return np.array(pts, dtype=float)

    def _get_monomial_exponents(self, p):
        exps = []
        for i in range(p + 1):
            for j in range(p + 1 - i):
                exps.append((i, j))
        return exps

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        m = eval_2d_monomials(self.p, xi, eta, self.exps)
        N = m @ self.C

        # Evaluate the derivatives of the monomials
        mgrad = eval_2d_monomial_grad(self.p, xi, eta, self.exps)
        Nxi = mgrad[:, 0] @ self.C
        Neta = mgrad[:, 1] @ self.C

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        return soln


class QuadLagrangeBasis(LagrangeBasis2D):
    def __init__(self, p, names, kind="input"):
        if p < 0:
            raise ValueError(f"Degree {p} must be >= 0")

        self.p = p
        nnodes = (p + 1) * (p + 1)
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = self._get_quad_nodes(self.p)
        self.exps = self._get_monomial_exponents(self.p)
        self.C = build_2d_lagrange_vandermonde(self.nnodes, self.p, self.pts, self.exps)

        return

    def _get_quad_nodes(self, p):
        """Get the node locations"""
        pts = []

        if p == 0:
            return [[0, 0]]
        else:
            # Set the vertices
            pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

            # At points on the edges
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
            for a, b in edges:
                for i in range(1, p):
                    t = i / p
                    xi = (1 - t) * pts[a][0] + t * pts[b][0]
                    eta = (1 - t) * pts[a][1] + t * pts[b][1]
                    pts.append([xi, eta])

            # Add the remaining points in the interior
            for j in range(1, p):
                for i in range(1, p):
                    xi = -1 + 2 * i / p
                    eta = -1 + 2 * j / p
                    pts.append((xi, eta))

        return np.array(pts, dtype=float)

    def _get_monomial_exponents(self, p):
        exps = []
        for j in range(p + 1):
            for i in range(p + 1):
                exps.append((i, j))
        return exps

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        m = eval_2d_monomials(self.p, xi, eta, self.exps)
        N = m @ self.C

        # Evaluate the derivatives of the monomials
        mgrad = eval_2d_monomial_grad(self.p, xi, eta, self.exps)
        Nxi = mgrad[:, 0] @ self.C
        Neta = mgrad[:, 1] @ self.C

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        return soln


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

        comp.add_data("signs", shape=(self.nnodes,))

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        d = comp.data["signs"]
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

            for i in range(1, self.nnodes):
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


class BasisCollection:
    def __init__(self, basis=[]):
        self.basis = basis

    def add_declarations(self, comp):
        for basis in self.basis:
            basis.add_declarations(comp)

    def eval(self, comp, pt):
        soln = {}
        for basis in self.basis:
            soln.update(basis.eval(comp, pt))
        return soln

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for basis in self.basis:
            soln.update(basis.transform(detJ, J, Jinv, orig))
        return soln

    def compute_transform(self, geo):
        return self.basis[0].compute_transform(geo)


def make_basis(
    space: SolutionSpace, cell_type: CellType, kind: str = "input"
) -> BasisCollection:
    objs = []

    for func in space.get_spaces():
        degree = func.degree
        names = space.get_names(func)

        obj = None
        if func.func_space == Space.CONST:
            obj = ConstantBasis(names)
        elif func.func_space == Space.H1:
            if cell_type == CellType.TRIANGLE:
                obj = TriangleLagrangeBasis(degree, names, kind=kind)
            elif cell_type == CellType.QUADRILATERAL:
                obj = QuadLagrangeBasis(degree, names, kind=kind)
            else:
                raise NotImplementedError
        elif func.func_space == Space.HDIV:
            if cell_type == CellType.TRIANGLE:
                obj = TriangleRTBasis(degree, names, kind=kind)
            elif cell_type == CellType.QUADRILATERAL:
                obj = QuadRTBasis(degree, names, kind=kind)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        objs.append(obj)

    return BasisCollection(objs)
