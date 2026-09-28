import amigo as am
import numpy as np
from dataclasses import dataclass
from .fem_space import Space, SolutionSpace
from .cell_types import CellType
from .vandermonde import Vandermonde1D, Vandermonde2D, VecVandermonde2D
from .fem_space import FunctionSpace
from .cell_types import (
    CellType,
    ReferenceCell,
    REFERENCE_CELLS,
    make_h1_dofs,
    make_hdiv_dofs,
)
from ..expressions import Expr


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def triple_product(d, x, y, n=1):
    val = d[0] * x[0] * y[0]
    for i in range(1, n):
        val = val + d[i] * x[i] * y[i]
    return val


def curl_2d(x, y, n=1):
    return x[0] * y[1] - x[1] * y[0]


def mat_vec(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


def mat_vec_transpose(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[1][0] * x[1], A[0][1] * x[0] + A[1][1] * x[1]]


@dataclass(frozen=True)
class DofLayout:
    # Reference cell that defines the cell type
    ref_cell: ReferenceCell

    # Function space type
    space: FunctionSpace

    # Points in parametric space where the degrees of freedom are located
    pts: tuple[tuple[float, ...], ...] = ()

    # Directions associated with vector elements
    dirs: tuple[tuple[float, ...] | None, ...] = ()

    # Entity dofs associated with the vertices, edges, faces and interior points.
    # Dof are ordered as follows: vertice, edges, faces then interior dof
    vertex_dofs: tuple[tuple[int, ...], ...] = ()
    edge_dofs: tuple[tuple[int, ...], ...] = ()
    face_dofs: tuple[tuple[int, ...], ...] = ()
    interior_dofs: tuple[int, ...] = ()

    @property
    def ndof(self):
        return len(self.pts)


@dataclass
class ConstValue:
    value: Expr


@dataclass
class H1Value:
    value: Expr
    grad: list[Expr]


@dataclass
class HdivValue:
    vec: list[Expr]
    div: Expr


class Basis:
    def __init__(self, names: list[str], layout: DofLayout, kind: str = "input"):
        if isinstance(names, (list, tuple)):
            self.names = names
        elif isinstance(names, str):
            self.names = [names]
        self.layout = layout
        self.kind = kind

        if not (
            self.kind == "input" or self.kind == "data" or self.kind == "multiplier"
        ):
            raise ValueError(f"{self.kind} not recognized")

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        nnodes = self.layout.ndof
        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(nnodes,))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(nnodes,))
        elif self.kind == "multiplier":
            for name in self.names:
                comp.add_constraint(f"res_{name}", shape=(nnodes,))


class ConstantBasis(Basis):
    def __init__(self, names, space, kind="input"):

        layout = DofLayout(ref_cell=CellType.POINT, space=space, pts=([0.0]))
        super().__init__(names, layout=layout, kind=kind)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name].value
            soln[name] = ConstValue(value=value)
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

            soln[name] = ConstValue(value=u[0])

        return soln


class LagrangeBasis1D(Basis):
    def __init__(self, p, names, kind="input"):
        self.p = p
        nnodes = p + 1
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = np.linspace(-1, 1, nnodes)
        self.vand = Vandermonde1D(self.p, self.pts)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name].value
            grad = orig[name].value
            soln[name] = H1Value(value=value, grad=[Jinv * grad[0]])

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi = geo["x"].grad[0]
        y_xi = geo["y"].grad[1]

        detJ = am.sqrt(x_xi**2 + y_xi**2)
        Jinv = 1.0 / detJ
        J = detJ
        return detJ, J, Jinv

    def eval(self, comp, pt):
        xi = pt[0]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi)
        Nx = self.vand.eval_basis_grad(xi)

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = H1Value(
                value=dot_product(u, N, n=self.nnodes),
                grad=[dot_product(u, Nx, n=self.nnodes)],
            )

        return soln


class LagrangeBasis2D(Basis):
    def __init__(self, names: list[str], space: FunctionSpace, kind="input"):

        super().__init__(names, nnodes=nnodes, kind=kind)

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name].value
            grad = orig[name].grad
            soln[name] = H1Value(
                value=value, grad=mat_vec_transpose(Jinv, grad, n=2, m=2)
            )

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"].grad
        y_xi, y_eta = geo["y"].grad

        detJ = x_xi * y_eta - x_eta * y_xi
        J = [[x_xi, x_eta], [y_xi, y_eta]]
        inv = 1.0 / detJ
        Jinv = [[y_eta * inv, -x_eta * inv], [-y_xi * inv, x_xi * inv]]

        return detJ, J, Jinv


class TriangleLagrangeBasis(LagrangeBasis2D):
    def __init__(self, names: list[str], space: FunctionSpace, kind: str = "input"):
        p = space.degree
        if p < 0:
            raise ValueError(f"Degree {p} must be >= 0")

        ref_cell = REFERENCE_CELLS[CellType.TRIANGLE]

        pts, dirs, vertex_dofs, edge_dofs, face_dofs, interior_dofs = make_h1_dofs(
            ref_cell, p
        )

        layout = DofLayout(
            ref_cell=ref_cell,
            space=space,
            pts=pts,
            dirs=dirs,
            vertex_dofs=vertex_dofs,
            edge_dofs=edge_dofs,
            face_dofs=face_dofs,
            interior_dofs=interior_dofs,
        )

        super().__init__(names, space, layout, kind=kind)

        self.pts = self._get_tri_nodes(self.p)
        self.exps = self._get_monomial_exponents(self.p)
        self.vand = Vandermonde2D(nnodes, self.exps, self.pts)

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
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = H1Value(
                value=dot_product(u, N, n=self.nnodes),
                grad=[
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            )

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
        self.vand = Vandermonde2D(nnodes, self.exps, self.pts)

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
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = H1Value(
                value=dot_product(u, N, n=self.nnodes),
                grad=[
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            )

        return soln


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

            soln[name] = HdivValue(vec=[vx, vy], div=div)

        return soln

    def transform(self, detJ, J, Jinv, orig):
        soln = {}
        for name in orig:
            vec = orig[name].vec
            div = orig[name].div
            vx = (J[0][0] * vec[0] + J[0][1] * vec[1]) / detJ
            vy = (J[1][0] * vec[0] + J[1][1] * vec[1]) / detJ
            soln[name] = HdivValue(vec=[vx, vy], div=div / detJ)

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"].grad
        y_xi, y_eta = geo["y"].grad

        detJ = x_xi * y_eta - x_eta * y_xi
        J = [[x_xi, x_eta], [y_xi, y_eta]]
        inv = 1.0 / detJ
        Jinv = [[y_eta * inv, -x_eta * inv], [-y_xi * inv, x_xi * inv]]

        return detJ, J, Jinv


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
