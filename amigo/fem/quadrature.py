import numpy as np
from .fem_space import SolutionSpace
from .cell_types import CellType


class Quadrature:
    def get_args(self):
        return []


class TriangleQuadrature(Quadrature):
    def __init__(self, order):
        self.weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        self.points = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        self.args = [{"n": 0}, {"n": 1}, {"n": 2}]

        if order == 1:
            # 1-point (exact for degree 1)
            self.xi = np.array([1 / 3])
            self.eta = np.array([1 / 3])
            self.weights = np.array([1.0 / 2.0])

        elif order == 2:
            # 3-point (exact for degree 2)
            self.xi = np.array([1 / 6, 2 / 3, 1 / 6])
            self.eta = np.array([1 / 6, 1 / 6, 2 / 3])
            self.weights = np.array([1 / 6, 1 / 6, 1 / 6])

        elif order == 3:
            # 4-point (exact for degree 3)
            self.xi = np.array([1 / 3, 3 / 5, 1 / 5, 1 / 5])
            self.eta = np.array([1 / 3, 1 / 5, 3 / 5, 1 / 5])
            self.weights = np.array([-9 / 32, 25 / 96, 25 / 96, 25 / 96])

        elif order == 4:
            a = 0.445948490915965
            b = 0.108103018168070
            c = 0.091576213509771
            d = 0.816847572980459

            w1 = 0.223381589678011
            w2 = 0.109951743655322

            self.xi = np.array([a, a, b, c, c, d])
            self.eta = np.array([a, b, a, c, d, c])
            self.weights = np.array([w1, w1, w1, w2, w2, w2])
        else:
            raise NotImplementedError

        self.args = []
        for n in range(len(self.weights)):
            self.args.append({"n": n})

        return

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], [self.xi[n], self.eta[n]]


class QuadQuadrature(Quadrature):
    def __init__(self, npts):
        self.points, self.weights = np.polynomial.legendre.leggauss(npts)
        self.args = []

        for m in range(npts):
            for n in range(npts):
                self.args.append({"n": n, "m": m})

    def get_args(self):
        return self.args

    def get_point(self, n=0, m=0):
        wt = self.weights[n] * self.weights[m]
        pt = [self.points[n], self.points[m]]
        return wt, pt


class ReducedQuadQuadrature(Quadrature):
    def __init__(self):
        self.args = [{"n": 0, "m": 0}]
        self.points = np.array([0.0])
        self.weights = np.array([4.0])  # full area of biunit square

    def get_args(self):
        return self.args

    def get_point(self, n=0, m=0):
        return self.weights[0], [0.0, 0.0]


class LineQuadrature(Quadrature):
    def __init__(self, npts):
        pts, wts = np.polynomial.legendre.leggauss(npts)
        self.xi = pts
        self.weights = wts
        self.args = [{"n": n} for n in range(npts)]

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], [self.xi[n]]


def make_quadrature(space: SolutionSpace, cell_type: CellType) -> Quadrature:
    max_degree = 0
    for func in space.get_spaces():
        if func.degree > max_degree:
            max_degree = func.degree

    if cell_type == CellType.SEGMENT:
        return LineQuadrature(max_degree + 1)
    elif cell_type == CellType.TRIANGLE:
        return TriangleQuadrature(max_degree + 1)
    elif cell_type == CellType.QUADRILATERAL:
        return QuadQuadrature(max_degree + 1)
    else:
        raise NotImplementedError
