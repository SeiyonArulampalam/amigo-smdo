from dataclasses import dataclass
from enum import Enum, auto
from .fem_space import Space, FunctionSpace


class CellType(Enum):
    POINT = auto()
    SEGMENT = auto()
    TRIANGLE = auto()
    QUADRILATERAL = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()
    PRISM = auto()
    PYRAMID = auto()


@dataclass(frozen=True)
class ReferenceCell:
    cell_type: CellType
    dimension: int
    vertices: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[int, int], ...] = ()
    faces: tuple[tuple[int, ...], ...] = ()
    face_types: tuple[CellType, ...] = ()


REFERENCE_CELLS: dict[CellType, ReferenceCell] = {
    CellType.POINT: ReferenceCell(
        cell_type=CellType.POINT,
        dimension=0,
        vertices=((),),
    ),
    CellType.SEGMENT: ReferenceCell(
        cell_type=CellType.SEGMENT,
        dimension=1,
        vertices=(
            (-1.0,),  # 0
            (1.0,),  # 1
        ),
        edges=((0, 1),),
    ),
    CellType.TRIANGLE: ReferenceCell(
        cell_type=CellType.TRIANGLE,
        dimension=2,
        vertices=(
            (0.0, 0.0),  # 0
            (1.0, 0.0),  # 1
            (0.0, 1.0),  # 2
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 0),
        ),
        faces=((0, 1, 2),),
        face_types=(CellType.TRIANGLE,),
    ),
    CellType.QUADRILATERAL: ReferenceCell(
        cell_type=CellType.QUADRILATERAL,
        dimension=2,
        vertices=(
            (-1.0, -1.0),  # 0
            (1.0, -1.0),  # 1
            (1.0, 1.0),  # 2
            (-1.0, 1.0),  # 3
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ),
        faces=((0, 1, 2, 3),),
        face_types=(CellType.QUADRILATERAL,),
    ),
    CellType.TETRAHEDRON: ReferenceCell(
        cell_type=CellType.TETRAHEDRON,
        dimension=3,
        vertices=(
            (0.0, 0.0, 0.0),  # 0
            (1.0, 0.0, 0.0),  # 1
            (0.0, 1.0, 0.0),  # 2
            (0.0, 0.0, 1.0),  # 3
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (1, 3),
            (2, 3),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 2, 1),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ),
        face_types=(
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
        ),
    ),
    CellType.HEXAHEDRON: ReferenceCell(
        cell_type=CellType.HEXAHEDRON,
        dimension=3,
        vertices=(
            (-1.0, -1.0, -1.0),  # 0
            (1.0, -1.0, -1.0),  # 1
            (1.0, 1.0, -1.0),  # 2
            (-1.0, 1.0, -1.0),  # 3
            (-1.0, -1.0, 1.0),  # 4
            (1.0, -1.0, 1.0),  # 5
            (1.0, 1.0, 1.0),  # 6
            (-1.0, 1.0, 1.0),  # 7
        ),
        edges=(
            # Bottom
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 3, 2, 1),  # bottom
            (4, 5, 6, 7),  # top
            (0, 1, 5, 4),  # front
            (1, 2, 6, 5),  # right
            (2, 3, 7, 6),  # back
            (3, 0, 4, 7),  # left
        ),
        face_types=(
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
        ),
    ),
    CellType.PRISM: ReferenceCell(
        cell_type=CellType.PRISM,
        dimension=3,
        vertices=(
            (0.0, 0.0, -1.0),  # 0
            (1.0, 0.0, -1.0),  # 1
            (0.0, 1.0, -1.0),  # 2
            (0.0, 0.0, 1.0),  # 3
            (1.0, 0.0, 1.0),  # 4
            (0.0, 1.0, 1.0),  # 5
        ),
        edges=(
            # Bottom triangle
            (0, 1),
            (1, 2),
            (2, 0),
            # Top triangle
            (3, 4),
            (4, 5),
            (5, 3),
            # Vertical
            (0, 3),
            (1, 4),
            (2, 5),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 2, 1),  # bottom triangle
            (3, 4, 5),  # top triangle
            (0, 1, 4, 3),  # side
            (1, 2, 5, 4),  # side
            (2, 0, 3, 5),  # side
        ),
        face_types=(
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
        ),
    ),
    CellType.PYRAMID: ReferenceCell(
        cell_type=CellType.PYRAMID,
        dimension=3,
        vertices=(
            (-1.0, -1.0, 0.0),  # 0
            (1.0, -1.0, 0.0),  # 1
            (1.0, 1.0, 0.0),  # 2
            (-1.0, 1.0, 0.0),  # 3
            (0.0, 0.0, 1.0),  # 4: apex
        ),
        edges=(
            # Base
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Apex edges
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 3, 2, 1),  # base
            (0, 1, 4),  # front
            (1, 2, 4),  # right
            (2, 3, 4),  # back
            (3, 0, 4),  # left
        ),
        face_types=(
            CellType.QUADRILATERAL,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
        ),
    ),
}


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
    cell_dofs: tuple[int, ...] = ()

    @property
    def ndof(self):
        return len(self.pts)

    @classmethod
    def make_layout(cls, space: FunctionSpace, cell_type: CellType):
        if space.func_space == Space.CONST:
            return cls.make_const(space, cell_type)
        elif space.func_space == Space.H1:
            return cls.make_h1(space, cell_type)
        elif space.func_space == Space.HDIV:
            return cls.make_hdiv(space, cell_type)
        else:
            raise NotImplementedError

    @classmethod
    def make_const(cls, space: FunctionSpace, cell_type: CellType):
        ref_cell = REFERENCE_CELLS[cell_type]

        return cls(
            ref_cell=ref_cell,
            pts=((0.0, 0.0, 0.0),),
            space=space,
            cell_dofs=((0,)),
        )

    @classmethod
    def make_h1(cls, space: FunctionSpace, cell_type: CellType):
        """
        Construct nodal H1 Lagrange DOFs on a reference cell.

        The ordering is

            vertices
            edge interiors
            face interiors
            cell interiors

        Parameters
        ----------
        ref_cell : ReferenceCell
            Reference-cell topology and vertex coordinates.

        degree : int
            Polynomial degree p >= 1.

        Returns
        -------
        pts : list[tuple[float, ...]]
            Coordinates of all interpolation points.

        dirs : list[None]
            Direction information associated with each DOF.
            H1 scalar DOFs have no direction, so every entry is None.

        vertex_dofs : tuple[int, ...]
            DOF number associated with each vertex.

        edge_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each edge, ordered according
            to ref_cell.edges.

        face_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each face, ordered according
            to ref_cell.faces.

        cell_dofs : tuple[int, ...]
            DOFs in the interior of the cell.
        """

        ref_cell = REFERENCE_CELLS[cell_type]

        if space.degree < 1:
            raise ValueError("H1 Lagrange degree must be >= 1")

        p = space.degree
        verts = ref_cell.vertices

        pts = []
        dirs = []

        vertex_dofs = []
        edge_dofs = []
        face_dofs = []
        cell_dofs = []

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------

        def add_point(x):
            dof = len(pts)
            pts.append(tuple(float(v) for v in x))
            dirs.append(None)
            return dof

        def weighted_point(indices, weights):
            """
            Affine combination of reference-cell vertices.
            """
            dim = ref_cell.dimension

            return tuple(
                sum(weights[k] * verts[indices[k]][d] for k in range(len(indices)))
                for d in range(dim)
            )

        def edge_point(v0, v1, i):
            """
            Point i/p along the directed edge v0 -> v1.
            """
            t = i / p
            return weighted_point(
                (v0, v1),
                (1.0 - t, t),
            )

        def triangle_point(v0, v1, v2, i, j, denom=None):
            """
            Triangle point with barycentric coordinates

                lambda1 = i / denom
                lambda2 = j / denom
                lambda0 = 1 - lambda1 - lambda2
            """
            if denom is None:
                denom = p

            l1 = i / denom
            l2 = j / denom
            l0 = 1.0 - l1 - l2

            return weighted_point(
                (v0, v1, v2),
                (l0, l1, l2),
            )

        def quad_point(v0, v1, v2, v3, i, j, denom=None):
            """
            Tensor-product point on a quadrilateral.

            Vertex ordering is assumed to be cyclic:

                v3 ---- v2
                |       |
                |       |
                v0 ---- v1
            """
            if denom is None:
                denom = p

            s = i / denom
            t = j / denom

            return weighted_point(
                (v0, v1, v2, v3),
                (
                    (1.0 - s) * (1.0 - t),
                    s * (1.0 - t),
                    s * t,
                    (1.0 - s) * t,
                ),
            )

        # Vertices: all dimensions
        for v in range(len(verts)):
            vertex_dofs.append(add_point(verts[v]))

        # Edges: every topological edge, including a SEGMENT itself
        for v0, v1 in ref_cell.edges:
            dofs = []

            for i in range(1, p):
                dofs.append(add_point(edge_point(v0, v1, i)))

            edge_dofs.append(tuple(dofs))

        # Faces: every topological face, including a TRIANGLE/QUAD itself
        for face, face_type in zip(ref_cell.faces, ref_cell.face_types):
            dofs = []

            if face_type == CellType.TRIANGLE:
                v0, v1, v2 = face

                for j in range(1, p):
                    for i in range(1, p - j):
                        dofs.append(add_point(triangle_point(v0, v1, v2, i, j)))

            elif face_type == CellType.QUADRILATERAL:
                v0, v1, v2, v3 = face

                for j in range(1, p):
                    for i in range(1, p):
                        dofs.append(add_point(quad_point(v0, v1, v2, v3, i, j)))

            else:
                raise ValueError(f"Unsupported H1 face type: {face_type}")

            if len(dofs) > 0:
                face_dofs.append(tuple(dofs))

        # Cell interior
        cell_type = ref_cell.cell_type

        if cell_type == CellType.TETRAHEDRON:
            for k in range(1, p):
                for j in range(1, p - k):
                    for i in range(1, p - j - k):

                        l1 = i / p
                        l2 = j / p
                        l3 = k / p
                        l0 = 1.0 - l1 - l2 - l3

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    (0, 1, 2, 3),
                                    (l0, l1, l2, l3),
                                )
                            )
                        )

        elif cell_type == CellType.HEXAHEDRON:

            # Standard hexahedron ordering:
            #
            #       7-------6
            #      /|      /|
            #     4-------5 |
            #     | |     | |
            #     | 3-----|-2
            #     |/      |/
            #     0-------1
            #

            for k in range(1, p):
                z = k / p

                for j in range(1, p):
                    y = j / p

                    for i in range(1, p):
                        x = i / p

                        weights = (
                            (1 - x) * (1 - y) * (1 - z),
                            x * (1 - y) * (1 - z),
                            x * y * (1 - z),
                            (1 - x) * y * (1 - z),
                            (1 - x) * (1 - y) * z,
                            x * (1 - y) * z,
                            x * y * z,
                            (1 - x) * y * z,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    tuple(range(8)),
                                    weights,
                                )
                            )
                        )

        elif cell_type == CellType.PRISM:

            # Standard triangular-prism ordering:
            #
            # bottom: 0, 1, 2
            # top:    3, 4, 5
            #
            # with
            #
            #     0--3
            #     1--4
            #     2--5
            #

            for k in range(1, p):
                z = k / p

                for j in range(1, p):
                    for i in range(1, p - j):

                        l1 = i / p
                        l2 = j / p
                        l0 = 1.0 - l1 - l2

                        weights = (
                            (1 - z) * l0,
                            (1 - z) * l1,
                            (1 - z) * l2,
                            z * l0,
                            z * l1,
                            z * l2,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    tuple(range(6)),
                                    weights,
                                )
                            )
                        )

        elif cell_type == CellType.PYRAMID:

            # Standard ordering:
            #
            # base:  0,1,2,3
            # apex:  4
            #
            # Each horizontal layer is a shrinking quadrilateral.
            #
            # This gives the natural nodal layout for a pyramid, although
            # the polynomial/rational function space associated with a
            # high-order pyramid deserves separate treatment.

            v0, v1, v2, v3, va = 0, 1, 2, 3, 4

            for k in range(1, p):
                t = k / p

                # Number of intervals across this layer.
                q = p - k

                # q == 1 has no layer-interior points.
                for j in range(1, q):
                    eta = j / q

                    for i in range(1, q):
                        xi = i / q

                        # Point on the base quadrilateral.
                        base_weights = (
                            (1 - xi) * (1 - eta),
                            xi * (1 - eta),
                            xi * eta,
                            (1 - xi) * eta,
                        )

                        weights = (
                            (1 - t) * base_weights[0],
                            (1 - t) * base_weights[1],
                            (1 - t) * base_weights[2],
                            (1 - t) * base_weights[3],
                            t,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    (v0, v1, v2, v3, va),
                                    weights,
                                )
                            )
                        )

        return cls(
            ref_cell=ref_cell,
            space=space,
            pts=pts,
            dirs=dirs,
            vertex_dofs=tuple(vertex_dofs),
            edge_dofs=tuple(edge_dofs),
            face_dofs=tuple(face_dofs),
            cell_dofs=tuple(cell_dofs),
        )

    @classmethod
    def make_hdiv(cls, space: FunctionSpace, cell_type: CellType):
        """
        Construct nodal H(div) DOFs on a reference cell.

        The target spaces are the Raviart-Thomas family, numbered so that
        degree = 1 is the lowest-order element (matching TriangleRTBasis /
        QuadRTBasis):

            SEGMENT        P_k                      (k + 1 DOFs)
            TRIANGLE       RT_k                     (k (k + 2) DOFs)
            QUADRILATERAL  RTCF_k, Q_{k,k-1} x Q_{k-1,k}
            TETRAHEDRON    RT_k                     (k (k + 1) (k + 3) / 2 DOFs)
            HEXAHEDRON     NCF_k, Q_{k,k-1,k-1} x ...
            PRISM          lowest order only (degree = 1)
            PYRAMID        lowest order only (degree = 1)

        Each DOF is a point evaluation u(pt) . dir.

        Facet DOFs (edges in 2D, faces in 3D, vertices in 1D) evaluate the
        normal flux. Their direction is the outward normal scaled by the
        reference facet measure, n * |f|. Under the contravariant Piola map
        u_phys . n_phys |f_phys| = u_ref . n_ref |f_ref|, so this scaling makes
        facet DOFs agree between neighbouring cells of different reference
        types (e.g. the triangle hypotenuse gets (1, 1), the legs unit normals).

        Facet points are a unisolvent set for P_{k-1} (simplex facets) or
        Q_{k-1} (quadrilateral facets), taken strictly inside the facet.

        Interior DOFs evaluate the Cartesian components of u, ordered
        component-major. On simplices they sit on the interior points of a
        principal lattice (unisolvent for P_{k-2}). On tensor-product cells,
        component c uses i / k in direction c and i / (k + 1) in the others,
        which completes a tensor Lagrange grid together with the facet points.

        The ordering is

            vertices (SEGMENT only; no vertex DOFs otherwise)
            edge interiors (2D cells)
            face interiors (3D cells)
            cell interiors

        Parameters
        ----------
        ref_cell : ReferenceCell
            Reference-cell topology and vertex coordinates.

        degree : int
            Raviart-Thomas degree k >= 1.

        Returns
        -------
        pts : list[tuple[float, ...]]
            Coordinates of all DOF evaluation points.

        dirs : list[tuple[float, ...]]
            Direction the vector field is dotted with at each DOF.

        vertex_dofs : tuple[tuple[int, ...], ...]
            DOFs associated with each vertex (non-empty only for SEGMENT).

        edge_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each edge, ordered according
            to ref_cell.edges (non-empty only for 2D cells).

        face_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each face, ordered according
            to ref_cell.faces (non-empty only for 3D cells).

        cell_dofs : tuple[int, ...]
            DOFs in the interior of the cell.
        """

        ref_cell = REFERENCE_CELLS[cell_type]

        if space.degree < 1:
            raise ValueError("H(div) degree must be >= 1")

        k = space.degree
        dim = ref_cell.dimension
        verts = ref_cell.vertices
        cell_type = ref_cell.cell_type

        if cell_type == CellType.POINT:
            raise ValueError("H(div) DOFs are not defined on a POINT cell")

        if cell_type in (CellType.PRISM, CellType.PYRAMID) and k > 1:
            raise NotImplementedError(
                f"H(div) degree {k} is not implemented for {cell_type.name}; "
                "only the lowest-order (degree = 1) element is available"
            )

        pts = []
        dirs = []

        vertex_dofs = []
        edge_dofs = []
        face_dofs = []
        cell_dofs = []

        def add_dof(x, d):
            dof = len(pts)
            pts.append(tuple(float(v) for v in x))
            dirs.append(tuple(float(v) for v in d))
            return dof

        def weighted_point(indices, weights):
            """
            Affine combination of reference-cell vertices.
            """
            return tuple(
                sum(weights[m] * verts[indices[m]][c] for m in range(len(indices)))
                for c in range(dim)
            )

        def diff(a, b):
            """
            Vector from vertex b to vertex a.
            """
            return tuple(verts[a][c] - verts[b][c] for c in range(dim))

        def cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def unit(c):
            """
            Cartesian unit vector in direction c.
            """
            return tuple(1.0 if d == c else 0.0 for d in range(dim))

        def edge_point(v0, v1, t):
            return weighted_point((v0, v1), (1.0 - t, t))

        def triangle_point(v0, v1, v2, l1, l2):
            return weighted_point((v0, v1, v2), (1.0 - l1 - l2, l1, l2))

        def quad_point(v0, v1, v2, v3, s, t):
            """
            Bilinear point on a quadrilateral with cyclic vertex ordering:

                v3 ---- v2
                |       |
                |       |
                v0 ---- v1
            """
            return weighted_point(
                (v0, v1, v2, v3),
                (
                    (1.0 - s) * (1.0 - t),
                    s * (1.0 - t),
                    s * t,
                    (1.0 - s) * t,
                ),
            )

        def hex_point(x, y, z):
            return weighted_point(
                tuple(range(8)),
                (
                    (1 - x) * (1 - y) * (1 - z),
                    x * (1 - y) * (1 - z),
                    x * y * (1 - z),
                    (1 - x) * y * (1 - z),
                    (1 - x) * (1 - y) * z,
                    x * (1 - y) * z,
                    x * y * z,
                    (1 - x) * y * z,
                ),
            )

        # Facet normals (outward, scaled by the facet measure)
        def edge_normal_2d(v0, v1):
            # Edges are counterclockwise, so rotating the tangent by -90
            # degrees points outward; |t| is the edge length.
            tx, ty = diff(v1, v0)
            return (ty, -tx)

        def triangle_face_normal(v0, v1, v2):
            # Faces are counterclockwise viewed from outside; |cross| = 2 * area
            n = cross(diff(v1, v0), diff(v2, v0))
            return tuple(0.5 * c for c in n)

        def quad_face_normal(v0, v1, v2, v3):
            # Planar parallelogram face: |cross| = area
            return cross(diff(v1, v0), diff(v3, v0))

        # Vertices
        # Only a SEGMENT has vertex facets. The normal flux at an end point
        # is +/- u, pointing away from the opposite vertex.
        for v in range(len(verts)):
            dofs = []

            if cell_type == CellType.SEGMENT:
                other = 1 - v
                sign = 1.0 if verts[v][0] > verts[other][0] else -1.0
                dofs.append(add_dof(verts[v], (sign,)))

            vertex_dofs.append(tuple(dofs))

        # Edges
        # In 2D the edges are the facets: k normal-flux DOFs per edge at
        # t = i / (k + 1), unisolvent for P_{k-1} on the edge.
        # In 3D the edges carry no H(div) DOFs.
        if dim >= 2:
            for v0, v1 in ref_cell.edges:
                dofs = []

                if dim == 2:
                    normal = edge_normal_2d(v0, v1)

                    for i in range(1, k + 1):
                        t = i / (k + 1)
                        dofs.append(add_dof(edge_point(v0, v1, t), normal))

                edge_dofs.append(tuple(dofs))

        # Faces
        # In 3D the faces are the facets.
        #   triangle: strictly interior points of the lattice of order k + 2
        #             (unisolvent for P_{k-1})
        #   quad:     k x k tensor grid at i / (k + 1) (unisolvent for Q_{k-1})
        if dim == 3:
            if len(ref_cell.faces) != len(ref_cell.face_types):
                raise ValueError(
                    "ref_cell.faces and ref_cell.face_types must "
                    "have the same length"
                )

            for face, face_type in zip(ref_cell.faces, ref_cell.face_types):
                dofs = []

                if face_type == CellType.TRIANGLE:

                    v0, v1, v2 = face
                    normal = triangle_face_normal(v0, v1, v2)
                    n = k + 2

                    for j in range(1, n - 1):
                        for i in range(1, n - j):
                            pt = triangle_point(v0, v1, v2, i / n, j / n)
                            dofs.append(add_dof(pt, normal))

                elif face_type == CellType.QUADRILATERAL:

                    v0, v1, v2, v3 = face
                    normal = quad_face_normal(v0, v1, v2, v3)

                    for j in range(1, k + 1):
                        for i in range(1, k + 1):
                            pt = quad_point(v0, v1, v2, v3, i / (k + 1), j / (k + 1))
                            dofs.append(add_dof(pt, normal))

                else:
                    raise ValueError(f"Unsupported H(div) face type: {face_type}")

                face_dofs.append(tuple(dofs))

        # Cell interior
        if cell_type == CellType.SEGMENT:

            # Together with the two end points this gives k + 1 nodes for P_k
            for i in range(1, k):
                cell_dofs.append(add_dof(edge_point(0, 1, i / k), unit(0)))

        elif cell_type == CellType.TRIANGLE:

            # 2 * dim P_{k-2} DOFs: strictly interior points of the lattice
            # of order k + 1, both Cartesian components at each point.
            n = k + 1

            for c in range(2):
                for j in range(1, n - 1):
                    for i in range(1, n - j):
                        pt = triangle_point(0, 1, 2, i / n, j / n)
                        cell_dofs.append(add_dof(pt, unit(c)))

        elif cell_type == CellType.QUADRILATERAL:

            # u_x in Q_{k,k-1}: x at i / k (interior), y at j / (k + 1)
            # u_y in Q_{k-1,k}: x at i / (k + 1),      y at j / k (interior)
            for j in range(1, k + 1):
                for i in range(1, k):
                    pt = quad_point(0, 1, 2, 3, i / k, j / (k + 1))
                    cell_dofs.append(add_dof(pt, unit(0)))

            for j in range(1, k):
                for i in range(1, k + 1):
                    pt = quad_point(0, 1, 2, 3, i / (k + 1), j / k)
                    cell_dofs.append(add_dof(pt, unit(1)))

        elif cell_type == CellType.TETRAHEDRON:

            # 3 * dim P_{k-2} DOFs: strictly interior points of the lattice
            # of order k + 2, all three Cartesian components at each point.
            n = k + 2

            for c in range(3):
                for l in range(1, n - 2):
                    for j in range(1, n - 1 - l):
                        for i in range(1, n - j - l):
                            l1 = i / n
                            l2 = j / n
                            l3 = l / n
                            l0 = 1.0 - l1 - l2 - l3

                            pt = weighted_point((0, 1, 2, 3), (l0, l1, l2, l3))
                            cell_dofs.append(add_dof(pt, unit(c)))

        elif cell_type == CellType.HEXAHEDRON:

            # Component c: i / k (interior) in direction c and i / (k + 1)
            # in the two transverse directions.
            for c in range(3):
                counts = [(k + 1 if d != c else k) for d in range(3)]

                for l in range(1, counts[2]):
                    for j in range(1, counts[1]):
                        for i in range(1, counts[0]):
                            pt = hex_point(i / counts[0], j / counts[1], l / counts[2])
                            cell_dofs.append(add_dof(pt, unit(c)))

        elif cell_type in (CellType.PRISM, CellType.PYRAMID):
            # Lowest order: facet DOFs only
            pass

        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        return cls(
            ref_cell=ref_cell,
            space=space,
            pts=pts,
            dirs=dirs,
            vertex_dofs=tuple(vertex_dofs),
            edge_dofs=tuple(edge_dofs),
            face_dofs=tuple(face_dofs),
            cell_dofs=tuple(cell_dofs),
        )
