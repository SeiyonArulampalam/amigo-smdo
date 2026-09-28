from dataclasses import dataclass
from enum import Enum, auto


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


def make_h1_dofs(ref_cell: ReferenceCell, degree: int):
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

    interior_dofs : tuple[int, ...]
        DOFs in the interior of the cell.
    """

    if degree < 1:
        raise ValueError("H1 Lagrange degree must be >= 1")

    p = degree
    verts = ref_cell.vertices

    pts = []
    dirs = []

    vertex_dofs = []
    edge_dofs = []
    face_dofs = []
    interior_dofs = []

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

    # ------------------------------------------------------------
    # Vertices
    # ------------------------------------------------------------

    for v in range(len(verts)):
        vertex_dofs.append(add_point(verts[v]))

    # ------------------------------------------------------------
    # Proper edges
    #
    # For cells of dimension >= 2, ref_cell.edges are proper
    # subentities. For a SEGMENT, the cell itself is the 1D entity,
    # so its p-1 non-vertex DOFs are handled as cell interiors below.
    # ------------------------------------------------------------

    if ref_cell.dimension >= 2:
        for v0, v1 in ref_cell.edges:
            dofs = []

            for i in range(1, p):
                dofs.append(add_point(edge_point(v0, v1, i)))

            edge_dofs.append(tuple(dofs))

    # ------------------------------------------------------------
    # Proper faces
    #
    # Only 3D cells have proper 2D face subentities.
    # ------------------------------------------------------------

    if ref_cell.dimension == 3:
        if len(ref_cell.faces) != len(ref_cell.face_types):
            raise ValueError(
                "ref_cell.faces and ref_cell.face_types must " "have the same length"
            )

        for face, face_type in zip(ref_cell.faces, ref_cell.face_types):
            dofs = []

            if face_type == CellType.TRIANGLE:

                v0, v1, v2 = face

                # Strictly positive barycentric coordinates.
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

            face_dofs.append(tuple(dofs))

    # ------------------------------------------------------------
    # Cell interior
    # ------------------------------------------------------------

    cell_type = ref_cell.cell_type

    if cell_type == CellType.POINT:
        # The single vertex already represents the point element.
        pass

    elif cell_type == CellType.SEGMENT:

        v0, v1 = 0, 1

        for i in range(1, p):
            interior_dofs.append(add_point(edge_point(v0, v1, i)))

    elif cell_type == CellType.TRIANGLE:

        v0, v1, v2 = 0, 1, 2

        for j in range(1, p):
            for i in range(1, p - j):
                interior_dofs.append(add_point(triangle_point(v0, v1, v2, i, j)))

    elif cell_type == CellType.QUADRILATERAL:

        v0, v1, v2, v3 = 0, 1, 2, 3

        for j in range(1, p):
            for i in range(1, p):
                interior_dofs.append(add_point(quad_point(v0, v1, v2, v3, i, j)))

    elif cell_type == CellType.TETRAHEDRON:

        # Standard tetrahedral vertex ordering:
        #
        #     0, 1, 2, 3
        #
        # Use strictly positive barycentric coordinates
        #
        #     lambda0 + lambda1 + lambda2 + lambda3 = 1
        #

        for k in range(1, p):
            for j in range(1, p - k):
                for i in range(1, p - j - k):

                    l1 = i / p
                    l2 = j / p
                    l3 = k / p
                    l0 = 1.0 - l1 - l2 - l3

                    interior_dofs.append(
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

                    interior_dofs.append(
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

                    interior_dofs.append(
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

                    interior_dofs.append(
                        add_point(
                            weighted_point(
                                (v0, v1, v2, v3, va),
                                weights,
                            )
                        )
                    )

    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")

    return (
        pts,
        dirs,
        tuple(vertex_dofs),
        tuple(edge_dofs),
        tuple(face_dofs),
        tuple(interior_dofs),
    )
