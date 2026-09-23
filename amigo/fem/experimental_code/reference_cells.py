from dataclasses import dataclass
from enum import Enum, auto
"""
dim = 0 -> entity = point   (0-D)
dim = 1 -> entity = edge    (1-D)
dim = 2 -> entity = face    (2-D)
dim = 3 -> entity = volume  (3-D)
"""

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

    def num_entities(self, dim: int) -> int:
        return len(self.entity_vertices(dim))

    def entity_vertices(self, dim: int) -> tuple[tuple[int, ...], ...]:
        """Return the local vertices defining each entity of dimension `dim`."""
        if dim < 0 or dim > self.dimension:
            raise ValueError(
                f"{self.cell_type.name} has no entities of dimension {dim}"
            )

        if dim == self.dimension:
            return (tuple(range(len(self.vertices))),)

        if dim == 0:
            return tuple((i,) for i in range(len(self.vertices)))

        if dim == 1:
            return self.edges

        if dim == 2:
            return self.faces

        raise ValueError(f"{self.cell_type.name} has no entities of dimension {dim}")

    def entity_type(self, dim: int, entity: int) -> CellType:
        entities = self.entity_vertices(dim)

        if not 0 <= entity < len(entities):
            raise IndexError(f"Entity {entity} is invalid for dimension {dim}")

        if dim == self.dimension:
            return self.cell_type

        if dim == 0:
            return CellType.POINT

        if dim == 1:
            return CellType.SEGMENT

        if dim == 2:
            return self.face_types[entity]

        raise ValueError(f"{self.cell_type.name} has no entities of dimension {dim}")


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
