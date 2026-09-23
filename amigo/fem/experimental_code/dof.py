from dataclasses import dataclass
from reference_cells import CellType, ReferenceCell, REFERENCE_CELLS


@dataclass(frozen=True)
class EntityDofLayout:
    num_dofs: int
    points: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class ElementDofLayout:
    cell_type: CellType
    entity_dofs: tuple[tuple[EntityDofLayout, ...], ...]

    @property
    def reference_cell(self) -> ReferenceCell:
        return REFERENCE_CELLS[self.cell_type]

    def get_entity_dofs(
        self,
        dim: int,
        entity: int,
    ) -> EntityDofLayout:
        return self.entity_dofs[dim][entity]


class DoFHandler:
    # map local dof layout to the global degrees of freedom
    pass


def H1TriangleDof() -> ElementDofLayout:
    ref = REFERENCE_CELLS[CellType.TRIANGLE]

    vertex_dofs = []
    for v in ref.vertices:
        vertex_dofs.append(EntityDofLayout(num_dofs=1, points=(v)))

    edge_dofs = []
    for _ in ref.edges:
        edge_dofs.append(EntityDofLayout(num_dofs=0))

    return ElementDofLayout(
        cell_type=CellType.TRIANGLE,
        entity_dofs=(tuple(vertex_dofs), tuple(edge_dofs)),
    )


def RT0TriangleDof():
    ref = REFERENCE_CELLS[CellType.TRIANGLE]

    vertex_dofs = []
    for v in ref.vertices:
        vertex_dofs.append(EntityDofLayout(num_dofs=0))

    edge_dofs = []
    for e in ref.edges:
        edge_dofs.append(EntityDofLayout(num_dofs=1, points=(e)))

    return ElementDofLayout(
        cell_type=CellType.TRIANGLE,
        entity_dofs=(tuple(vertex_dofs), tuple(edge_dofs)),
    )


# a = H1TriangleDof()
# for i in a.entity_dofs:
#     print(i)

b = RT0TriangleDof()
for i in b.entity_dofs:
    print(i)
