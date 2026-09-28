import amigo as am
import numpy as np
from dataclasses import dataclass
from .fem_space import Space, Conformity, FunctionSpace, SolutionSpace
from .cell_types import CellType, ReferenceCell, REFERENCE_CELLS
from .basis import BasisCollection
from .mesh import Mesh


@dataclass
class EntityDofLayout:
    # Function space type
    space: FunctionSpace

    # Local degree of freedom number
    dof: tuple[int]

    # Points in parametric space where the degrees of freedom are located
    pts: tuple[tuple[float, ...], ...]

    # Directions associated with vector elements
    dirs: tuple[tuple[float, ...] | None, ...]

    # Entity dofs associated with the vertices, edges, faces and interior points.
    # Dof are ordered as follows: vertice, edges, faces then interior dof
    vertex_dofs: tuple[tuple[int, ...], ...]
    edge_dofs: tuple[tuple[int, ...], ...]
    face_dofs: tuple[tuple[int, ...], ...]
    interior_dofs: tuple[int, ...]


@dataclass
class ElementDofLayout:
    # Reference cell that defines the cell type
    ref_cell: ReferenceCell

    # Layouts for each of the finite-element spaces that belong to this element
    layouts: tuple[EntityDofLayout]

    def make_basis(self):
        objs = []

        for layout in self.layouts:
            p = layout.space.degree
            if layout.space.func_space == Space.H1:
                basis = make_lagrange_basis(self.ref_cell, p, layout.pts)
            elif layout.space.func_space == Space.HDIV:
                basis = make_hdiv_basis(self.ref_cell, p, layout.pts, layout.dirs)
            else:
                raise NotImplementedError(layout.space.func_space)

            objs.append(basis)

        return BasisCollection(objs)


class LagrangeH1Layout:
    def __init__(self, cell_type: CellType, degree: int = 1):
        self.cell_type = cell_type
        pts = self._build(cell_type, degree)

    def _build(self, cell_type: CellType, degree: int):
        ref = REFERENCE_CELLS[cell_type]

        # Convert the vertices to a numpy array
        verts = np.array(ref.vertices)

        # Add the vertex dof
        pts = []
        for v in range(len(ref.vertices)):
            pts.append(verts[v, :])

        # Add the edge dof
        for e in ref.edges:
            for i in range(1, degree):
                u = 1.0 * i / (degree + 1)
                p = (1.0 - u) * verts[e[0], :] + u * verts[e[1], :]
                pts.append(p)

        # Add the face dof
        for face in ref.faces:
            if len(face) == 3:
                # This is a triangle face
                raise NotImplementedError
            else:  # len(face) == 4
                raise NotImplementedError
        # Add any interior dof

        return pts


class DofHandler:
    """
    Construct canonical finite-element DOF numbering for each distinct
    FunctionSpace in a SolutionSpace.

    Field names are NOT represented here. For example, if u, v, and w all
    belong to the same H1(p=2, GLOBAL) FunctionSpace, they use the same
    canonical DOF connectivity.

    The DegreesOfFreedom class is responsible for applying that canonical
    numbering separately to u, v, w, etc.
    """

    def __init__(self, mesh: Mesh, space: SolutionSpace):
        self.mesh = mesh
        self.space = space

        # Number of DOFs associated with each FunctionSpace
        #
        #   FunctionSpace -> int
        #
        self._num_dof = {}

        # Element connectivity for each function space
        #
        #   (FunctionSpace, domain, CellType) -> ndarray
        #
        # Each array has shape
        #
        #   (num_elements, num_element_dof)
        #
        self._dof_conn = {}

        # Orientation information associated with each element chunk.
        #
        # These are mesh-topology orientations, not yet basis-specific
        # transformations.
        self._edge_orientation = {}
        self._face_orientation = {}

        # Build the DOF numbering.
        self._build()

    def get_num_dof(self, func_space: FunctionSpace) -> int:
        """
        Return the total number of canonical DOFs for a FunctionSpace.
        """
        return self._num_dof[func_space]

    def get_dof_conn(
        self, func_space: FunctionSpace, domain: str, cell_type: CellType
    ) -> np.ndarray:
        """
        Return element-local -> global DOF connectivity.

        Returns
        -------
        conn : ndarray
            Shape (num_elements, num_element_dof).
        """
        return self._dof_conn[(func_space, domain, cell_type)]

    def get_edge_orientation(
        self, func_space: FunctionSpace, domain: str, cell_type: CellType
    ):
        """
        Return the edge orientation information for this element chunk.

        For spaces that do not use edge DOFs, this may still be present
        because it comes directly from the mesh topology.
        """
        return self._edge_orientation.get((func_space, domain, cell_type), None)

    def get_face_orientation(
        self, func_space: FunctionSpace, domain: str, cell_type: CellType
    ):
        """
        Return the face orientation information for this element chunk.
        """
        return self._face_orientation.get((func_space, domain, cell_type), None)

    def _build(self):
        """
        Build the DOF numbering independently for each FunctionSpace.
        """

        for func_space in self.space.get_spaces():
            self._build_space(func_space)

    def _build_space(self, func_space: FunctionSpace):
        """
        Build one canonical global numbering for a FunctionSpace.
        """

        # These dictionaries identify DOFs associated with mesh entities.
        #
        # key -> global DOF number
        vertex_dof = {}
        edge_dof = {}
        face_dof = {}
        cell_dof = {}

        next_dof = 0

        for domain in self.mesh.get_domains():

            for cell_type in self.mesh.get_cell_types(domain):
                layout = self._get_element_layout(func_space, cell_type)
                vertex_conn = self.mesh.get_vertex_conn(domain, cell_type)
                nelem = vertex_conn.shape[0]

                # Edge/face topology may not exist for every reference cell.
                edge_conn = None
                edge_orientation = None

                if len(layout.edge_dofs) > 0:
                    edge_conn, edge_orientation = self.mesh.get_edge_conn(
                        domain, cell_type
                    )

                face_conn = None
                face_orientation = None

                if len(layout.face_dofs) > 0:
                    face_conn, face_orientation = self.mesh.get_face_conn(
                        domain, cell_type
                    )

                conn = np.empty((nelem, layout.ndof), dtype=np.int64)

                for elem in range(nelem):
                    # Vertex DOFs
                    for local_vertex, local_dofs in enumerate(layout.vertex_dofs):
                        entity_id = int(vertex_conn[elem, local_vertex])

                        for entity_dof, local_dof in enumerate(local_dofs):

                            key = self._make_entity_key(
                                func_space=func_space,
                                domain=domain,
                                cell_type=cell_type,
                                elem=elem,
                                entity_id=entity_id,
                                local_entity=local_vertex,
                                entity_dof=entity_dof,
                            )

                            if key not in vertex_dof:
                                vertex_dof[key] = next_dof
                                next_dof += 1

                            conn[elem, local_dof] = vertex_dof[key]

                    # Edge DOFs
                    for local_edge, local_dofs in enumerate(layout.edge_dofs):
                        entity_id = int(edge_conn[elem, local_edge])

                        for entity_dof, local_dof in enumerate(local_dofs):

                            key = self._make_entity_key(
                                func_space=func_space,
                                domain=domain,
                                cell_type=cell_type,
                                elem=elem,
                                entity_id=entity_id,
                                local_entity=local_edge,
                                entity_dof=entity_dof,
                            )

                            if key not in edge_dof:
                                edge_dof[key] = next_dof
                                next_dof += 1

                            conn[elem, local_dof] = edge_dof[key]

                    # Face DOFs
                    for local_face, local_dofs in enumerate(layout.face_dofs):
                        entity_id = int(face_conn[elem, local_face])

                        for entity_dof, local_dof in enumerate(local_dofs):

                            key = self._make_entity_key(
                                func_space=func_space,
                                domain=domain,
                                cell_type=cell_type,
                                elem=elem,
                                entity_id=entity_id,
                                local_entity=local_face,
                                entity_dof=entity_dof,
                            )

                            if key not in face_dof:
                                face_dof[key] = next_dof
                                next_dof += 1

                            conn[elem, local_dof] = face_dof[key]

                    # Cell-interior DOFs
                    for entity_dof, local_dof in enumerate(layout.interior_dofs):
                        # Interior DOFs belong to the element itself.
                        #
                        # The domain/cell_type/elem tuple uniquely
                        # identifies the element within the mesh chunk.
                        key = (domain, cell_type, elem, entity_dof)

                        if key not in cell_dof:
                            cell_dof[key] = next_dof
                            next_dof += 1

                        conn[elem, local_dof] = cell_dof[key]

                # Store element connectivity.
                chunk_key = (func_space, domain, cell_type)
                self._dof_conn[chunk_key] = conn

                if edge_orientation is not None:
                    self._edge_orientation[chunk_key] = edge_orientation

                if face_orientation is not None:
                    self._face_orientation[chunk_key] = face_orientation

        self._num_dof[func_space] = next_dof

    @staticmethod
    def _make_entity_key(
        func_space: FunctionSpace,
        domain: str,
        cell_type: CellType,
        elem: int,
        entity_id: int,
        local_entity: int,
        entity_dof: int,
    ):
        """
        Construct the key determining whether two element DOFs represent
        the same global DOF.

        GLOBAL
            DOFs are shared wherever the underlying mesh entity is shared.

        COMPONENT
            DOFs are shared only inside the same domain/component.

        DISCONTINUOUS
            No DOFs are shared between elements.
        """

        conformity = func_space.conformity
        if conformity == Conformity.GLOBAL:
            return (entity_id, entity_dof)
        elif conformity == Conformity.COMPONENT:
            return (domain, entity_id, entity_dof)
        elif conformity == Conformity.DISCONTINUOUS:
            return (domain, cell_type, elem, local_entity, entity_dof)

        raise ValueError(f"Unsupported conformity {conformity}")

    # ------------------------------------------------------------------
    # Element layout lookup
    # ------------------------------------------------------------------

    def _get_element_layout(
        self,
        func_space: FunctionSpace,
        cell_type: CellType,
    ):
        """
        Return the EntityDofLayout for this FunctionSpace and CellType.

        Replace the body of this method with however you store your
        reference-element layouts.
        """

        raise NotImplementedError(
            "Connect _get_element_layout() to the reference "
            "ElementDofLayout database."
        )


class DofSource(am.Component):
    def __init__(self, input_names=[], data_names=[], con_names=[], output_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)
        for name in output_names:
            self.add_output(name)

        return


class DegreesOfFreedom:
    def __init__(self, mesh: Mesh, space: SolutionSpace, kind="input", name="src"):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.space = space
        self.kind = kind
        self.name = name

        # Create the DOF handler
        self.dof_handler = DofHandler(mesh, space)

        return

    def add_source(self, model: am.Model):

        spaces = self.space.get_spaces()
        for func_space in spaces:
            names = self.space.get_names(func_space)
            if len(names) == 0:
                continue

            input_names, data_names, con_names = [], [], []
            if self.kind == "input":
                input_names = names
            elif self.kind == "data":
                data_names = names
            elif self.kind == "multiplier":
                con_names = [f"res_{name}" for name in names]

            # Create the source component
            dof_src = DofSource(
                input_names=input_names, con_names=con_names, data_names=data_names
            )

            # Get the number of degrees of freedom associated with the
            # associated space
            ndof = self.dof_handler.get_num_dof(func_space)

            # Add the dof from the source mesh
            model.add_component(self.name, ndof, dof_src)

        return

    def link_dof(
        self, model: am.Model, domain: str, cell_type: CellType, elem_name: str
    ):
        spaces = self.space.get_spaces()
        for func_space in spaces:
            names = self.space.get_names(func_space)
            if len(names) == 0:
                continue

            if self.kind == "multiplier":
                con_names = [f"res_{name}" for name in names]
                names = con_names

            # Get the connectivity for the function space
            conn = self.dof_handler.get_dof_conn(func_space, domain, cell_type)

            # Link the degrees of freedom
            for name in names:
                model.link(
                    f"{self.name}.{name}", f"{elem_name}.{name}", src_indices=conn
                )

            # TODO: Add the signs for H(div) here...
