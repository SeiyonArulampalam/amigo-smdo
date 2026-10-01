import amigo as am
import numpy as np
from .fem_space import Space, Conformity, FunctionSpace, SolutionSpace
from .cell_types import CellType, DofLayout
from .mesh import Mesh


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

    def __init__(self, mesh: Mesh, solution_space: SolutionSpace):
        self.mesh = mesh
        self.solution_space = solution_space

        # Number of DOFs associated with each FunctionSpace
        #  (FunctionSpace) -> int
        self._num_dof = {}

        # Element connectivity for each function space
        #   (FunctionSpace, domain, CellType) -> ndarray
        # Each array has shape (num_elements, num_element_dof)
        self._dof_conn = {}

        # Save the dof signs for H(div) and H(curl) spaces
        #   (FunctionSpace, domain, CellType) -> ndarray [+/- 1] only
        self._dof_signs = {}

        # Mapping between the mesh entities and the dof
        # (FunctionSpace) -> mapping : dict {entity key -> dof}
        self._vertex_dof = {}
        self._edge_dof = {}
        self._face_dof = {}
        self._cell_dof = {}

        # Build the DOF numbering.
        self._build()

    def get_num_dof(self, func_space: FunctionSpace) -> int:
        """
        Return the total number of DOFs for a FunctionSpace.
        """
        return self._num_dof[func_space]

    def get_dof_conn(
        self, space: FunctionSpace, domain: str, cell_type: CellType
    ) -> np.ndarray:
        """
        Return element-local -> global DOF connectivity.

        Returns
        -------
        conn : ndarray
            Shape (num_elements, num_element_dof).
        """
        return self._dof_conn[(space, domain, cell_type)]

    def get_dof_signs(
        self, space: FunctionSpace, domain: str, cell_type: CellType
    ) -> np.ndarray:
        """
        Return element-local -> global DOF array of sign transformations.

        Returns
        -------
        signs : ndarray of +/- 1s
            Shape (num_elements, num_element_dof).
        """
        if space.func_space == Space.HDIV or space.func_space == Space.HCURL:
            return self._dof_signs[(space, domain, cell_type)]
        else:
            raise NotImplementedError

    def get_dof_in_domain(self, name: str, domain: str):
        """
        Return all the DOFs for the given variable name in the specified domain
        """

        # Get the function space associated with the variable name
        space = self.solution_space.get_space(name)

        all_dof = []
        for cell_type in self.mesh.get_cell_types(domain):
            all_dof.extend(self._dof_conn[(space, domain, cell_type)])

        return np.array(all_dof, dtype=np.int64)

    def _build(self):
        """
        Build the DOF numbering independently for each FunctionSpace.
        """

        for func_space in self.solution_space.get_spaces():
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

        # Build the layouts
        elem_layouts = {}
        for domain in self.mesh.get_domains():
            for cell_type in self.mesh.get_cell_types(domain):
                if not (func_space, cell_type) in elem_layouts:
                    elem_layouts[(func_space, cell_type)] = DofLayout.make_layout(
                        func_space, cell_type
                    )

        # For each cell type in each domain
        for domain in self.mesh.get_domains():
            for cell_type in self.mesh.get_cell_types(domain):
                layout = elem_layouts[(func_space, cell_type)]

                # Get the vertex connectivity
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

                # Create the connectivity
                conn = np.empty((nelem, layout.ndof), dtype=np.int64)

                # Create the signs array
                signs = None
                if (
                    func_space.func_space == Space.HDIV
                    or func_space.func_space == Space.HCURL
                ):
                    signs = np.ones((nelem, layout.ndof), dtype=float)

                for elem in range(nelem):
                    # Vertex DOFs
                    for vertex_index, local_dof in enumerate(layout.vertex_dofs):
                        entity_id = int(vertex_conn[elem, vertex_index])

                        key = self.make_entity_key(
                            func_space=func_space,
                            domain=domain,
                            entity_id=entity_id,
                            entity_dof=0,
                        )

                        if key not in vertex_dof:
                            vertex_dof[key] = next_dof
                            next_dof += 1

                        conn[elem, vertex_index] = vertex_dof[key]

                    # Edge DOFs
                    for edge_index, local_dofs in enumerate(layout.edge_dofs):
                        entity_id = int(edge_conn[elem, edge_index])

                        for entity_dof, local_dof in enumerate(local_dofs):
                            # Flip the edge orientation
                            edge_entity_dof = entity_dof
                            edge_sign = 1.0
                            if edge_orientation[elem, edge_index] < 0:
                                edge_entity_dof = len(local_dofs) - 1 - entity_dof
                                edge_sign = -1.0

                            key = self.make_entity_key(
                                func_space=func_space,
                                domain=domain,
                                entity_id=entity_id,
                                entity_dof=edge_entity_dof,
                            )

                            if key not in edge_dof:
                                edge_dof[key] = next_dof
                                next_dof += 1

                            conn[elem, local_dof] = edge_dof[key]
                            signs[elem, local_dof] = edge_sign

                    # Face DOFs
                    for local_face, local_dofs in enumerate(layout.face_dofs):
                        entity_id = int(face_conn[elem, local_face])

                        for entity_dof, local_dof in enumerate(local_dofs):
                            key = self.make_entity_key(
                                func_space=func_space,
                                domain=domain,
                                entity_id=entity_id,
                                entity_dof=entity_dof,
                            )

                            if key not in face_dof:
                                face_dof[key] = next_dof
                                next_dof += 1

                            conn[elem, local_dof] = face_dof[key]

                    # Cell-interior DOFs
                    for entity_dof, local_dof in enumerate(layout.cell_dofs):
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
                self._dof_signs[chunk_key] = signs

        self._num_dof[func_space] = next_dof
        self._vertex_dof[func_space] = vertex_dof
        self._edge_dof[func_space] = edge_dof
        self._face_dof[func_space] = face_dof
        self._cell_dof[func_space] = cell_dof
        return

    def get_vertex_dof_mapping(self, space: FunctionSpace):
        """Get a map for the mesh vertex ordering to the dof ordering"""
        return self._vertex_dof[space]

    def get_edge_dof_mapping(self, space: FunctionSpace):
        """Get a map for the mesh edge ordering to the dof ordering"""
        return self._edge_dof[space]

    def get_face_dof_mapping(self, space: FunctionSpace):
        """Get a map for the mesh face ordering to the dof ordering"""
        return self._face_dof[space]

    def get_cell_dof_mapping(self, space: FunctionSpace):
        """Get a map for the cell face ordering to the dof ordering"""
        return self._cell_dof[space]

    @staticmethod
    def make_entity_key(
        func_space: FunctionSpace,
        domain: str,
        entity_id: int,
        entity_dof: int,
    ):
        """
        Construct the key determining whether two element DOFs represent
        the same global DOF.

        GLOBAL
            DOFs are shared wherever the underlying mesh entity is shared.

        COMPONENT
            DOFs are shared only inside the same domain/component.
        """

        conformity = func_space.conformity
        if conformity == Conformity.GLOBAL:
            return (entity_id, entity_dof)
        elif conformity == Conformity.COMPONENT:
            return (domain, entity_id, entity_dof)

        raise ValueError(f"Unsupported conformity {conformity}")


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
    def __init__(
        self, mesh: Mesh, solution_space: SolutionSpace, kind="input", name="src"
    ):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.solution_space = solution_space
        self.kind = kind
        self.name = name

        # Create the DOF handler
        self.dof_handler = DofHandler(mesh, solution_space)

        return

    def get_dof_handler(self):
        return self.dof_handler

    def add_source(self, model: am.Model):

        for space in self.solution_space.get_spaces():
            names = self.solution_space.get_names(space)
            if len(names) == 0:
                continue

            if space.func_space == Space.CONST:
                # Create a sub-model for each data set
                sub_model = am.Model()
                for data_name in names:

                    domains = self.mesh.get_domains()

                    domain_names = [name for name in domains]
                    input_names, data_names, con_names = [], [], []
                    if self.kind == "input":
                        input_names = domain_names
                    elif self.kind == "data":
                        data_names = domain_names
                    elif self.kind == "multiplier":
                        con_names = [f"res_{name}" for name in domain_names]

                    dof_src = DofSource(
                        input_names=input_names,
                        con_names=con_names,
                        data_names=data_names,
                    )
                    sub_model.add_component(data_name, 1, dof_src)

                model.add_model(self.name, sub_model)

            else:
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
                ndof = self.dof_handler.get_num_dof(space)

                # Add the dof from the source mesh
                model.add_component(self.name, ndof, dof_src)

        return

    def link_dof(
        self, model: am.Model, domain: str, cell_type: CellType, elem_name: str
    ):
        for space in self.solution_space.get_spaces():
            names = self.solution_space.get_names(space)
            if len(names) == 0:
                continue

            if space.func_space == Space.CONST:
                for name in names:
                    model.link(
                        f"{self.name}.{name}.{domain}",
                        f"{elem_name}.{name}[:]",
                    )

            else:
                if self.kind == "multiplier":
                    con_names = [f"res_{name}" for name in names]
                    names = con_names

                # Get the connectivity for the function space
                conn = self.dof_handler.get_dof_conn(space, domain, cell_type)

                # Link the degrees of freedom
                if space.func_space == Space.HDIV:
                    for name in names:
                        model.link(
                            f"{self.name}.{name}",
                            f"{elem_name}.{name}[:, 0, :]",
                            src_indices=conn,
                        )
                    for name in names:
                        model.link(
                            f"{self.name}.{name}",
                            f"{elem_name}.{name}[:, 1, :]",
                            src_indices=conn,
                        )
                else:
                    for name in names:
                        model.link(
                            f"{self.name}.{name}",
                            f"{elem_name}.{name}",
                            src_indices=conn,
                        )

        return

    def get_vertex_dof_to_node(self, space):
        """
        Build a mapping from global DOF number to mesh vertex index for the
        vertex DOFs of a degree-1 H1 FunctionSpace.

        The DOF handler numbers DOFs in order of first encounter while
        traversing domains/elements, which is a permutation of the mesh node
        ordering (not the identity). To write nodal quantities such as the
        geometry coordinates into the data vector, we must scatter each mesh
        node's value into the slot of the DOF that represents it.

        Returns
        -------
        dof_to_node : np.ndarray
            Array of length get_num_dof(space) where dof_to_node[dof] is the
            mesh node index represented by that DOF.
        """
        ndof = self.dof_handler.get_num_dof(space)
        dof_to_node = np.full(ndof, -1, dtype=np.int64)

        # Build the layouts
        elem_layouts = {}
        for domain in self.mesh.get_domains():
            for cell_type in self.mesh.get_cell_types(domain):
                if not (space, cell_type) in elem_layouts:
                    elem_layouts[(space, cell_type)] = DofLayout.make_layout(
                        space, cell_type
                    )

        for domain in self.mesh.get_domains():
            for cell_type in self.mesh.get_cell_types(domain):
                # DOF connectivity (global DOF numbers) for this chunk
                conn = self.dof_handler.get_dof_conn(space, domain, cell_type)

                # Mesh vertex connectivity (global node numbers)
                vertex_conn = self.mesh.get_vertex_conn(domain, cell_type)

                layout = elem_layouts[(space, cell_type)]

                # The first len(vertex_dofs) local DOFs are the vertex DOFs,
                # placed at layout.vertex_dofs local positions and aligned with
                # the mesh vertex ordering.
                for local_vertex, local_dof in enumerate(layout.vertex_dofs):
                    dof_to_node[conn[:, local_dof]] = vertex_conn[:, local_vertex]

        return dof_to_node

    def get_node_coordinates(self, space, X):
        """
        Reorder mesh node coordinates X (shape (num_nodes, dim)) so that the
        result is indexed by global DOF number for a degree-1 H1 geometry
        space. coords[dof] == X[node_represented_by_dof].
        """
        dof_to_node = self.get_vertex_dof_to_node(space)
        if np.any(dof_to_node < 0):
            raise ValueError(
                "Some geometry DOFs are not associated with a mesh vertex; "
                "setting nodal coordinates requires a degree-1 H1 geometry space."
            )
        return X[dof_to_node]


class ScaledBC(am.Component):
    def __init__(self, name, input_name=[], scale=[1.0, 1.0]):
        super().__init__(name)

        if len(scale) != 2:
            raise ValueError("scale must be of length 2")

        self.input_name = input_name

        self.add_constant(f"scale_left", value=scale[0])
        self.add_constant(f"scale_right", value=scale[1])

        for name in self.input_name:
            self.add_input(f"{name}_left", value=1.0)
            self.add_input(f"{name}_right", value=1.0)
            self.add_constraint(f"res_{name}")

        return

    def compute(self):
        scale_left = self.constants["scale_left"]
        scale_right = self.constants["scale_right"]

        for name in self.input_name:
            self.constraints[f"res_{name}"] = (
                scale_left * self.inputs[f"{name}_left"]
                + scale_right * self.inputs[f"{name}_right"]
            )
        return


class BoundaryConditions:
    def __init__(
        self,
        bc_name: str,
        dof_handler: DofHandler,
        bc={},
        integrand_formulation: str = "potential",
    ):
        if not (
            bc["type"] == "dirichlet"
            or bc["type"] == "continuity"
            or bc["type"] == "scaled"
        ):
            typ = bc["type"]
            raise ValueError(f"Unrecognized boundary condition type {typ}")
        self.bc_name = bc_name
        self.dof_handler = dof_handler
        self.bc = bc
        self.integrand_formulation = integrand_formulation
        return

    def _get_target_dof(
        self,
        name: str,
        targets: list[str],
        start: bool = True,
        end: bool = True,
    ):
        all_dof = []
        for target in targets:
            dof = self.dof_handler.get_dof_in_domain(name, target)
            all_dof.extend(dof)

        unique = np.unique(all_dof)

        # unique = list(dict.fromkeys(all_dof))

        if not start or not end:
            raise NotImplementedError

            # This logic no longer works - need to find a better way
            # if not start:
            #     unique = unique[1:]
            # if not end:
            #     unique = unique[:-1]

        return unique

    # def _reorder_nodes(self, nodes_left, nodes_right):
    #     nodes_left = np.array(nodes_left)
    #     nodes_right = np.array(nodes_right)

    #     y_left = self.mesh.X[nodes_left, 1]
    #     y_right = self.mesh.X[nodes_right, 1]

    #     idx_left = np.argsort(y_left)
    #     idx_right = np.argsort(y_right)

    #     return nodes_left[idx_left], nodes_right[idx_right]

    # def _get_matched_nodes(self, targets, start=True, end=True):
    #     left_target_lines = targets[0]
    #     right_target_lines = targets[1]
    #     nodes_left = self._get_bc_nodes(left_target_lines, start, end)
    #     nodes_right = self._get_bc_nodes(right_target_lines, start, end)

    #     if len(nodes_left) != len(nodes_right):
    #         raise Exception(f"nnodes left != nnodes right")

    #     # Reorder the nodes to match
    #     return self._reorder_nodes(nodes_left, nodes_right)

    def add_bcs(self, model):
        """Add the boundary conditions to the model"""

        if self.bc["type"] == "dirichlet":
            input_names = self.bc["input"]
            for name in input_names:
                dof = self._get_target_dof(name, self.bc["target"])

                # Fix the DOF
                model.add_fixed(f"soln.{name}", dof)
                if self.integrand_formulation == "weak":
                    model.add_fixed(f"multiplier.res_{name}", dof)

        else:
            raise NotImplementedError
            # targets = self.bc["target"]
            # start = self.bc.get("start", True)
            # end = self.bc.get("end", True)

            # nodes_left, nodes_right = self._get_matched_nodes(
            #     targets, start=start, end=end
            # )

            # input_names = self.bc["input"]
            # if self.bc["type"] == "continuity":
            #     for name in input_names:
            #         model.link(
            #             f"soln.{name}",
            #             f"soln.{name}",
            #             src_indices=nodes_left,
            #             tgt_indices=nodes_right,
            #         )

            # elif self.bc["type"] == "scaled":
            #     scale = self.bc["scale"]
            #     class_name = f"ScaledBC_{self.bc_name}"
            #     bc_src = ScaledBC(class_name, input_names, scale=scale)

            #     if len(nodes_left) > 0:
            #         model.add_component(
            #             f"{self.bc_name}",
            #             len(nodes_left),
            #             bc_src,
            #         )

            #         for name in input_names:
            #             model.link(
            #                 f"soln.{name}",
            #                 f"{self.bc_name}.{name}_left",
            #                 src_indices=nodes_left,
            #             )
            #             model.link(
            #                 f"soln.{name}",
            #                 f"{self.bc_name}.{name}_right",
            #                 src_indices=nodes_right,
            #             )
