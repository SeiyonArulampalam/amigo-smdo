import amigo as am
from .cell_types import CellType, DofLayout, REFERENCE_CELLS
import numpy as np
from .element import FiniteElement, FiniteElementOutput
from .fem_space import Space, SolutionSpace
from .basis import make_basis
from .dof_handler import DofSource, DegreesOfFreedom, BoundaryConditions
from .quadrature import make_quadrature, ReducedQuadQuadrature
import pyvista as pv


class Problem:
    def __init__(
        self,
        mesh,
        soln_space: SolutionSpace,
        data_space: SolutionSpace,
        geo_space: SolutionSpace,
        integrand_map=None,
        integrand_formulation="potential",
        output_map=None,
        bc_map=None,
        element_objs=None,
        output_objs=None,
    ):
        if integrand_map is None:
            integrand_map = {}
        if output_map is None:
            output_map = {}
        if bc_map is None:
            bc_map = {}
        if element_objs is None:
            element_objs = {}
        if output_objs is None:
            output_objs = {}
        self.mesh = mesh
        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.integrand_map = integrand_map
        self.integrand_formulation = integrand_formulation
        self.bc_map = bc_map
        self.output_map = output_map

        # Set the element objects and output objects
        self.element_objs = element_objs
        self.output_objs = output_objs

        # Allocate constraints for the weak formulation
        self.test_dof = None
        if self.integrand_formulation == "weak":
            self.test_dof = DegreesOfFreedom(
                self.mesh,
                self.soln_space,
                kind="multiplier",
                name="multiplier",
            )

        # Initialize Dofs
        self.soln_dof = DegreesOfFreedom(
            self.mesh,
            self.soln_space,
            kind="input",
            name="soln",
        )
        self.geo_dof = DegreesOfFreedom(
            self.mesh,
            self.geo_space,
            kind="data",
            name="geo",
        )
        self.data_dof = DegreesOfFreedom(
            self.mesh,
            self.data_space,
            kind="data",
            name="data",
        )

        # Build the boundary conditions
        self.boundary_conditions = []

        # Get the handler for the boundary conditions
        dof_handler = self.soln_dof.get_dof_handler()

        for name in bc_map:
            bc = bc_map[name]
            self.boundary_conditions.append(
                BoundaryConditions(name, dof_handler, bc, self.integrand_formulation)
            )

        return

    def _create_element_objs(self):
        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]
            integrand = self.integrand_map[integrand_name]["integrand"]
            integration_rule = self.integrand_map[integrand_name].get("rule", None)

            # Figure out the cell types that we need
            ctypes = []
            for target in targets:
                for ctype in domains[target]:
                    if not ctype in ctypes:
                        ctypes.append(ctype)

            # Loop over the element types for this integrand
            for ctype in ctypes:
                if (integrand_name, ctype) in self.element_objs:
                    continue

                # Set the element name
                elem_name = f"Element{integrand_name}{ctype.name}"

                # Get the basis objects for the element type
                test_basis = None
                if self.test_dof is not None:
                    test_basis = make_basis(self.soln_space, ctype, kind="multiplier")

                soln_basis = make_basis(self.soln_space, ctype, kind="input")
                geo_basis = make_basis(self.geo_space, ctype, kind="data")
                data_basis = make_basis(self.data_space, ctype, kind="data")

                # reduced integration option if rule is given
                if integration_rule == ["reduced"]:
                    # override get quadrature with custom quadrature
                    quadrature = ReducedQuadQuadrature()
                elif integration_rule is not None:
                    raise ValueError("Non-standard integration rule not supported")
                else:
                    quadrature = make_quadrature(self.soln_space, ctype)

                # Create the element object
                obj = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    integrand,
                    test_basis=test_basis,
                )

                # Set this into the element dictionary
                self.element_objs[(integrand_name, ctype)] = obj

        return

    def _create_output_objs(self):
        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        # Create the output objects
        for out_name in self.output_map:
            targets = self.output_map[out_name]["target"]
            output_names = self.output_map[out_name]["names"]
            output_func = self.output_map[out_name]["function"]

            # Figure out the element types we need
            ctypes = []
            for target in targets:
                for ctype in domains[target]:
                    if not ctype in ctypes:
                        ctypes.append(ctype)

            # Loop over the element types for generating the output function
            for ctype in ctypes:
                if (out_name, ctype) in self.element_objs:
                    continue

                elem_name = f"ElementOutput{out_name}{ctype.name}"

                # Get the basis objects for the element type
                soln_basis = make_basis(self.soln_space, ctype, kind="input")
                geo_basis = make_basis(self.geo_space, ctype, kind="data")
                data_basis = make_basis(self.data_space, ctype, kind="data")

                # Create the quadrature instance
                quadrature = make_quadrature(self.soln_space, ctype)

                # Create the output object
                obj = FiniteElementOutput(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    output_names,
                    output_func,
                )

                # Set this into the output dictionary
                self.output_objs[(out_name, ctype)] = obj

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        if self.test_dof is not None:
            self.test_dof.add_source(model)
        self.soln_dof.add_source(model)
        self.data_dof.add_source(model)
        self.geo_dof.add_source(model)

        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        # Figure out which elements need to be created
        self._create_element_objs()

        # Get the dof handler
        dof_handler = self.soln_dof.get_dof_handler()

        # Add the element component objects
        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]

            for target in targets:
                for ctype in domains[target]:
                    elem = self.element_objs[(integrand_name, ctype)]
                    comp_name = f"Element{integrand_name}{ctype.name}{target}"

                    # Add the element/component
                    nelems = self.mesh.get_num_elements(target, ctype)
                    model.add_component(comp_name, nelems, elem)

                    # Link all the element dof to the component
                    self.soln_dof.link_dof(model, target, ctype, comp_name)
                    self.data_dof.link_dof(model, target, ctype, comp_name)
                    self.geo_dof.link_dof(model, target, ctype, comp_name)

                    # Link the constraints (if using the weak formulation)
                    if self.test_dof is not None:
                        self.test_dof.link_dof(model, target, ctype, comp_name)

                    # Set the element signs if relevant
                    for space in self.soln_space.get_spaces():
                        if space.func_space == Space.HDIV:
                            signs = dof_handler.get_dof_signs(space, target, ctype)
                            model.set_data(f"{comp_name}.hdiv_signs", signs)
                        elif space.func_space == Space.HCURL:
                            signs = dof_handler.get_dof_signs(space, target, ctype)
                            model.set_data(f"{comp_name}.hcurl_signs", signs)

        # Add BC components and links
        for bc in self.boundary_conditions:
            bc.add_bcs(model)

        # Make a list of all of the outputs
        all_outputs = []
        for out_name in self.output_map:
            for name in self.output_map[out_name]["names"]:
                if not (name in all_outputs):
                    all_outputs.append(name)

        # Add the outputs component
        model.add_component("outputs", 1, DofSource(output_names=all_outputs))

        self._create_output_objs()

        for out_name in self.output_map:
            targets = self.output_map[out_name]["target"]
            output_names = self.output_map[out_name]["names"]

            for target in targets:
                for etype in domains[target]:
                    obj = self.output_objs[(out_name, etype)]
                    comp_name = f"ElementOutput{out_name}{etype}{target}"

                    # Add the element/component
                    nelems = self.mesh.get_num_elements(target, etype)
                    model.add_component(comp_name, nelems, obj)

                    # Link all the element dof to the component
                    self.soln_dof.link_dof(model, target, etype, comp_name)
                    self.data_dof.link_dof(model, target, etype, comp_name)
                    self.geo_dof.link_dof(model, target, etype, comp_name)

                    # Link the outputs
                    for name in output_names:
                        model.link(f"{comp_name}.{name}", f"outputs.{name}[0]")

        # Set the node locations directly
        spatial_names = ["x", "y", "z"][: self.mesh.X.shape[1]]
        geo_spaces = self.geo_space.get_spaces()
        coords = self.geo_dof.get_node_coordinates(geo_spaces[0], self.mesh.X)
        for k, name in enumerate(self.geo_space.get_names(geo_spaces[0])):
            if name in spatial_names:
                model.set_data(f"geo.{name}", coords[:, k])

        # Link the output to the finite element class
        return model

    def field_to_nodes(self, field, kind="soln"):
        """
        Reorder a solution field from internal DOF ordering to mesh-node
        ordering.

        The DOF handler numbers degrees of freedom in order of first
        encounter while traversing the mesh, which is a permutation of the
        mesh node ordering (not the identity). Solution vectors returned by
        the model are therefore indexed by DOF number, whereas plotting and
        post-processing routines index by mesh node. This method scatters a
        DOF-ordered field back into node ordering so that ``result[node]``
        holds the value at that mesh vertex.

        Only valid for degree-1 H1 fields, where DOFs correspond one-to-one
        with mesh vertices.

        Parameters
        ----------
        field : np.ndarray
            Values indexed by global DOF number.
        kind : str
            Which DOF set to use: "soln" (default), "geo", or "data".

        Returns
        -------
        np.ndarray
            Values reordered so they are indexed by mesh node number.
        """
        dof = {
            "soln": self.soln_dof,
            "geo": self.geo_dof,
            "data": self.data_dof,
        }.get(kind)
        if dof is None:
            raise ValueError(f"Unknown field kind '{kind}'")

        space = dof.solution_space.get_spaces()[0]
        dof_to_node = dof.get_vertex_dof_to_node(space)

        field = np.asarray(field)
        result = np.empty_like(field)
        result[dof_to_node] = field
        return result
        # return field[dof_to_node >= 0]

    def visualize(self, x, field, domain):
        """
        Visualize an H1 triangle solution field of degree 1 or 2.
        """
        grid = self._build_visualization_grid(x, field=field, domain=domain)
        grid.plot(scalars=field, show_edges=False, cmap="coolwarm")
        return grid

    def save_vtu(self, x, field, domain):
        grid = self._build_visualization_grid(x, field=field, domain=domain)
        grid.save("output_field.vtu")
        return

    def _build_visualization_grid(self, x, field, domain):
        """
        Build a PyVista grid for an H1 triangle solution field of degree 1 or 2
        on a mesh whose geometry is only P1.
        """
        ctype = CellType.TRIANGLE

        soln_space = self.soln_dof.solution_space.get_spaces()[0]
        dof_handler = self.soln_dof.get_dof_handler()
        layout = DofLayout.make_layout(soln_space, ctype)
        ndof_local = len(layout.pts)

        # Map the number of local DOFs to the matching PyVista cell type.
        pv_cell_type = {
            3: pv.CellType.TRIANGLE,  # degree 1
            6: pv.CellType.QUADRATIC_TRIANGLE,  # degree 2
        }.get(ndof_local)

        if pv_cell_type is None:
            raise NotImplementedError(
                "build_visualization_grid supports degree-1 and degree-2 H1 "
                f"triangles (got degree={soln_space.degree}, "
                f"ndof/elem={ndof_local})."
            )

        # Global DOF connectivity for the solution space: (nelem, ndof_local),
        soln_conn = dof_handler.get_dof_conn(soln_space, domain, ctype)

        # P1 vertex connectivity and coordinates for the geometry.
        vertex_conn = self.mesh.get_vertex_conn(domain, ctype)  # (nelem, 3)
        Xv = np.asarray(self.mesh.X)  # (nnodes, dim)
        dim = Xv.shape[1]

        # Reference elment node layout
        param = np.asarray(layout.pts)  # (ndof_local, 2), (xi, eta)
        xi, eta = param[:, 0], param[:, 1]
        N = np.stack([1.0 - xi - eta, xi, eta], axis=1)  # (ndof_local, 3)

        ndof_global = dof_handler.get_num_dof(soln_space)
        points = np.zeros((ndof_global, 3))
        for e in range(soln_conn.shape[0]):
            vcoords = Xv[vertex_conn[e]]  # (3, dim)
            phys = N @ vcoords  # (ndof_local, dim)
            for a in range(ndof_local):
                points[soln_conn[e, a], :dim] = phys[a]

        grid = pv.UnstructuredGrid({pv_cell_type: soln_conn}, points)
        grid.point_data[field] = np.asarray(x[field])
        return grid
