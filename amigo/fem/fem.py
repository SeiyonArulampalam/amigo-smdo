import amigo as am
import numpy as np
from . import basis
from .element import FiniteElement, FiniteElementOutput
from .fem_space import SolutionSpace
from .basis import make_basis
from .mesh import Mesh
from .dof_handler import DegreesOfFreedom, DofSource
from .quadrature import make_quadrature, ReducedQuadQuadrature
from .plot_utils import plot
from pathlib import Path


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
    def __init__(self, bc_name, mesh, bc={}, integrand_formulation="potential"):
        if not (
            bc["type"] == "dirichlet"
            or bc["type"] == "continuity"
            or bc["type"] == "scaled"
        ):
            typ = bc["type"]
            raise ValueError(f"Unrecognized boundary condition type {typ}")
        self.bc_name = bc_name
        self.mesh = mesh
        self.bc = bc
        self.integrand_formulation = integrand_formulation
        return

    def _get_bc_nodes(self, targets, start=True, end=True):
        all_nodes = []
        for target in targets:
            nodes = self.mesh.get_nodes_in_domain(target)
            all_nodes.extend(nodes)

        unique = list(dict.fromkeys(all_nodes))

        if not start:
            unique = unique[1:]
        if not end:
            unique = unique[:-1]

        return unique

    def _reorder_nodes(self, nodes_left, nodes_right):
        nodes_left = np.array(nodes_left)
        nodes_right = np.array(nodes_right)

        y_left = self.mesh.X[nodes_left, 1]
        y_right = self.mesh.X[nodes_right, 1]

        idx_left = np.argsort(y_left)
        idx_right = np.argsort(y_right)

        return nodes_left[idx_left], nodes_right[idx_right]

    def _get_matched_nodes(self, targets, start=True, end=True):
        left_target_lines = targets[0]
        right_target_lines = targets[1]
        nodes_left = self._get_bc_nodes(left_target_lines, start, end)
        nodes_right = self._get_bc_nodes(right_target_lines, start, end)

        if len(nodes_left) != len(nodes_right):
            raise Exception(f"nnodes left != nnodes right")

        # Reorder the nodes to match
        return self._reorder_nodes(nodes_left, nodes_right)

    def add_bcs(self, model):
        """Add the boundary conditions to the model"""

        if self.bc["type"] == "dirichlet":
            nodes = self._get_bc_nodes(self.bc["target"])

            input_names = self.bc["input"]
            for name in input_names:
                model.add_fixed(f"soln.{name}", nodes)

                if self.integrand_formulation == "weak":
                    model.add_fixed(f"multiplier.res_{name}", nodes)

        else:
            targets = self.bc["target"]
            start = self.bc.get("start", True)
            end = self.bc.get("end", True)

            nodes_left, nodes_right = self._get_matched_nodes(
                targets, start=start, end=end
            )

            input_names = self.bc["input"]
            if self.bc["type"] == "continuity":
                for name in input_names:
                    model.link(
                        f"soln.{name}",
                        f"soln.{name}",
                        src_indices=nodes_left,
                        tgt_indices=nodes_right,
                    )

            elif self.bc["type"] == "scaled":
                scale = self.bc["scale"]
                class_name = f"ScaledBC_{self.bc_name}"
                bc_src = ScaledBC(class_name, input_names, scale=scale)

                if len(nodes_left) > 0:
                    model.add_component(
                        f"{self.bc_name}",
                        len(nodes_left),
                        bc_src,
                    )

                    for name in input_names:
                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_left",
                            src_indices=nodes_left,
                        )
                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_right",
                            src_indices=nodes_right,
                        )


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

        for name in bc_map:
            bc = bc_map[name]
            self.boundary_conditions.append(
                BoundaryConditions(name, self.mesh, bc, self.integrand_formulation)
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
        for k, name in enumerate(self.geo_space.get_names("H1")):
            if name in spatial_names:
                model.set_data(f"geo.{name}", self.mesh.X[:, k])

        # Link the output to the finite element class
        return model
