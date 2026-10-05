import amigo as am
import numpy as np
from .dof_handler import DofHandler


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


class CustomOrderedBCs:
    def __init__(self, dir=1, start=True, end=True):
        self.dir = dir
        self.start = start
        self.end = end

    def __call__(self, name, targets, dof_handler):
        mesh = dof_handler.mesh
        space = dof_handler.solution_space.get_space(name)

        mapping = dof_handler.get_node_to_dof_map(space)
        inverse = np.empty_like(mapping)
        inverse[mapping] = np.arange(len(mapping))

        # Get the targets
        a, b = targets
        id_a = mesh.get_block_ids(a)
        id_b = mesh.get_block_ids(b)

        coord_left, nodes_left = self._get_ordered_dof(
            id_a, mesh, inverse, self.start, self.end
        )
        coord_right, nodes_right = self._get_ordered_dof(
            id_b, mesh, inverse, self.start, self.end
        )

        if len(nodes_left) != len(nodes_right):
            raise Exception(f"nnodes left != nnodes right")

        # Reorder the nodes to match
        return self._reorder_nodes(coord_left, nodes_left, coord_right, nodes_right)

    def _get_ordered_dof(
        self, ids, mesh, imapping, start: bool = True, end: bool = True
    ):
        # Get the node numbers from the mesh
        all_nodes = []
        for id in ids:
            all_nodes.extend(mesh.blocks[id].connectivity.flatten())
        unique_nodes = list(dict.fromkeys(all_nodes))

        if not start or not end:
            if not start:
                unique_nodes = unique_nodes[1:]
            if not end:
                unique_nodes = unique_nodes[:-1]

        y = mesh.X[unique_nodes, self.dir]

        return y, imapping[unique_nodes]

    def _reorder_nodes(self, coord_left, nodes_left, coord_right, nodes_right):
        nodes_left = np.array(nodes_left)
        nodes_right = np.array(nodes_right)

        idx_left = np.argsort(coord_left)
        idx_right = np.argsort(coord_right)

        return nodes_left[idx_left], nodes_right[idx_right]


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

    def _get_target_dof(self, name: str, targets: list[str]):
        all_dof = []
        for target in targets:
            dof = self.dof_handler.get_dof_in_domain(name, target)
            all_dof.extend(dof)

        return np.unique(all_dof)

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
            input_names = self.bc["input"]
            target = self.bc["target"]
            callback = self.bc["callback"]

            for name in input_names:
                left_dofs, right_dofs = callback(name, target, self.dof_handler)

                if self.bc["type"] == "continuity":
                    model.link(
                        f"soln.{name}",
                        f"soln.{name}",
                        src_indices=left_dofs,
                        tgt_indices=right_dofs,
                    )

                elif self.bc["type"] == "scaled":
                    scale = self.bc["scale"]
                    class_name = f"ScaledBC_{self.bc_name}"
                    bc_src = ScaledBC(class_name, input_names, scale=scale)

                    if len(left_dofs) > 0:
                        model.add_component(f"{self.bc_name}", len(left_dofs), bc_src)

                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_left",
                            src_indices=left_dofs,
                        )
                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_right",
                            src_indices=right_dofs,
                        )
