import amigo as am
import numpy as np
import argparse
from typing import List
from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from amigo.fem import Mesh, Problem, CellType, SolutionSpace
from amigo.fem.elements import MITCShell


class GapConstraint(am.Component):
    """Floor contact in x: (x + u) - x_floor <= 0"""

    def __init__(self):
        super().__init__()
        self.add_data("x")
        self.add_data("x_floor", value=1.25)
        self.add_input("u")
        self.add_constraint("gap", lower=-am.inf, upper=0.0)

    def compute(self):
        x = self.data["x"]
        u = self.inputs["u"]
        x_floor = self.data["x_floor"]
        self.constraints["gap"] = (x + u) - x_floor


class IsoShell:
    def __init__(self, E=70e9, nu=0.3, ks=5.0 / 6.0, penalty=1e-4):
        self.E = E
        self.nu = nu
        self.ks = ks
        self.penalty = penalty

        return

    def constitutive(self, soln, geo=None, data=None):
        t = data["thickness"].value
        G = 0.5 * self.E / (1.0 + self.nu)
        D = self.E / (1.0 - self.nu**2)

        A11 = D * t
        A12 = self.nu * A11
        A22 = A11
        A66 = 0.5 * A11 * (1 - self.nu)

        D11 = D * t**3 / 12.0
        D12 = self.nu * D11
        D22 = D11
        D66 = 0.5 * D11 * (1 - self.nu)

        As44 = self.ks * G * t
        As55 = As44

        # Drill penalty
        kdrill = t * self.E * self.penalty

        return (A11, A12, A22, A66), (D11, D12, D22, D66), (As44, As55), kdrill


def traction(soln, geo=None, data=None):
    tx = 1e3
    return -tx * soln["u"].value


class FlumeCylinder(Analysis):
    def __init__(
        self,
        obj_name: str,
        sub_analyses: List,
        args,
        **kwargs,
    ):
        # Command line arguments
        self.args = args

        # Default parameters dictionary required by Analysis base class
        self.default_parameters = {"E": 70e4, "nu": 0.3}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Load the mesh
        self.mesh = Mesh("cylinder.inp")

        # Create the amigo model, build and initialize it
        self.initialize()

        # Initialize the flume variable
        num_thickness = np.ravel(self.model.get_indices("data.thickness")).size
        self.variables = {
            "thickness": State(
                value=0.01 * np.ones(num_thickness),
                desc="Shell thickness field",
                source=self,
            )
        }

        return

    def _analyze(self):
        # Retrieve the variables
        thickness = self.variables["thickness"].value

        # Set the model data
        data = self.model.get_data_vector()
        data["data.thickness"] = thickness

        # Reset the vector
        self.x[:] = 0.0

        # Initialize the optimizer for the post-optimality sensitivty analysis
        opt_options = {
            "max_iterations": 200,
            "convergence_tolerance": 1e-8,
            "max_line_search_iterations": 1,
            "initial_barrier_param": 0.1,
        }
        self.opt = am.Optimizer(self.model, self.x)
        self.opt.optimize(opt_options)

        # Compute the outputs from the optimized state (self.x is the converged
        # solution that compute_post_opt_derivatives differentiates)
        self.model.compute_output(self.x, self.output_vec)

        # Extract the outputs (scalar values)
        volume = float(self.output_vec["outputs.volume"][0])
        energy = float(self.output_vec["outputs.strain_energy"][0])

        # Declare the outputs
        self.outputs = {
            "volume": State(value=volume, desc="Cylinder volume", source=self),
            "strain_energy": State(value=energy, desc="Strain energy", source=self),
        }

        return

    def _analyze_adjoint(self):
        # Compute the post-optimality derivatives
        dfdx, of_map, wrt_map = self.opt.compute_post_opt_derivatives(
            of=["outputs.volume", "outputs.strain_energy"], wrt=["data.thickness"]
        )
        dvol_dt = dfdx[of_map["outputs.volume"], wrt_map["data.thickness"]]
        denergy_dt = dfdx[of_map["outputs.strain_energy"], wrt_map["data.thickness"]]

        # Scalar output seeds
        volume_b = self.outputs["volume"].deriv
        strain_energy_b = self.outputs["strain_energy"].deriv

        # Field derivative accumulator (length-N vector)
        thickness_b = self.variables["thickness"].deriv

        # Accumulate the seeded contributions
        thickness_b = thickness_b + volume_b * dvol_dt
        thickness_b = thickness_b + strain_energy_b * denergy_dt

        # Set the derivative value
        self.variables["thickness"].set_deriv_value(thickness_b)
        return

    def initialize(self):
        self._create_model()
        if self.args.build:
            self._build_model()
        self._initialize_model()
        return

    def _create_model(self):
        # Declare the domains of interest
        surfaces = ["SURFACE1"]
        bottom_line = "LINE3"
        lateral_line = "LINE2"
        top_line = "LINE1"

        # Get parameters
        E = self.parameters["E"]
        nu = self.parameters["nu"]

        iso = IsoShell(E=E, nu=nu)
        ctype = CellType.QUADRILATERAL
        mitc_shell = MITCShell(constitutive=iso.constitutive)
        soln_space, geo_space, data_space, output_space = (
            mitc_shell.get_default_spaces()
        )

        # Override the default data space
        data_space = SolutionSpace({"data": {"thickness": "H1"}})

        integrand_map = {
            "shell": {"target": surfaces},
            "traction": {"target": surfaces},
        }

        bc_map = {
            "pinned_bottom": {
                "type": "dirichlet",
                "input": ["u", "v", "w", "rz"],
                "target": [bottom_line],
            },
            "pinned_top": {
                "type": "dirichlet",
                "input": ["u", "v", "w", "rz"],
                "target": [top_line],
            },
        }

        # Create the element objects
        element_objs = {
            ("shell", ctype): mitc_shell.create_element(
                name="Shell", data_space=data_space, ctype=ctype
            ),
            ("traction", ctype): mitc_shell.create_element(
                name="Traction", data_space=data_space, ctype=ctype, integrand=traction
            ),
        }

        self.problem = Problem(
            self.mesh,
            soln_space,
            geo_space,
            data_space,
            integrand_map=integrand_map,
            bc_map=bc_map,
            element_objs=element_objs,
        )

        self.model = self.problem.create_model("cylinder_shell")

        # Add the shell normals to the output
        MITCShell.add_shell_normals(self.problem, self.model, surfaces)

        # Find the nodes that are strictly in the interior of the mesh
        handler = self.problem.soln_dof.get_dof_handler()
        ndofs = handler.get_num_dof("u")
        clamped_dofs = handler.get_dof_in_domain("u", [bottom_line, top_line])
        dofs = np.setdiff1d(np.arange(ndofs), clamped_dofs)

        # Add the contact constraint
        self.model.add_component("contact", len(dofs), GapConstraint())
        self.model.link("geo.x", "contact.x", src_indices=dofs)
        self.model.link("soln.u", "contact.u", src_indices=dofs)
        self.model.link("contact.x_floor", "contact.x_floor[0]")

        self.model.set_meta("lower", "data.thickness", 0.0, vtype="data")
        self.model.set_meta("upper", "data.thickness", am.inf, vtype="data")
        self.model.set_meta("value", "data.thickness", 0.01, vtype="data")

        output_names = ["strain_energy", "volume"]

        def output_function(soln, data=None, geo=None):
            outputs = {}
            outputs["volume"] = data["thickness"].value
            outputs["strain_energy"] = mitc_shell.potential(soln, data=data, geo=geo)
            return outputs

        output_obj = mitc_shell.create_element_output(
            name="Outputs",
            data_space=data_space,
            ctype=ctype,
            output_names=output_names,
            output_function=output_function,
        )
        output_map = {"shell": {"names": output_names, "target": surfaces}}
        output_objs = {("shell", ctype): output_obj}

        self.problem.add_integrated_output(
            self.model, output_map=output_map, output_objs=output_objs
        )
        return

    def _build_model(self):
        self.model.build_module()
        return

    def _initialize_model(self):
        if not self.model.is_initialize():
            # initialize the amigo model
            self.model.initialize()

            # set the shell normals
            MITCShell.set_shell_normals(self.model)

            # Create the system
            self.x = self.model.create_vector()
            self.g = self.model.create_vector()
            self.output_vec = self.model.create_vector()
            self.mat = self.model.create_matrix()

            # Symbolic reordering
            self.ldl = am.SparseLDL(self.mat, am.SolverType.LDL)
        return


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", default=False)
    args = parser.parse_args()

    # FlumeCylinder object
    fc = FlumeCylinder(obj_name="flume_cylinder", sub_analyses=[], args=args)

    # Declare the defined variables (thickness is a field)
    num_thickness = fc.variables["thickness"].shape[0]
    defined_vars = {"thickness": 1.0 * np.ones(num_thickness)}
    fc.declare_design_vars(variables=["thickness"])
    fc.test_isolated_adjoint(method="fd", defined_vars=defined_vars, dh_fd=1e-6)
