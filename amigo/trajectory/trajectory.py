import numpy as np
import amigo as am
from abc import abstractmethod


class TrajectorySource(am.Component):
    def __init__(self, state_size: int, inputs: list[dict] = []):
        super().__init__()

        self.add_input("q", shape=state_size)

        for input in inputs:
            name = input["name"]
            self.add_input(
                f"{name}", shape=input.get("shape"), label=input.get("label")
            )

        return


class TrajectoryComponent(am.Component):
    """
    Component wrapper designed for trajectory computations using the trapezoid rule.

    The user defines the `dynamics` method to implement the dynamics governing equations.
    The `compute` method automatically calls the dynamics based on any specified inputs
    (the state `q` is always assumed).

    Note:
    Can be paired with TrajectorySource either directly by the user or automatically if used
    to create a TrajectoryModel.

    Example Usage:
    ---------
    For a trajectory calculated at 20 points in time and a state vector of dimension 4,
    we also include two extra inputs: angle of attack and throttle. A scaling dictionary
    is also provided in this example.
    ```
    class ExampleDynamics(am.TrajectoryComponent):
        def __init__(self, scaling):
            super().__init__(20, 4, [{"name": "alpha"}, {"name": "throttle"}])

            self.scaling = scaling
            self.add_constant("g", value=9.81)
            return

        def dynamics(self, q, alpha, throttle):
            g = self.constants["g"]
            mass = q[0] * self.scaling["mass"]
            ...
            return qdot
    ```
    """

    def __init__(
        self, num_time_steps: int, state_size: int, aux_inputs: list[dict] = []
    ):
        """
        Automatically adds inputs for the state variables and final time, as well as
        any additional inputs specified in `aux_inputs`.

        The user can specify constants after calling `super().__init__()`

        Args:
            num_time_steps (int) : Number of evaluation points in the trajectory
            state_size (int) : Size of the state vector at a given point
            aux_inputs : list of auxiliary inputs besides the state vector `q`

        Inputs must be specified as a list of dictionaries, with a dictionary for every
        auxiliary input that contains at least the "name" entry.
        ```
        """
        super().__init__()

        self._input_names = [input["name"] for input in aux_inputs]
        self.num_time_steps = num_time_steps
        self._state_size = state_size
        self._aux_inputs = aux_inputs

        self.add_input("tf", label="final time")
        self.add_input("q1", shape=state_size)
        self.add_input("q2", shape=state_size)

        for input in aux_inputs:
            name = input["name"]
            self.add_input(
                f"{name}1", shape=input.get("shape"), label=input.get("label")
            )
            self.add_input(
                f"{name}2", shape=input.get("shape"), label=input.get("label")
            )

        self.add_constraint("res", shape=state_size)

        return

    @abstractmethod
    def dynamics(self, q, *args):
        """
        User must implement the dynamics and return qdot.

        Args:
            q : State vector

        Optional additional arguments must be specified in the same order as defined in `aux_inputs`.
        """
        pass

    def compute(self):
        """
        The user-defined dynamics are used to compute constraints.
        """
        dt = self.inputs["tf"] / self.num_time_steps
        input1_vars = [self.inputs[f"{name}1"] for name in self._input_names]
        input2_vars = [self.inputs[f"{name}2"] for name in self._input_names]

        q1, q2 = self.inputs["q1"], self.inputs["q2"]

        f1 = self.dynamics(q1, *input1_vars)
        f2 = self.dynamics(q2, *input2_vars)
        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * dt * (f1[i] + f2[i]) for i in range(self._state_size)
        ]

        return


class TrajectoryModel:
    """
    Model helper to streamline continuous-time dynamics and trajectory computation.

    Contains a `TrajectorySource` and `TrajectoryComponent` with automatic linking.

    The `TrajectorySource` is labeled as `source` in this model. The `TrajectoryComponent`
    is labeled as `kernel`.

    Example Usage
    -------
    Consider a problem to minimize the final time.
    Assume `dynamics` has been previously defined and has an auxiliary input `alpha`.
    For brevity: setting lower/upper bounds, boundary conditions, etc. are not included.
    ```
    traj = am.TrajectoryModel(dynamics)

    model = am.Model("example")
    model.add_model("traj", traj.create_model())

    # Link to a BSpline that controls alpha
    model.link("traj.source.alpha", "bspline.interp_values.alpha")

    # Link final time from objective
    model.link("obj.tf[0]", f"traj.kernel.tf[:]")
    ```
    """

    def __init__(
        self,
        dynamics: TrajectoryComponent,
    ):
        """
        Initialize a TrajectoryModel helper.

        Args:
            dynamics (TrajectoryComponent) : implements the governing dynamical equations

        Note:
        `TrajectorySource` is automatically created based on the input parameters to the
        trajectory component.
        """
        self._dynamics = dynamics
        self.num_time_steps = dynamics.num_time_steps
        self.state_size = dynamics._state_size
        self._input_list = dynamics._aux_inputs
        self._input_names = [input["name"] for input in self._input_list]

        return

    def create_model(
        self,
        module_name: str | None = None,
    ):
        """
        Create and link the Amigo model

        Args:
            module_name (str) : specify name for the module (optional)
        """
        model = am.Model(module_name)

        model.add_component(
            "source",
            self.num_time_steps + 1,
            TrajectorySource(self.state_size, self._input_list),
        )
        model.add_component("kernel", self.num_time_steps, self._dynamics)

        model.link(f"source.q[:-1,:]", f"kernel.q1")
        model.link(f"source.q[1:,:]", f"kernel.q2")

        for name in self._input_names:
            model.link(f"source.{name}[:-1]", f"kernel.{name}1")
            model.link(f"source.{name}[1:]", f"kernel.{name}2")

        return model

    def link_boundary_conditions(
        self, model: am.Model, traj_model_name: str, ic: str = None, fc: str = None
    ):
        """
        Helper utility to set boundary conditions.

        Args:
            model (am.Model) : parent model that `TrajectoryModel` belongs to
            traj_name (str) : name of the `TrajectoryModel` in `model`
            ic (str) : name of the initial condition
            fc (str) : name of the final condition

        Effectively replaces the following links:
        ```
        model.link("traj.source.q[0,:]", "ic.q[0,:]")
        model.link("traj.source.q[num_time_steps,:]", "fc.q[0,:]")
        ```
        """
        if ic:
            model.link(f"{traj_model_name}.source.q[0,:]", f"{ic}.q[0,:]")
        if fc:
            model.link(
                f"{traj_model_name}.source.q[{self.num_time_steps}, :]", f"{fc}.q[0,:]"
            )
