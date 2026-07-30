"""Per-iteration state carried through the optimization loop.

InteriorPointState holds the current point, its derivatives, the
barrier parameter, step lengths, and the error measures, along with
the currency flags that mark which of them are up to date.
"""

import amigo as am


class InteriorPointState:
    # Communicator rank
    comm_rank: int

    # Iteration counter
    iter: int

    # Barrier parameter
    mu: float

    # Fraction to the boundary parameter
    tau: float

    # Objective scaling factor
    obj_scale: float

    # Objective function value (scaled) and log-barrier term
    objective_value: float
    log_barrier_value: float
    con_infeasibility: float
    objective_current: bool

    # Current primal-dual vector
    current: am.OptVector

    # Gradient information
    gradient: am.Vector
    gradient_current: bool

    # Second-order information
    diagonal: am.Vector
    hessian: am.CSRMat
    hessian_current: bool

    # Max primal and max dual step lengths
    max_alpha_primal: float
    max_alpha_dual: float
    step: am.OptVector
    step_current: bool

    # Unscaled primal step norm, for the divergence watchdog
    raw_step_norm: float

    # Residual and step information
    residual_norm: float
    residual: am.Vector

    # Current error measures
    primal_infeas: float
    dual_infeas: float
    complementarity: float
    kkt_error: float
    residual_current: bool

    def __init__(self, x, options, problem, optimizer):
        self.comm_rank = 0
        self.iter = 0
        self.mu = options["initial_barrier_param"]
        self.tau = options["fraction_to_boundary"]
        self.obj_scale = 1.0
        self.objective_value = 0.0
        self.log_barrier_value = 0.0
        self.con_infeasibility = 0.0
        self.objective_current = False

        self.current = optimizer.create_opt_vector(x)
        self.gradient = problem.create_vector()
        self.gradient_current = False

        self.residual = problem.create_vector()
        self.diagonal = problem.create_vector()
        self.diagonal_current = False
        self.hessian = problem.create_matrix()
        self.hessian_current = False

        self.max_alpha_primal = 1.0
        self.max_alpha_dual = 1.0
        self.step = optimizer.create_opt_vector()
        self.step_current = False
        self.raw_step_norm = 0.0

        self.residual_norm = 0.0
        self.residual = problem.create_vector()
        self.primal_infeas = 0.0
        self.dual_infeas = 0.0
        self.complementarity = 0.0
        self.kkt_error = 0.0
        self.residual_current = False

    @property
    def barrier_objective(self):
        """Scaled objective plus the log-barrier term"""
        return self.objective_value + self.log_barrier_value

    def get_current_point(self):
        """Get the current primal-dual vector"""
        return self.current.get_solution()

    def get_trial_point(self):
        """Get the trial primal-dual vector"""
        return self.trial.get_solution()

    def invalidate(
        self, obj=True, grad=True, hess=True, res=True, step=True, diag=None
    ):
        # Diagonal depends on the point like the Hessian (not on mu)
        if diag is None:
            diag = hess
        if obj:
            self.objective_current = False
        if grad:
            self.gradient_current = False
        if hess:
            self.hessian_current = False
        if diag:
            self.diagonal_current = False
        if res:
            self.residual_current = False
        if step:
            self.step_current = False
