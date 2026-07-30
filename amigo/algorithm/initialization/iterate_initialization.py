"""Build the primal-dual starting point before the first iteration.

Relaxes variable bounds, projects the initial design vector into the
relaxed box, initializes slacks and bound multipliers, applies
gradient-based NLP scaling, and initializes constraint multipliers
(least-squares by default, affine step when requested).
"""


class SlackInitializer:
    def __init__(self, options, model, problem, optimizer):
        self.model = model
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize_slacks(self, evaluator, state):
        x = state.get_current_point()

        if self.options["warm_start"]:
            # Keep the x_init multipliers and slacks, set mu-consistent duals
            self.optimizer.initialize_duals_warm(
                state.mu,
                self.options["warm_start_bound_push"],
                self.options["warm_start_mult_init_max"],
                state.current,
            )
            state.invalidate()
            return

        # Zero the multipliers and place slacks and bound duals in the interior
        con_indices = self.problem.get_constraint_indices()
        x.fill_at(con_indices, 0.0)
        self.optimizer.initialize_duals(state.mu, state.current)
        state.invalidate()

        if self.model is not None:
            # The gradient constraint rows hold the residuals at zero multipliers
            evaluator.evaluate_gradient(state)
            slack_indices = self.model.slack_indices
            ineq_indices = self.model.ineq_constraint_indices

            x.copy_device_to_host()
            state.gradient.copy_device_to_host()
            x_array = x.get_array()
            grad_array = state.gradient.get_array()
            # Absorb each inequality residual into its slack for feasibility
            x_array[slack_indices] += grad_array[ineq_indices]
            x.copy_host_to_device()

            # Project the moved slacks back into the interior
            self.optimizer.initialize_duals(state.mu, state.current)
            state.invalidate()

        return
