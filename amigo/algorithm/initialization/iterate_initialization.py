"""Center the primal-dual iterate for the current barrier parameter.

Cold starts, warm starts, the presolve, and restoration exits share one
rule. Slacks absorb the inequality residuals, primals are projected
strictly interior, and the bound duals are set to the flat z = mu.
"""


class IterateCenterer:
    def __init__(self, options, model, problem, optimizer):
        self.model = model
        self.options = options
        self.problem = problem
        self.optimizer = optimizer
        self.temp_con = problem.create_constraint_vector()

    def center(self, evaluator, state, keep_multipliers=False):
        """Center the iterate at the current mu, keeping the multipliers when asked."""
        x = state.get_current_point()
        con_indices = self.problem.get_constraint_indices()

        # Save the multipliers when a warm start supplies them
        if keep_multipliers:
            x.get_values_at(con_indices, self.temp_con)

        # Zero the multipliers so the gradient holds the residuals below
        x.fill_at(con_indices, 0.0)
        state.invalidate()

        # Project the primals strictly interior and set the flat duals z = mu
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

            # Project the moved slacks back interior with the same flat rule
            self.optimizer.initialize_duals(state.mu, state.current)
            state.invalidate()

        # Restore the saved multipliers
        if keep_multipliers:
            x.set_values_at(con_indices, self.temp_con)
            state.invalidate()

        return
