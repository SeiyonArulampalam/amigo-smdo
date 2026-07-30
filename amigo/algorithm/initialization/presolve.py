"""Starting point strategy.

Restores feasibility from the initial point, centers the bound duals there,
then estimates the equality multipliers by least squares. The least-squares
estimate is only well-posed near the feasible manifold, hence the ordering.
"""


class FeasibilityPresolve:
    def __init__(self, options):
        self.enabled = options["feasibility_presolve"] and not options["warm_start"]
        self.bound_push = options["warm_start_bound_push"]
        self.mult_cap = options["warm_start_mult_init_max"]

    def run(
        self,
        feasible_resto,
        multiplier_init,
        optimizer,
        solver,
        evaluator,
        state,
        line_search,
    ):
        """Feasibility, then centered bound duals, then LS equality duals.

        Returns True when the presolve ran and succeeded."""
        if not self.enabled:
            return False
        resto_info = feasible_resto.restore(solver, evaluator, state, line_search)
        if not resto_info.success:
            return False
        optimizer.initialize_duals_warm(
            state.mu, self.bound_push, self.mult_cap, state.current
        )
        state.invalidate()
        multiplier_init.compute_least_squares_multipliers(evaluator, solver, state)
        return True
