"""Starting point strategy.

Restores feasibility from the initial point and then centers the start
at the requested initial mu with the shared centering rule.
"""


class FeasibilityPresolve:
    def __init__(self, options):
        self.enabled = options["feasibility_presolve"] and not options["warm_start"]
        self.initial_mu = options["initial_barrier_param"]
        self.tau_min = options["tau_min"]

    def run(
        self,
        feasible_resto,
        multiplier_init,
        centerer,
        solver,
        evaluator,
        state,
        line_search,
    ):
        """Restore feasibility and center the start, returning True on success."""
        if not self.enabled:
            return False

        resto_info = feasible_resto.restore(solver, evaluator, state, line_search)
        if not resto_info.success:
            return False

        # The schedule starts at the requested initial mu
        state.mu = self.initial_mu
        state.tau = max(self.tau_min, 1.0 - state.mu)

        # Center the restored point at the schedule mu
        centerer.center(evaluator, state)

        # Least squares multipliers subject to the magnitude cap
        multiplier_init.compute_least_squares_multipliers(evaluator, solver, state)
        return True
