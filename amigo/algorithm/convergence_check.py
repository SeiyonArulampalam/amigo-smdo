"""Convergence tests for the interior-point loop.

Primary convergence requires every KKT component below its tolerance.
Divergence halts the solve when the iterate magnitude or the Newton
step exceeds a safety bound. Acceptable convergence flags a weaker
solution once relaxed tolerances hold for several iterations in a row.
Precision-floor detection catches bit-identical residuals, the limit
below which further progress is not possible. A non-finite error or a
stall at a non-acceptable point asks for feasibility restoration.
"""

import numpy as np

# Return codes for convergence check
CONTINUE = 0
CONVERGED = 1
CONVERGED_ACCEPTABLE = 2
DIVERGED = 3
PRECISION_FLOOR = 4
ITERATING = 5
RESTORATION_NEEDED = 6
LOCALLY_INFEASIBLE = 7


class ConvergenceCheck:
    """Convergence checks: primary, acceptable, divergence, precision floor."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # The previous residual norm - initialize to zero
        self.prev_res_norm = 0.0

        # Set the precision floor count
        self.precision_floor_count = 0

        # Set the acceptable counter
        self.acceptable_counter = 0

        # Count consecutive iterations with a divergent Newton step
        self.diverging_step_count = 0

        # Count consecutive accepted steps that stalled at a tiny alpha
        self.small_alpha_count = 0

    def test_small_alpha_stall(self, state):
        """Report whether accepted steps keep stalling while infeasible.

        Counts consecutive iterations whose primal step length is below
        resto_trigger_alpha at a still-infeasible point, and returns True
        once resto_trigger_iters of them accumulate. The counter resets on
        the first healthy step and when the trigger fires.
        """
        stalled = (
            state.max_alpha_primal < self.options["resto_trigger_alpha"]
            and state.con_infeasibility > self.options["acceptable_constr_viol_tol"]
        )
        if not stalled:
            self.small_alpha_count = 0
            return False

        self.small_alpha_count += 1
        if self.small_alpha_count >= self.options["resto_trigger_iters"]:
            self.small_alpha_count = 0
            return True
        return False

    def reset_step_watchdog(self, state):
        """Clear the stale step norm so the watchdog does not re-fire."""
        self.diverging_step_count = 0
        state.raw_step_norm = 0.0

    def test_convergence(self, evaluator, state):
        """
        Check for convergence
        """
        tol = self.options["convergence_tolerance"]
        dual_inf_tol = self.options["dual_inf_tol"]
        constr_viol_tol = self.options["constr_viol_tol"]
        compl_inf_tol = self.options["compl_inf_tol"]
        diverging_iterates_tol = self.options["diverging_iterates_tol"]
        acceptable_tol = self.options["acceptable_tol"]
        acceptable_iter = self.options["acceptable_iter"]
        acceptable_dual_inf_tol = self.options["acceptable_dual_inf_tol"]
        acceptable_constr_viol_tol = self.options["acceptable_constr_viol_tol"]
        acceptable_compl_inf_tol = self.options["acceptable_compl_inf_tol"]

        # Current iteration counter
        iteration = state.iter

        # Evaluate the residual to compute the KKT error metrics
        evaluator.evaluate_residual(state)

        # Compute NLP error components at mu_target=0
        d_inf_nlp = state.dual_infeas
        p_inf_nlp = state.primal_infeas
        c_inf_nlp = state.complementarity
        overall_error = state.kkt_error

        # A non-finite iterate cannot recover, every test is False on NaN
        if not np.isfinite(overall_error):
            if state.comm_rank == 0:
                print("  Non-finite KKT error: terminating (restoration needed)")
            return RESTORATION_NEEDED

        # Primary convergence: ALL 4 conditions must hold
        if (
            overall_error <= tol
            and d_inf_nlp <= dual_inf_tol
            and p_inf_nlp <= constr_viol_tol
            and c_inf_nlp <= compl_inf_tol
        ):
            return CONVERGED

        x_max = self.problem.maxabs(state.current.get_solution())
        if x_max > diverging_iterates_tol:
            if state.comm_rank == 0:
                print(f"  Diverging iterates: max |x| = {x_max:.2e}")
            return DIVERGED

        # Persistently large Newton steps indicate an unreliable factorization
        step_norm = state.raw_step_norm
        diverging_step_tol = self.options["diverging_step_tol"]
        if diverging_step_tol > 0.0 and step_norm > diverging_step_tol:
            self.diverging_step_count += 1
            if self.diverging_step_count >= self.options["diverging_step_iters"]:
                if state.comm_rank == 0:
                    print(
                        f"  Diverging step: ||d|| = {step_norm:.2e} for "
                        f"{self.diverging_step_count} iterations"
                    )
                return DIVERGED
        else:
            self.diverging_step_count = 0

        # Acceptable convergence
        is_acceptable = (
            overall_error <= acceptable_tol
            and d_inf_nlp <= acceptable_dual_inf_tol
            and p_inf_nlp <= acceptable_constr_viol_tol
            and c_inf_nlp <= acceptable_compl_inf_tol
        )
        if acceptable_iter > 0 and is_acceptable:
            self.acceptable_counter += 1
            if self.acceptable_counter >= acceptable_iter:
                return CONVERGED_ACCEPTABLE
        else:
            self.acceptable_counter = 0

        # Precision floor: bit-identical residuals
        denom = max(state.residual_norm, 1e-30)
        rel_change = abs(state.residual_norm - self.prev_res_norm) / denom

        # Update the previous residual norm
        self.prev_res_norm = state.residual_norm

        if rel_change < 1e-14 and iteration > 0:
            self.precision_floor_count += 1
        else:
            self.precision_floor_count = 0
        if self.precision_floor_count >= 3 and is_acceptable:
            if state.comm_rank == 0:
                print(
                    f"  Precision floor: residual unchanged "
                    f"for {self.precision_floor_count} iterations"
                )
            return PRECISION_FLOOR

        # A stall at a non-acceptable point requires restoration
        if self.precision_floor_count >= 6:
            if state.comm_rank == 0:
                print(
                    f"  Stalled at non-acceptable point for "
                    f"{self.precision_floor_count} iterations (restoration needed)"
                )
            return RESTORATION_NEEDED

        return CONTINUE
