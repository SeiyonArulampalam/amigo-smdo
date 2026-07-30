"""Filter line search with second-order correction.

Trial points are accepted against a two-dimensional filter over the barrier
objective and the 1-norm constraint violation. A second-order correction is
applied on the first rejection.
"""

from abc import ABC, abstractmethod

import numpy as np
from .filter_acceptance import Filter


class LineSearch(ABC):
    def reset_on_new_barrier(self, state, barrier_info):
        pass

    def reset_after_restoration(self):
        pass

    def augment_filter(self, phi, theta):
        pass

    def restoration_acceptable(self, phi, theta):
        return True

    @abstractmethod
    def line_search(self, solver, evaluator, state):
        pass


class LineSearchInfo:
    success: bool = False
    num_search_iters: int = 0
    alpha_primal: float = 1.0
    alpha_dual: float = 1.0


class FilterLineSearch(LineSearch):
    """Filter-based line search, SOC, and watchdog procedure."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.trial = self.optimizer.create_opt_vector()
        self.trial_grad = self.problem.create_vector()
        self.trial_res = self.problem.create_vector()

        # Second-order correction workspace
        self.soc_rhs = self.problem.create_vector()
        self.soc_sol = self.problem.create_vector()
        self.soc_step = self.optimizer.create_opt_vector()
        self.soc_trial = self.optimizer.create_opt_vector()
        self.soc_trial_grad = self.problem.create_vector()
        self.soc_con_a = self.problem.create_constraint_vector()
        self.soc_con_b = self.problem.create_constraint_vector()

        self.filter = Filter()

        # Constant = 10 times machine precision
        self.EPS10 = 10.0 * np.finfo(float).eps

        # The theta limits need the first iterate, set them lazily
        self.theta_limits_initialized = False

        self.current_info = None

        # Successive filter rejections drive the filter reset trigger
        self.successive_filter_rejections = 0
        self.filter_reset_count = 0

        # Whether the last acceptance check was rejected by the filter history
        self.reject_filter = False

        return

    def reset_on_new_barrier(self, state, barrier_info):
        """Drop the filter when the barrier parameter changes."""

        # Entries store phi_mu, a new mu makes them incomparable
        reset = bool(barrier_info.new_barrier)

        if reset:
            self.filter.clear()
            self.successive_filter_rejections = 0
            self.filter_reset_count = 0
            # theta limits are fixed at the initial point for the whole solve

        return

    class FilterBasePoint:
        ref_barr: float
        ref_theta: float
        ref_dphi: float

    def reset_after_restoration(self):
        # Keep the filter, it holds the failure point that forced restoration
        # Re-key the theta limits on the restored point
        self.successive_filter_rejections = 0
        self.theta_limits_initialized = False

    def augment_filter(self, phi, theta):
        # Record the failure point so the search cannot return there
        if np.isfinite(phi) and np.isfinite(theta):
            self.filter.add(phi, theta)

    def restoration_acceptable(self, phi, theta):
        if self.theta_limits_initialized and theta > self.theta_max:
            return False
        return self.filter.is_acceptable(phi, theta)

    def set_reference_values(self, ref_theta):
        # Infeasibility floor and ceiling, keyed to the initial violation
        self.theta_min = 1e-4 * max(1.0, ref_theta)
        self.theta_max = self.options["filter_theta_max_fact"] * max(1.0, ref_theta)
        return

    def is_ftype(self, base, alpha_test):
        """An f-type step primarily reduces the barrier objective."""
        delta = self.options["filter_delta"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]

        return (
            base.ref_dphi < 0.0
            and alpha_test * (-base.ref_dphi) ** s_phi > delta * base.ref_theta**s_theta
        )

    def armijo_holds(self, base, trial_barr, alpha_test):
        """Check if the Armijo condition holds relative to the base point"""
        eta_phi = self.options["filter_eta_phi"]

        return (
            trial_barr - base.ref_barr
        ) - eta_phi * alpha_test * base.ref_dphi <= self.EPS10 * abs(base.ref_barr)

    def acceptable_to_iterate(self, base, trial_barr, trial_theta):
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        obj_max_inc = 5.0

        if trial_barr > base.ref_barr:
            basval = 1.0
            if abs(base.ref_barr) > 10.0:
                basval = np.log10(abs(base.ref_barr))
            if np.log10(max(trial_barr - base.ref_barr, 1e-300)) > obj_max_inc + basval:
                return False

        feas_reduce = trial_theta - (1.0 - gamma_theta) * base.ref_theta
        feas_check = feas_reduce <= self.EPS10 * abs(base.ref_theta)

        obj_reduce = (trial_barr - base.ref_barr) + gamma_phi * base.ref_theta
        obj_check = obj_reduce <= self.EPS10 * abs(base.ref_barr)

        return feas_check or obj_check

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        # Only a filter-history rejection counts toward the reset trigger
        self.reject_filter = False
        if trial_theta > self.theta_max:
            return False
        if (
            alpha_test > 0.0
            and self.is_ftype(base, alpha_test)
            and base.ref_theta <= self.theta_min
        ):
            return self.armijo_holds(base, trial_barr, alpha_test)

        if not self.acceptable_to_iterate(base, trial_barr, trial_theta):
            return False

        accepted = self.filter.is_acceptable(trial_barr, trial_theta)
        self.reject_filter = not accepted
        return accepted

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):
        is_ftype = (
            base.ref_theta <= self.theta_min
            and self.is_ftype(base, alpha_test)
            and self.armijo_holds(base, trial_barr, alpha_test)
        )

        if not is_ftype:
            self.filter.add(base.ref_barr, base.ref_theta)

    def build_base_point(self, evaluator, state):
        base = self.FilterBasePoint()

        # Evaluate the objective and infeasibility at this point
        evaluator.evaluate_objective_and_infeasibility(state)
        base.ref_barr = state.barrier_objective
        base.ref_theta = state.con_infeasibility

        # Evaluate the directional derivative
        base.ref_dphi = evaluator.evaluate_directional_derivative(state)

        return base

    def compute_alpha_min(self, base):
        # The smallest step length the search attempts before restoration
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]
        delta = self.options["filter_delta"]
        alpha_min_frac = 0.05

        alpha_min = gamma_theta
        if base.ref_dphi < 0.0:
            alpha_min = min(gamma_theta, gamma_phi * base.ref_theta / (-base.ref_dphi))
            if base.ref_theta <= self.theta_min:
                alpha_min = min(
                    alpha_min,
                    delta * base.ref_theta**s_theta / (-base.ref_dphi) ** s_phi,
                )
        alpha_min *= alpha_min_frac

        return alpha_min

    def _second_order_correction(
        self, solver, evaluator, state, base, alpha_first, theta_first
    ):
        """Second-order correction on the first rejection.

        Reuses the current KKT factorization: the rhs keeps the Newton
        residual's gradient rows but replaces the constraint rows with the
        corrected violation c_soc = alpha*c_k + c(trial). Iterates while
        the violation shrinks by kappa_soc per round, and the corrected
        trial must pass the ordinary acceptance test. Returns the accepted
        trial tuple or None, leaving the state untouched for the normal
        backtracking.
        """
        max_soc = self.options["filter_max_soc"]
        kappa_soc = self.options["filter_kappa_soc"]
        if max_soc <= 0:
            return None
        con_indices = self.problem.get_constraint_indices()

        # Residual at the current point (rhs convention of the Newton solve)
        evaluator.evaluate_residual(state)
        self.soc_rhs.copy(state.residual)
        state.residual.get_values_at(con_indices, self.soc_con_a)

        # Constraint rows of the residual at the rejected trial point
        evaluator.evaluate_residual_from_point(
            state.mu, self.trial, self.trial_grad, self.trial_res
        )
        self.trial_res.get_values_at(con_indices, self.soc_con_b)

        # c_soc = alpha * c_k + c(trial)
        self.soc_con_a.scale(alpha_first)
        self.soc_con_a.axpy(1.0, self.soc_con_b)
        self.soc_rhs.set_values_at(con_indices, self.soc_con_a)

        theta_old = theta_first
        for _ in range(max_soc):
            solver.solve(self.soc_rhs, self.soc_sol)
            self.optimizer.compute_update(
                state.mu, state.current, self.soc_sol, self.soc_step
            )
            ax, _, az, _ = self.optimizer.compute_max_step(
                state.tau, state.current, self.soc_step
            )
            self.optimizer.apply_step_update(
                ax, az, state.current, self.soc_step, self.soc_trial
            )
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.soc_trial, self.soc_trial_grad
            )
            fobj, barrier, infeas = (
                evaluator.evaluate_objective_and_infeasibility_from_point(
                    state.mu, state.obj_scale, self.soc_trial, self.soc_trial_grad
                )
            )
            tb, tt = fobj + barrier, infeas
            finite = np.isfinite(tb) and np.isfinite(tt)
            # Acceptance is tested with the original trial alpha
            if finite and self.check_acceptance(base, tb, tt, alpha_first):
                return (fobj, barrier, infeas, tb, tt, ax, az)
            if not finite or tt > kappa_soc * theta_old:
                return None
            theta_old = tt

            # Next round: c_soc = alpha_soc * c_soc + c(trial_soc)
            evaluator.evaluate_residual_from_point(
                state.mu, self.soc_trial, self.soc_trial_grad, self.trial_res
            )
            self.trial_res.get_values_at(con_indices, self.soc_con_b)
            self.soc_con_a.scale(ax)
            self.soc_con_a.axpy(1.0, self.soc_con_b)
            self.soc_rhs.set_values_at(con_indices, self.soc_con_a)

        return None

    def line_search(self, solver, evaluator, state):
        # Base point for the acceptance tests
        base = self.build_base_point(evaluator, state)

        if not self.theta_limits_initialized:
            self.set_reference_values(base.ref_theta)
            self.theta_limits_initialized = True

        # Reset the filter after too many successive filter rejections
        if (
            self.successive_filter_rejections >= self.options["filter_reset_trigger"]
            and self.filter_reset_count < self.options["max_filter_resets"]
        ):
            self.filter.clear()
            self.filter_reset_count += 1
            self.successive_filter_rejections = 0

        alpha_min = self.compute_alpha_min(base)

        alpha_primal = state.max_alpha_primal
        alpha_dual = state.max_alpha_dual

        # Bound the mismatch between the primal and dual step lengths
        if alpha_primal < 0.1 * alpha_dual:
            alpha_dual = alpha_primal
        elif alpha_dual < 0.1 * alpha_primal:
            alpha_primal = alpha_dual

        max_line_iters = self.options["max_line_search_iterations"]

        # Failure info by default, overwritten on acceptance
        info = LineSearchInfo()
        info.success = False
        info.num_search_iters = max_line_iters

        for line_iter in range(max_line_iters):
            self.optimizer.apply_step_update(
                alpha_primal, alpha_dual, state.current, state.step, self.trial
            )
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.trial, self.trial_grad
            )
            fobj, barrier, infeas = (
                evaluator.evaluate_objective_and_infeasibility_from_point(
                    state.mu, state.obj_scale, self.trial, self.trial_grad
                )
            )
            trial_barr = fobj + barrier
            trial_theta = infeas

            # Non-finite trial values can never be accepted
            trial_finite = np.isfinite(trial_barr) and np.isfinite(trial_theta)
            # The f-type and Armijo tests keep this alpha, also after a SOC
            alpha_test = alpha_primal
            accepted = trial_finite and self.check_acceptance(
                base, trial_barr, trial_theta, alpha_test
            )

            # Track successive filter rejections of the first trial
            if line_iter == 0:
                if accepted or not self.reject_filter:
                    self.successive_filter_rejections = 0
                else:
                    self.successive_filter_rejections += 1

            if (
                not accepted
                and line_iter == 0
                and trial_finite
                and self.options["second_order_correction"]
                and trial_theta >= base.ref_theta
            ):
                soc = self._second_order_correction(
                    solver, evaluator, state, base, alpha_primal, trial_theta
                )
                if soc is not None:
                    (
                        fobj,
                        barrier,
                        infeas,
                        trial_barr,
                        trial_theta,
                        alpha_primal,
                        alpha_dual,
                    ) = soc
                    self.trial.copy(self.soc_trial)
                    self.trial_grad.copy(self.soc_trial_grad)
                    accepted = True

            if not accepted and alpha_primal <= alpha_min:
                # Below alpha_min the search cannot succeed, restoration takes over
                break

            if accepted:
                self.update_filter(base, trial_barr, trial_theta, alpha_test)

                # Invalidate the state and keep the trial evaluations
                state.invalidate()
                state.objective_value = fobj
                state.log_barrier_value = barrier
                state.con_infeasibility = infeas
                state.objective_current = True

                state.current.copy(self.trial)
                state.gradient.copy(self.trial_grad)
                state.gradient_current = True

                info.success = True
                info.num_search_iters = line_iter + 1
                info.alpha_primal = alpha_primal
                info.alpha_dual = alpha_dual
                self.current_info = info
                return info

            # Backtrack the primal and dual step lengths together
            backtrack_factor = self.options["backtracking_factor"]
            alpha_new = backtrack_factor * alpha_primal
            if alpha_new > alpha_min:
                alpha_primal = alpha_new
                alpha_dual = backtrack_factor * alpha_dual
            else:
                tau = alpha_primal / alpha_min
                alpha_primal = alpha_min
                alpha_dual = tau * alpha_dual

        return info

    def add_log_info(self, info):
        if self.current_info is not None:
            info["filter_size"] = len(self.filter)
            info["line_iters"] = self.current_info.num_search_iters
            info["alpha_x"] = self.current_info.alpha_primal
            info["alpha_z"] = self.current_info.alpha_dual
