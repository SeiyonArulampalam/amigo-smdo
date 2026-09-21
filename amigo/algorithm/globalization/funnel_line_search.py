"""Funnel line search.

A monotonically contracting funnel width bounds the violation of every
accepted iterate. Backtracking, second-order correction, and restoration
plumbing are shared with the filter line search.
"""

import numpy as np

from .filter_line_search import FilterLineSearch


class Funnel:
    """Funnel width and its update rules."""

    def __init__(self, options, initial_infeasibility):
        self.width = max(
            options["funnel_ubd"], options["funnel_fact"] * initial_infeasibility
        )
        self.margin = options["funnel_beta"]
        self.kappa = options["funnel_kappa"]
        self.strategy = options["funnel_update_strategy"]

    def acceptable(self, trial_theta):
        return trial_theta <= self.width

    def sufficient_decrease(self, trial_theta):
        return trial_theta <= self.margin * self.width

    def update(self, current_theta, trial_theta):
        """Contract the width on an accepted h-type step."""
        if self.strategy == 1:
            if trial_theta < current_theta:
                self.width = max(
                    self.margin * self.width,
                    self.kappa * current_theta + (1.0 - self.kappa) * trial_theta,
                )
            else:
                self.width = self.margin * self.width
        elif self.strategy == 2:
            self.width = self.kappa * self.width + (1.0 - self.kappa) * trial_theta
        elif self.strategy == 3:
            self.width = self.margin * self.width
        else:
            raise ValueError(f"Unknown funnel_update_strategy {self.strategy}")

    def update_restoration(self, current_theta):
        """Contract toward the restored violation on restoration exit."""
        self.width = self.kappa * self.width + (1.0 - self.kappa) * current_theta


class FunnelLineSearch(FilterLineSearch):
    """Funnel acceptance on the shared backtracking/SOC scaffolding."""

    def __init__(self, options, problem, optimizer):
        super().__init__(options, problem, optimizer)
        self.funnel = None
        self.delta = options["funnel_switching_delta"]
        self.h_exponent = options["funnel_switching_infeasibility_exponent"]
        self.armijo_fraction = options["funnel_armijo_fraction"]
        self.armijo_tol = options["funnel_armijo_tolerance"]
        self.min_alpha = options["funnel_min_step_length"]
        self.require_wrt_current = options[
            "funnel_require_acceptance_wrt_current_iterate"
        ]
        self.gamma = options["funnel_gamma"]
        # Restoration bookkeeping: violation at entry and at the accepted exit
        self._resto_entry_theta = None
        self._resto_exit_theta = None
        # Set on acceptance so the caller-side update sees the step type
        self._accepted_htype = False

    def reset_on_new_barrier(self, state, barrier_info):
        # No reset on mu changes, the width bounds the mu-independent violation
        return

    def set_reference_values(self, ref_theta):
        # theta_max and theta_min are filter devices, replaced by the funnel
        # Initialize the funnel width from the first reference violation
        self.theta_min = 0.0
        self.theta_max = np.inf
        if self.funnel is None:
            self.funnel = Funnel(self.options, ref_theta)

    def compute_alpha_min(self, base):
        # Plain backtracking down to a fixed minimum step length
        return self.min_alpha

    def is_ftype(self, base, alpha_test):
        """Switching condition on the alpha-scaled model decrease."""
        pred = alpha_test * (-base.ref_dphi)
        return (
            base.ref_dphi < 0.0 and pred > self.delta * base.ref_theta**self.h_exponent
        )

    def armijo_holds(self, base, trial_barr, alpha_test):
        """Armijo sufficient decrease on the barrier objective:
        actual >= fraction * max(0, predicted - tol), with roundoff
        protection on the actual reduction."""
        actual = base.ref_barr - trial_barr
        actual += self.EPS10 * abs(base.ref_barr)
        pred = alpha_test * (-base.ref_dphi)
        return actual >= self.armijo_fraction * max(0.0, pred - self.armijo_tol)

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        self.reject_filter = False
        self._accepted_htype = False

        if not self.funnel.acceptable(trial_theta):
            return False

        if self.require_wrt_current:
            feas = trial_theta < self.funnel.margin * base.ref_theta
            obj = trial_barr <= base.ref_barr - self.gamma * trial_theta
            if not (feas or obj):
                return False

        if self.is_ftype(base, alpha_test):
            return self.armijo_holds(base, trial_barr, alpha_test)

        if self.funnel.sufficient_decrease(trial_theta):
            self._accepted_htype = True
            return True

        return False

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):
        # Only h-type steps contract the funnel, nothing else is stored
        if self._accepted_htype:
            self.funnel.update(base.ref_theta, trial_theta)

    def augment_filter(self, phi, theta):
        # Restoration entry: remember the violation for the exit test
        if np.isfinite(theta):
            self._resto_entry_theta = theta

    def restoration_acceptable(self, phi, theta):
        """Exit test: theta <= beta * min(width, theta at entry)."""
        ref = self.funnel.width if self.funnel is not None else np.inf
        if self._resto_entry_theta is not None:
            ref = min(ref, self._resto_entry_theta)
        ok = theta <= self.funnel.margin * ref if self.funnel is not None else True
        if ok:
            self._resto_exit_theta = theta
        return ok

    def reset_after_restoration(self):
        super().reset_after_restoration()
        if self.funnel is not None and self._resto_exit_theta is not None:
            self.funnel.update_restoration(self._resto_exit_theta)
        self._resto_entry_theta = None
        self._resto_exit_theta = None
