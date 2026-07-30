"""Feasibility restoration by proximal damped Gauss-Newton on ||c(x)||.

Entered when the line search fails, the step collapses, or the iterate
diverges. Reduces the constraint violation on the same compiled KKT and
returns control once the required reduction is met.
"""

import numpy as np

# Levenberg damping schedule and limits
ZETA_MIN = 1e-8
ZETA_MAX = 1e12
ZETA_DECREASE = 0.1
ZETA_INCREASE = 100.0
# Gentle interior push when re-initializing duals at a successful exit
EXIT_PUSH = 1e-6


class RestorationInfo:
    success: bool = False
    infeasible: bool = False
    iterations: int = 0
    theta_enter: float = 0.0
    theta_final: float = 0.0


class FeasibilityRestoration:
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.diag = problem.create_vector()
        self.sol = problem.create_vector()
        self.xr = problem.create_vector()
        self.prox_w = problem.create_vector()
        self.prox_g = problem.create_vector()
        self.tmp_primal = problem.create_primal_vector()
        self.tmp_primal2 = problem.create_primal_vector()
        self.tmp_con = problem.create_constraint_vector()
        self.step = optimizer.create_opt_vector()
        self.trial = optimizer.create_opt_vector()
        self.trial_grad = problem.create_vector()

        self.primal_indices = problem.get_primal_indices()
        self.con_indices = problem.get_constraint_indices()

    def restore(self, solver, evaluator, state, line_search):
        """Reduce the constraint violation from the current point.

        The iterate is updated in place. obj_scale and mu are restored on
        exit, mu resumes at mu_R on success, and multipliers are left for
        the driver to re-estimate.
        """
        info = RestorationInfo()
        obj_scale_store = state.obj_scale
        mu_store = state.mu
        converged = False
        try:
            theta = self._enter(evaluator, state)
            info.theta_enter = theta
            converged, theta, iters = self._minimize(
                solver,
                evaluator,
                state,
                line_search,
                theta,
                obj_scale_store,
                mu_store,
            )
            info.iterations = iters
            info.theta_final = theta
            info.success = converged
            if not converged and theta > self.options["constr_viol_tol"]:
                info.infeasible = self._certify_infeasible(evaluator, state, theta)
        finally:
            state.obj_scale = obj_scale_store
            if converged:
                # Resume warm at mu_R so exploded bound duals do not survive
                self.optimizer.initialize_duals_warm(
                    state.mu,
                    EXIT_PUSH,
                    self.options["warm_start_mult_init_max"],
                    state.current,
                )
                state.tau = max(self.options["tau_min"], 1.0 - state.mu)
            else:
                state.mu = mu_store
            state.invalidate()

        if state.comm_rank == 0:
            tag = "ok" if converged else ("infeasible" if info.infeasible else "failed")
            print(
                f"  Restoration: theta {info.theta_enter:.3e} -> "
                f"{info.theta_final:.3e} in {info.iterations} iters ({tag})"
            )
        return info

    def _enter(self, evaluator, state):
        """Repair the entry state and return theta at the repaired point."""
        x = state.get_current_point()
        state.obj_scale = 0.0
        x.fill_at(self.con_indices, 0.0)

        # mu_R = push^2 keeps Sigma = mu/gap^2 order one at the pushed point
        push = self.options["resto_bound_push"]
        state.mu = push**2

        # Unpin bound-stuck primals, else Sigma crushes every step
        self.optimizer.initialize_duals_warm(
            state.mu,
            push,
            self.options["warm_start_mult_init_max"],
            state.current,
        )
        state.invalidate()
        evaluator.evaluate_objective_and_infeasibility(state)

        # Proximal weights W^2 = mu_R min(1, 1/|x_R|)^2
        self.xr.copy(x)
        self._xr_p = self._primal_host(self.xr, self.tmp_primal).copy()
        self._w2 = (
            state.mu * np.minimum(1.0, 1.0 / np.maximum(np.abs(self._xr_p), 1e-12)) ** 2
        )
        self.tmp_primal.get_array()[:] = self._w2
        self.tmp_primal.copy_host_to_device()
        self.prox_w.zero()
        self.prox_w.set_values_at(self.primal_indices, self.tmp_primal)

        return state.con_infeasibility

    def _primal_host(self, vec, tmp):
        """Host view of a vector's primal entries."""
        vec.get_values_at(self.primal_indices, tmp)
        tmp.copy_device_to_host()
        return tmp.get_array()

    def _prox_value(self, xp):
        dx = xp - self._xr_p
        return 0.5 * float(np.dot(self._w2 * dx, dx)), dx

    def _minimize(
        self, solver, evaluator, state, line_search, theta, obj_scale_store, mu_store
    ):
        """Damped Gauss-Newton loop, returns (converged, theta, iterations)."""
        max_iters = self.options["resto_max_iterations"]
        kappa = self.options["resto_required_reduction"]
        tol = self.options["convergence_tolerance"]
        x = state.get_current_point()

        target = max(kappa * theta, tol)
        zeta = self.options["resto_bound_push"]  # initial Levenberg damping

        it = 0
        for it in range(max_iters):
            if theta <= target:
                if self._exit_acceptable(
                    evaluator, state, line_search, obj_scale_store, mu_store
                ):
                    return True, theta, it + 1
                target = max(0.5 * target, tol)
                if target <= tol:
                    break

            # Fresh least-squares solve each iteration: multipliers stay zero
            x.fill_at(self.con_indices, 0.0)
            evaluator.evaluate_objective_and_infeasibility(state)
            bar_cur = state.log_barrier_value
            evaluator.evaluate_gradient(state)

            # Proximal gradient W^2 (x - x_R), assembled on the host
            prox_cur, dx = self._prox_value(self._primal_host(x, self.tmp_primal))
            self.tmp_primal.get_array()[:] = self._w2 * dx
            self.tmp_primal.copy_host_to_device()
            self.prox_g.zero()
            self.prox_g.set_values_at(self.primal_indices, self.tmp_primal)
            state.gradient.axpy(1.0, self.prox_g)
            merit_cur = 0.5 * theta**2 + bar_cur + prox_cur

            ax, az = self._gn_direction(solver, evaluator, state, zeta)
            accepted, theta_trial = self._line_search(
                evaluator, state, ax, az, theta, merit_cur
            )

            if accepted:
                state.current.copy(self.trial)
                state.invalidate()
                theta = theta_trial
                zeta = max(ZETA_DECREASE * zeta, ZETA_MIN)
            else:
                # Levenberg: damp harder and retry from the same point
                zeta *= ZETA_INCREASE
                state.invalidate()
                if zeta > ZETA_MAX:
                    break

        return False, theta, it + 1

    def _gn_direction(self, solver, evaluator, state, zeta):
        """Factor and solve the proximal Gauss-Newton system.

        The assembled matrix must track x since it carries the constraint
        Jacobian, while the Lagrangian block stays zero at lambda = 0.
        Sigma + W^2 + zeta on the primal block and -1 on the dual block
        make the system quasi-definite, so no inertia loop is needed.
        """
        evaluator.evaluate_hessian(state)
        evaluator.evaluate_diagonal(state)
        evaluator.evaluate_residual(state)

        self.diag.copy(state.diagonal)
        self.diag.axpy(1.0, self.prox_w)
        self.diag.add_scalar_at(self.primal_indices, zeta)
        self.diag.add_scalar_at(self.con_indices, -1.0)
        solver.factor(state.hessian, self.diag)
        solver.solve(state.residual, self.sol)

        self.optimizer.compute_update(state.mu, state.current, self.sol, self.step)
        ax, _, az, _ = self.optimizer.compute_max_step(
            state.tau, state.current, self.step
        )
        return ax, az

    def _line_search(self, evaluator, state, ax, az, theta, merit_cur):
        """Backtrack on theta descent, falling back to the phase merit."""
        armijo = self.options["armijo_constant"]
        backtrack = self.options["backtracking_factor"]
        max_ls = self.options["max_line_search_iterations"]

        alpha = 1.0
        theta_trial = theta
        for _ in range(max_ls):
            self.optimizer.apply_step_update(
                alpha * ax, alpha * az, state.current, self.step, self.trial
            )
            evaluator.evaluate_gradient_from_point(0.0, self.trial, self.trial_grad)
            _, bar_trial, theta_trial = (
                evaluator.evaluate_objective_and_infeasibility_from_point(
                    state.mu, 0.0, self.trial, self.trial_grad
                )
            )
            if not np.isfinite(theta_trial):
                alpha *= backtrack
                continue
            if theta_trial <= (1.0 - armijo * alpha) * theta:
                return True, theta_trial
            if np.isfinite(bar_trial):
                prox_trial, _ = self._prox_value(
                    self._primal_host(self.trial.get_solution(), self.tmp_primal2)
                )
                merit_trial = 0.5 * theta_trial**2 + bar_trial + prox_trial
                if merit_trial <= merit_cur - armijo * alpha * max(1.0, abs(merit_cur)):
                    return True, theta_trial
            alpha *= backtrack
        return False, theta_trial

    def _exit_acceptable(
        self, evaluator, state, line_search, obj_scale_store, mu_store
    ):
        """Filter acceptability of the current point in main-phase units."""
        state.obj_scale = obj_scale_store
        mu_resto = state.mu
        state.mu = mu_store
        state.invalidate()
        evaluator.evaluate_objective_and_infeasibility(state)
        phi = state.barrier_objective
        theta = state.con_infeasibility
        state.obj_scale = 0.0
        state.mu = mu_resto
        state.invalidate()
        return line_search.restoration_acceptable(phi, theta)

    def _certify_infeasible(self, evaluator, state, theta):
        """Stationarity of 0.5||c||^2: with multipliers set to c, the
        Lagrangian gradient's primal rows are exactly J^T c."""
        x = state.get_current_point()
        state.invalidate()
        evaluator.evaluate_gradient(state)
        state.gradient.get_values_at(self.con_indices, self.tmp_con)
        x.set_values_at(self.con_indices, self.tmp_con)
        state.invalidate()
        evaluator.evaluate_gradient(state)
        state.gradient.get_values_at(self.primal_indices, self.tmp_primal)
        stat = self.problem.maxabs(self.tmp_primal)
        x.fill_at(self.con_indices, 0.0)
        return stat <= self.options["resto_infeas_tol"] * max(1.0, theta)
