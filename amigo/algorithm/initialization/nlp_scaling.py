"""Gradient-based NLP scaling, computed once at the start point.

Scales the objective through state.obj_scale and each constraint at the
problem boundary, so the derivatives run in scaled space. Multipliers are
then stored scaled.
"""

import numpy as np


class NLPScaling:
    def __init__(self, options, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer
        self.enabled = options["nlp_scaling"]
        self.g_max = options["nlp_scaling_max_gradient"]
        self.min_value = options["nlp_scaling_min_value"]

        self.obj_scale = 1.0
        self.min_con_scale = 1.0
        self.temp_primal = problem.create_primal_vector()
        self.temp_con = problem.create_constraint_vector()

    def compute(self, evaluator, state):
        """Compute the scaling factors at the current point and apply them."""
        if not self.enabled:
            return
        self.obj_scale = self._objective_scale(evaluator, state)
        state.obj_scale = self.obj_scale
        self._set_constraint_scale(evaluator, state)
        if self.obj_scale != 1.0 or self.min_con_scale != 1.0:
            print(
                f"NLP scaling: obj_scale = {self.obj_scale:.3e}, "
                f"min constraint scale = {self.min_con_scale:.3e}"
            )

    def _objective_scale(self, evaluator, state):
        """df from the inf-norm of the objective gradient at x0."""
        x = state.get_current_point()
        con_indices = self.problem.get_constraint_indices()
        primal_indices = self.problem.get_primal_indices()

        # Objective gradient alone: zero multipliers, obj_scale = 1
        x.get_values_at(con_indices, self.temp_con)
        x.fill_at(con_indices, 0.0)
        obj_scale_store = state.obj_scale
        state.obj_scale = 1.0
        state.invalidate()
        evaluator.evaluate_gradient(state)
        state.obj_scale = obj_scale_store

        state.gradient.get_values_at(primal_indices, self.temp_primal)
        max_grad_f = self.problem.maxabs(self.temp_primal)

        # Restore the multipliers and mark the state stale
        x.set_values_at(con_indices, self.temp_con)
        state.invalidate()

        if max_grad_f > self.g_max:
            return max(self.g_max / max_grad_f, self.min_value)
        return 1.0

    def _set_constraint_scale(self, evaluator, state):
        """dc[i] from the constraint Jacobian row inf-norms at x0.

        The assembled KKT has a structurally zero dual diagonal, so the dual
        row max-abs is exactly ||J_i||_inf.
        """
        con_indices = self.problem.get_constraint_indices()

        evaluator.evaluate_hessian(state)
        rowmax = self.problem.create_vector()
        state.hessian.row_maxabs(rowmax)
        rowmax.get_values_at(con_indices, self.temp_con)
        self.temp_con.copy_device_to_host()

        r = self.temp_con.get_array()
        dc = np.minimum(1.0, self.g_max / np.maximum(r, 1e-300))
        dc = np.maximum(dc, self.min_value)
        self.min_con_scale = float(dc.min())

        # dc goes on the constraint duals, primals and slacks stay unscaled
        self.temp_con.get_array()[:] = dc
        self.temp_con.copy_host_to_device()
        var_scale = self.problem.create_vector()
        var_scale.get_array()[:] = 1.0
        var_scale.copy_host_to_device()
        var_scale.set_values_at(con_indices, self.temp_con)
        self.problem.set_var_scale(var_scale)

        state.invalidate()

    def get_log_info(self):
        return {"obj_scale": self.obj_scale, "min_con_scale": self.min_con_scale}
