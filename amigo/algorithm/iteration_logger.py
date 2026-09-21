"""Iteration-data assembly and progress-table printing.

Builds the per-iteration record (objective, NLP error, step sizes,
step norm, filter size, etc.) and prints the IPM progress table.
"""

import sys
import time

from .convergence_check import (
    CONTINUE,
    CONVERGED,
    CONVERGED_ACCEPTABLE,
    PRECISION_FLOOR,
    DIVERGED,
)


class OptimizationLogger:
    def __init__(self, log_objs, options, problem, optimizer):
        self.options = options
        self.log_objs = log_objs
        self.problem = problem
        self.optimizer = optimizer

        # Set the start time on creation of the optimization logger
        self.start_time = time.perf_counter()

        # Store the optimization data
        self.opt_data = {"options": self.options, "converged": False, "iterations": []}

        # Counter of steps whose dual infeasibility left its linear model
        self._prev_inf_du = None
        self._dual_viol_count = 0
        return

    def get_data(self):
        return self.opt_data

    def log_iteration(self, status, state):
        """Assemble the dict of per-iteration diagnostics for logging/history."""
        elapsed_time = time.perf_counter() - self.start_time

        iter_data = {
            "status": status,
            "iteration": state.iter,
            "time": elapsed_time,
            "residual": state.residual_norm,
            "barrier_param": state.mu,
        }

        # Add additional iteration-dependent data
        for obj in self.log_objs:
            obj.add_log_info(iter_data)

        # NLP-level quantities (mu=0) for display
        iter_data["inf_pr"] = state.primal_infeas
        iter_data["inf_du"] = state.dual_infeas
        iter_data["compl"] = state.complementarity
        iter_data["nlp_error"] = state.kkt_error
        iter_data["objective"] = state.barrier_objective
        px = state.step.get_solution()
        iter_data["step_norm"] = self.problem.maxabs(px)

        # Flag steps whose dual infeasibility exceeds the linear model by a slack factor
        slack = 10.0
        alpha = iter_data.get("alpha_x", 0.0)
        if self._prev_inf_du is not None and alpha > 0.0:
            bound = (1.0 - alpha) * self._prev_inf_du + slack * max(
                self._prev_inf_du, 1.0
            )
            if state.dual_infeas > bound:
                self._dual_viol_count += 1
                iter_data["dual_viol"] = True
        iter_data["dual_viol_total"] = self._dual_viol_count
        self._prev_inf_du = state.dual_infeas

        # Write out the values
        self._write_log(status, state, iter_data)

        # Append the iteration data
        self.opt_data["iterations"].append(iter_data)

        if status == CONVERGED:
            self.opt_data["converged"] = True
        elif status == CONVERGED_ACCEPTABLE:
            self.opt_data["converged"] = True
            self.opt_data["acceptable"] = True
        elif status == PRECISION_FLOOR:
            self.opt_data["converged"] = True
            self.opt_data["acceptable"] = True
            self.opt_data["precision_floor"] = True
        elif status == DIVERGED or status == CONTINUE:
            self.opt_data["converged"] = False

        return

    def _write_log(self, status, state, iter_data):
        """Print the IPM iteration table row for this iteration."""
        iteration = iter_data["iteration"]
        if iteration % 20 == 0:
            print(
                f"{'iter':>4s}  {'nlp_error':>9s}  {'objective':>12s}  "
                f"{'inf_pr':>9s}  {'inf_du':>9s}  {'compl':>9s}  "
                f"{'mu':>9s}  {'||d||':>9s}  {'delta_w':>8s}  {'ic':>2s}  "
                f"{'alpha_x':>8s}  {'alpha_z':>8s}  "
                f"{'ls':>2s}  {'filt':>4s}  {'rej':>4s}  {'frej':>4s}  {'dv':>3s}"
            )

        mu = iter_data.get("barrier_param", 1.0)
        delta_w = iter_data.get("inertia_delta", 0.0)
        nlp_err = iter_data.get("nlp_error", 0.0)
        obj = iter_data.get("objective", 0.0)
        inf_pr = iter_data.get("inf_pr", 0.0)
        inf_du = iter_data.get("inf_du", 0.0)
        compl = iter_data.get("compl", 0.0)
        step_norm = iter_data.get("step_norm", 0.0)
        ax = iter_data.get("alpha_x", 0.0)
        az = iter_data.get("alpha_z", 0.0)
        ls = iter_data.get("line_iters", 0)
        fsize = iter_data.get("filter_size", 0)
        rej = iter_data.get("rejections", 0)
        frej = iter_data.get("filter_rejections", 0)
        # Factorizations spent on inertia correction, the main GPU cost
        ic = iter_data.get("inertia_attempts", 0)

        dw_str = f"{delta_w:8.1e}" if delta_w > 0 else f"{'---':>8s}"

        dv = iter_data.get("dual_viol_total", 0)
        print(
            f"{iteration:4d}  {nlp_err:9.2e}  {obj:12.5e}  "
            f"{inf_pr:9.2e}  {inf_du:9.2e}  {compl:9.2e}  "
            f"{mu:9.2e}  {step_norm:9.2e}  {dw_str}  {ic:2d}  "
            f"{ax:8.2e}  {az:8.2e}  "
            f"{ls:2d}  {fsize:4d}  {rej:4d}  {frej:4d}  {dv:3d}"
        )

        d_inf_nlp = state.dual_infeas
        p_inf_nlp = state.primal_infeas
        c_inf_nlp = state.complementarity
        overall_error = state.kkt_error
        obj_final = state.objective_value

        if status == CONVERGED:
            if state.comm_rank == 0:
                print(f"\n{'='*70}")
                print(f"  Amigo converged in {iteration} iterations")
                print(f"{'='*70}")
                print(f"  Objective value         {obj_final:>20.10e}")
                print(f"  NLP error               {overall_error:>20.10e}")
                print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                print(f"  Barrier parameter       {state.mu:>20.10e}")
                print(f"  Total iterations        {iteration:>20d}")
                total_rej = iter_data.get("total_rejections")
                if total_rej is not None:
                    frej_total = iter_data.get("total_filter_rejections", 0)
                    print(f"  Trial rejections        {total_rej:>20d}")
                    print(f"  Filter vetoes           {frej_total:>20d}")
                print(f"{'='*70}")
        elif status == CONVERGED_ACCEPTABLE:
            if state.comm_rank == 0:
                print(f"\n{'='*70}")
                print(
                    f"  Amigo converged to acceptable point in {iteration} iterations"
                )
                print(f"{'='*70}")
                print(f"  Objective value         {obj_final:>20.10e}")
                print(f"  NLP error               {overall_error:>20.10e}")
                print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                print(f"  Total iterations        {iteration:>20d}")
                total_rej = iter_data.get("total_rejections")
                if total_rej is not None:
                    frej_total = iter_data.get("total_filter_rejections", 0)
                    print(f"  Trial rejections        {total_rej:>20d}")
                    print(f"  Filter vetoes           {frej_total:>20d}")
                print(f"{'='*70}")

        sys.stdout.flush()
