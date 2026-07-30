"""Bordered GPU direct solver for KKT systems with dense hub columns.

A globally coupled scalar, such as a free final time linked into every
element, puts a dense column into the KKT matrix that makes the cuDSS
symbolic analysis superlinear and serializes its numeric factorization.
This solver detects those columns from the sparsity pattern, factors the
hub-free block with cuDSS, and eliminates the border exactly through a
small Schur complement, so the iterates match factoring the full matrix.
"""

from . import LinearSolver


class BorderedCudaSolver(LinearSolver):
    def __init__(self, options, state):
        try:
            from amigo.amigo import CSRMatFactorCudaBordered, Vector
        except Exception:
            raise NotImplementedError("Amigo compiled without CUDA/cuDSS support")

        get = options.get if hasattr(options, "get") else lambda k, d: d
        self.pivot_eps = float(get("cuda_pivot_eps", 1e-8))
        self.ir_steps = int(get("cuda_ir_steps", 2))
        self.check_residual = bool(get("cuda_check_residual", True))
        self.residual_rtol = float(get("cuda_residual_rtol", 1e-4))

        self.mat_copy = state.hessian.duplicate()
        self.solver = CSRMatFactorCudaBordered(self.mat_copy, self.pivot_eps)
        self.solver.set_ir_steps(self.ir_steps)
        if self.solver.num_bordered() == 0:
            print(
                "  BorderedCudaSolver: no hub columns detected, behaves as "
                "the plain cuDSS solver"
            )

        self._Vector = Vector
        self._tmp = None
        self._eps_eff = self.pivot_eps
        self.last_rel_residual = 0.0
        self.num_residual_violations = 0
        self.last_perturbed_pivots = 0
        self.num_perturbed_factorizations = 0

        if not get("perturb_always_cd", False):
            print(
                "  BorderedCudaSolver: set perturb_always_cd=True, static "
                "pivoting is only reliable on a quasi-definite KKT"
            )

    def factor(self, hessian, diagonal):
        self.mat_copy.copy(hessian)
        self.mat_copy.add_diagonal(diagonal)
        self.solver.factor()

        self.last_perturbed_pivots = self.solver.num_perturbed_pivots()
        if self.last_perturbed_pivots > 0:
            self.num_perturbed_factorizations += 1
            n = self.num_perturbed_factorizations
            if n <= 3 or n % 25 == 0:
                print(
                    f"  BorderedCudaSolver: {self.last_perturbed_pivots} "
                    f"statically perturbed pivots (factorization {n})"
                )

    def solve(self, bx, px):
        self.solver.solve(bx, px)

        if self.check_residual:
            # r = K p - b on device against the FULL bordered matrix
            if self._tmp is None:
                self._tmp = self._Vector(len(bx.get_array()))
            self.solver.residual(bx, px, self._tmp)
            bnorm = bx.dot(bx) ** 0.5
            rnorm = self._tmp.dot(self._tmp) ** 0.5
            self.last_rel_residual = rnorm / max(bnorm, 1e-300)
            if self.last_rel_residual > self.residual_rtol:
                self.num_residual_violations += 1
                n = self.num_residual_violations
                if n <= 3 or n % 25 == 0:
                    print(
                        f"  BorderedCudaSolver: solve residual "
                        f"{self.last_rel_residual:.2e} > rtol "
                        f"{self.residual_rtol:.1e} (violation {n})"
                    )
                if n % 3 == 0 and self.pivot_eps > 1e-14:
                    self.pivot_eps = max(self.pivot_eps / 10.0, 1e-14)
                    print(
                        f"  BorderedCudaSolver: tightening pivot epsilon to "
                        f"{self.pivot_eps:.1e}"
                    )

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.solver.get_inertia()

    def set_pivot_tolerance(self, pivtol):
        # Same monotone pivtol to epsilon mapping as DirectCudaSolver
        eps = max(1e-14, self.pivot_eps * (1e-6 / max(float(pivtol), 1e-30)))
        self._eps_eff = eps
        self.solver.set_pivot_epsilon(eps)

    def static_pivot_floor(self):
        return 10.0 * self._eps_eff

    def add_log_info(self, info):
        info["cuda_rel_residual"] = self.last_rel_residual
        info["cuda_residual_violations"] = self.num_residual_violations
        info["cuda_perturbed_pivots"] = self.last_perturbed_pivots
