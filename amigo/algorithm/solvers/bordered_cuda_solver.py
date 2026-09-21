"""Bordered GPU direct solver for KKT systems with dense hub columns.

A globally coupled scalar, such as a free final time linked into every
element, puts a dense column into the KKT matrix that makes the cuDSS
symbolic analysis superlinear and serializes its numeric factorization.
This solver detects those columns from the sparsity pattern, factors the
hub-free block with cuDSS, and eliminates the border exactly through a
small Schur complement, so the iterates match factoring the full matrix.

The regularization, perturbed-pivot accounting, and residual trust check
are inherited from DirectCudaSolver. Here the residual covers the whole
bordered elimination, not just the hub-free factorization.
"""

from .cuda_solver import DirectCudaSolver


class BorderedCudaSolver(DirectCudaSolver):
    _factor_class = "CSRMatFactorCudaBordered"
    _requires = "CUDA/cuDSS"

    def _report_backend(self):
        if self.solver.num_bordered() == 0:
            print(
                f"  {self._name}: no hub columns detected, behaves as "
                "the plain cuDSS solver"
            )
