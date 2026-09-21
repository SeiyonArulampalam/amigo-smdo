#ifndef AMIGO_CUDA_MATRIX_FACTOR_H
#define AMIGO_CUDA_MATRIX_FACTOR_H

#include "amigo.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

class CudssFactorBackend;
class CuSolverFactorBackend;
class BorderedCudssBackend;

class CSRMatFactorCuda {
 public:
  CSRMatFactorCuda(std::shared_ptr<CSRMat<double>> m,
                   double pivot_tol_in = 1e-12);
  ~CSRMatFactorCuda();

  void factor();

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x);

  void get_inertia(int* pos, int* neg);

  // Number of statically perturbed pivots in the last factorization
  int num_perturbed_pivots();

  // Static pivot perturbation threshold used by the next factorization
  void set_pivot_epsilon(double eps);

  // Iterative refinement steps cuDSS applies inside solve
  void set_ir_steps(int steps);

  // Residual r = A x - b on the device as a trust check on the factorization
  void residual(std::shared_ptr<Vector<double>> b,
                std::shared_ptr<Vector<double>> x,
                std::shared_ptr<Vector<double>> r);

 private:
  // Pointer to the CSR matrix
  std::shared_ptr<CSRMat<double>> mat;

#ifdef AMIGO_USE_CUDSS
  CudssFactorBackend* obj;
#else
  CuSolverFactorBackend* obj;
#endif
};

// Bordered variant eliminating hub columns by Schur complement
class CSRMatFactorCudaBordered {
 public:
  CSRMatFactorCudaBordered(std::shared_ptr<CSRMat<double>> m,
                           double pivot_tol_in = 1e-12);
  ~CSRMatFactorCudaBordered();

  void factor();
  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x);
  void get_inertia(int* pos, int* neg);
  int num_perturbed_pivots();
  void set_pivot_epsilon(double eps);
  void set_ir_steps(int steps);
  void residual(std::shared_ptr<Vector<double>> b,
                std::shared_ptr<Vector<double>> x,
                std::shared_ptr<Vector<double>> r);

  // Number of hub columns moved into the border
  int num_bordered();

 private:
  std::shared_ptr<CSRMat<double>> mat;
  BorderedCudssBackend* obj;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_MATRIX_FACTOR_H