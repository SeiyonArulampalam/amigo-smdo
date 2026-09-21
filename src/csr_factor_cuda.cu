#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#ifdef AMIGO_USE_CUDSS
#include <cudss.h>

#ifndef AMIGO_CHECK_CUDSS
#define AMIGO_CHECK_CUDSS(call)                                           \
  do {                                                                    \
    auto err__ = (call);                                                  \
    if (err__ != CUDSS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuDSS error %s:%d: %d\n", __FILE__, __LINE__, \
                   int(err__));                                           \
      std::abort();                                                       \
    }                                                                     \
  } while (0)
#endif
#endif

// cuDSS in CUDA <= 12.9: uses classic CSR
#if CUDA_VERSION < 13000
#define AMIGO_CUDSS_USE_CLASSIC_CSR
#endif

#include "amigo.h"
#include "cuda/csr_factor_cuda.h"

namespace amigo {

// Residual r = A x - b with one thread per row for the solve trust check
__global__ void k_csr_residual(int n, const int* rowp, const int* cols,
                               const double* vals, const double* x,
                               const double* b, double* r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double yi = 0.0;
    for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
      yi += vals[jp] * x[cols[jp]];
    }
    r[i] = yi - b[i];
  }
}

class CudssFactorBackend {
 public:
  CudssFactorBackend(std::shared_ptr<CSRMat<double>> m,
                     double pivot_tol_in = 1e-12)
      : mat(m), pivot_tol(pivot_tol_in), n(0), nnz(0) {
#ifdef AMIGO_USE_CUDSS
    mat->get_data(&n, nullptr, &nnz, nullptr, nullptr, nullptr);

    // Handle for cudss
    AMIGO_CHECK_CUDSS(cudssCreate(&handle));

    // Create the data object
    AMIGO_CHECK_CUDSS(cudssDataCreate(handle, &data));

    // Create matrix
    int* d_rowp = nullptr;
    int* d_cols = nullptr;
    double* d_data = nullptr;
    mat->get_device_data(&d_rowp, &d_cols, &d_data);

    d_start_rowp = nullptr, d_end_rowp = nullptr;

#ifdef AMIGO_CUDSS_USE_CLASSIC_CSR
    AMIGO_CHECK_CUDSS(cudssMatrixCreateCsr(
        &A, (int64_t)n, (int64_t)n, (int64_t)nnz, d_rowp, nullptr, d_cols,
        d_data, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC,
        CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
#else
    AMIGO_CHECK_CUDA(cudaMalloc(&d_start_rowp, n * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_end_rowp, n * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMemcpy(d_start_rowp, d_rowp, n * sizeof(int),
                                cudaMemcpyDeviceToDevice));
    AMIGO_CHECK_CUDA(cudaMemcpy(d_end_rowp, d_rowp + 1, n * sizeof(int),
                                cudaMemcpyDeviceToDevice));

    AMIGO_CHECK_CUDSS(cudssMatrixCreateCsr(
        &A, (int64_t)n, (int64_t)n, (int64_t)nnz, d_rowp, nullptr, d_cols,
        d_data, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC,
        CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
#endif  // AMIGO_CUDSS_USE_CLASSIC_CSR

    // Create the configuration settings
    AMIGO_CHECK_CUDSS(cudssConfigCreate(&config));

    // Example configuration: reordering algorithm, etc.
    cudssAlgType_t reorder = CUDSS_ALG_DEFAULT;
    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                                     &reorder, sizeof(reorder)));

    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON,
                                     &pivot_tol, sizeof(pivot_tol)));

    // Iterative refinement is configured from the options via set_ir_steps
    int ir_steps = 0;
    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &ir_steps,
                                     sizeof(ir_steps)));

    AMIGO_CHECK_CUDA(cudaMalloc(&d_X, n * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(double)));
    AMIGO_CHECK_CUDSS(cudssMatrixCreateDn(&X, n, 1, n, d_X, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));
    AMIGO_CHECK_CUDSS(cudssMatrixCreateDn(&B, n, 1, n, d_B, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));

    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B));

#endif  // AMIGO_USE_CUDSS
  }

  ~CudssFactorBackend() {
#ifdef AMIGO_USE_CUDSS
    if (d_start_rowp) {
      cudaFree(d_start_rowp);
    }
    if (d_end_rowp) {
      cudaFree(d_end_rowp);
    }
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(B);
    cudssMatrixDestroy(X);
    cudaFree(d_X);
    cudaFree(d_B);
    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    cudssDestroy(handle);
#endif
  }

  void factor() {
#ifdef AMIGO_USE_CUDSS
    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, X, B));
#endif
  }

  // Raw device pointer solve for the bordered backend
  void solve_raw(const double* d_b, double* d_x) {
#ifdef AMIGO_USE_CUDSS
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_B, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));

    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, X, B));

    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_x, d_X, n * sizeof(double), cudaMemcpyDeviceToDevice));
#endif
  }

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
#ifdef AMIGO_USE_CUDSS
    solve_raw(b->get_device_array(), x->get_device_array());
#endif
  }

  void set_pivot_epsilon(double eps) {
    pivot_tol = eps;
#ifdef AMIGO_USE_CUDSS
    // cuDSS reads the config at each execute so this applies to the next factorization
    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON,
                                     &pivot_tol, sizeof(pivot_tol)));
#endif
  }

  void set_ir_steps(int steps) {
#ifdef AMIGO_USE_CUDSS
    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &steps,
                                     sizeof(steps)));
#endif
  }

  void get_inertia(int* num_pos, int* num_neg) {
    *num_pos = 0;
    *num_neg = 0;
#ifdef AMIGO_USE_CUDSS
    size_t size_written = 0;
    // cuDSS 0.7.x returns the inertia as int[2] and newer versions as int64_t[2]
    int inertia[2] = {0, 0};
    AMIGO_CHECK_CUDSS(cudssDataGet(handle, data, CUDSS_DATA_INERTIA, inertia,
                                   sizeof(inertia), &size_written));

    *num_pos = inertia[0];
    *num_neg = inertia[1];
#endif
  }

  int num_perturbed_pivots() {
    int npivots = 0;
#ifdef AMIGO_USE_CUDSS
    size_t size_written = 0;
    AMIGO_CHECK_CUDSS(cudssDataGet(handle, data, CUDSS_DATA_NPIVOTS, &npivots,
                                   sizeof(npivots), &size_written));
#endif
    return npivots;
  }

 private:
  std::shared_ptr<CSRMat<double>> mat;

  // The pivot tolerance
  double pivot_tol;
  int n, nnz;

  // The end of each row
  int* d_start_rowp;
  int* d_end_rowp;

#ifdef AMIGO_USE_CUDSS
  // Handle for the matrix
  cudssHandle_t handle;

  // Set the configuration options
  cudssConfig_t config;

  // Data created during the different phases
  cudssData_t data;

  // The matrix itself
  cudssMatrix_t A;
  cudssMatrix_t X, B;
  double *d_X, *d_B;
#endif  // AMIGO_USE_CUDSS
};

class CuSolverFactorBackend {
 public:
  CuSolverFactorBackend(std::shared_ptr<CSRMat<double>> m, int reorder_in = 3,
                        double pivot_tol_in = 1e-12)
      : mat(m), reorder(reorder_in), pivot_tol(pivot_tol_in), n(0), nnz(0) {
    // Get the matrix sizes
    mat->get_data(&n, nullptr, &nnz, nullptr, nullptr, nullptr);

    AMIGO_CHECK_CUSOLVER(cusolverSpCreate(&handle));

    // Descriptor for the original matrix A
    AMIGO_CHECK_CUSPARSE(cusparseCreateMatDescr(&descA));
    AMIGO_CHECK_CUSPARSE(
        cusparseSetMatType(descA, CUSPARSE_MATRIX_TYPE_GENERAL));
    AMIGO_CHECK_CUSPARSE(
        cusparseSetMatIndexBase(descA, CUSPARSE_INDEX_BASE_ZERO));
  }

  ~CuSolverFactorBackend() {
    cusparseDestroyMatDescr(descA);
    cusolverSpDestroy(handle);
  }

  void factor() {}

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
    // Get device-side data
    int* d_rowp = nullptr;
    int* d_cols = nullptr;
    double* d_data = nullptr;
    mat->get_device_data(&d_rowp, &d_cols, &d_data);

    double* d_b = b->get_device_array();
    double* d_x = x->get_device_array();

    int singularity = -1;
    AMIGO_CHECK_CUSOLVER(cusolverSpDcsrlsvqr(handle, n, nnz, descA, d_data,
                                             d_rowp, d_cols, d_b, pivot_tol,
                                             reorder, d_x, &singularity));
  }

  void get_inertia(int* pos, int* neg) {
    *pos = 0;
    *neg = 0;
  }

  int num_perturbed_pivots() { return 0; }

  void set_pivot_epsilon(double eps) { pivot_tol = eps; }

  void set_ir_steps(int) {}

 private:
  std::shared_ptr<CSRMat<double>> mat;

  int reorder;
  double pivot_tol;
  int n, nnz;

  cusolverSpHandle_t handle;
  cusparseMatDescr_t descA;
};

CSRMatFactorCuda::CSRMatFactorCuda(std::shared_ptr<CSRMat<double>> m,
                                   double pivot_tol)
    : mat(m) {
#ifdef AMIGO_USE_CUDSS
  obj = new CudssFactorBackend(m, pivot_tol);
#else
  obj = new CuSolverFactorBackend(m, pivot_tol);
#endif
}

CSRMatFactorCuda::~CSRMatFactorCuda() { delete obj; }

void CSRMatFactorCuda::factor() { obj->factor(); }

void CSRMatFactorCuda::solve(std::shared_ptr<Vector<double>> b,
                             std::shared_ptr<Vector<double>> x) {
  obj->solve(b, x);
}

void CSRMatFactorCuda::get_inertia(int* pos, int* neg) {
  obj->get_inertia(pos, neg);
}

int CSRMatFactorCuda::num_perturbed_pivots() {
  return obj->num_perturbed_pivots();
}

void CSRMatFactorCuda::set_pivot_epsilon(double eps) {
  obj->set_pivot_epsilon(eps);
}

void CSRMatFactorCuda::set_ir_steps(int steps) { obj->set_ir_steps(steps); }

void CSRMatFactorCuda::residual(std::shared_ptr<Vector<double>> b,
                                std::shared_ptr<Vector<double>> x,
                                std::shared_ptr<Vector<double>> r) {
  // Residual against the matrix as factorized with perturbations included
  int n = 0, nnz = 0;
  mat->get_data(&n, nullptr, &nnz, nullptr, nullptr, nullptr);
  int* d_rowp = nullptr;
  int* d_cols = nullptr;
  double* d_data = nullptr;
  mat->get_device_data(&d_rowp, &d_cols, &d_data);
  int bn = 256;
  k_csr_residual<<<(n + bn - 1) / bn, bn>>>(n, d_rowp, d_cols, d_data,
                                            x->get_device_array(),
                                            b->get_device_array(),
                                            r->get_device_array());
}

// Bordered factorization of K by Schur complement on the hub columns

__global__ void k_gather(int n, const int* src_idx, const double* src,
                         double* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[src_idx[i]];
  }
}

__global__ void k_gather_rows(int n, const int* rows, const double* src,
                              double* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[rows[i]];
  }
}

// Scatter x_full[keep_rows[i]] = w[i] - sum_t z[t] * V[t*n_b + i]
__global__ void k_combine_scatter(int n_b, int k, const int* keep_rows,
                                  const double* w, const double* V,
                                  const double* z, double* x_full) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_b) {
    double xi = w[i];
    for (int t = 0; t < k; t++) {
      xi -= z[t] * V[std::size_t(t) * n_b + i];
    }
    x_full[keep_rows[i]] = xi;
  }
}

__global__ void k_scatter_border(int nb_ent, const int* src_idx,
                                 const int* dst_idx, const double* src,
                                 double* B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nb_ent) {
    B[dst_idx[i]] = src[src_idx[i]];
  }
}

class BorderedCudssBackend {
 public:
  BorderedCudssBackend(std::shared_ptr<CSRMat<double>> m, double pivot_tol)
      : full(m), kb_backend(nullptr), num_hubs(0), n(0), nnz(0), n_b(0) {
#ifdef AMIGO_USE_CUDSS
    const int* rowp = nullptr;
    const int* cols = nullptr;
    full->get_data(&n, nullptr, &nnz, &rowp, &cols,
                   static_cast<const double**>(nullptr));

    // Hub detection by degree as row length plus column occurrences
    double ratio = 32.0;
    int max_hubs = 8;
    if (const char* e = std::getenv("AMIGO_BORDER_RATIO")) {
      ratio = std::atof(e);
    }
    if (const char* e = std::getenv("AMIGO_BORDER_MAX")) {
      max_hubs = std::atoi(e);
    }
    std::vector<long long> deg(std::size_t(n), 0);
    for (int i = 0; i < n; i++) {
      deg[std::size_t(i)] += rowp[i + 1] - rowp[i];
    }
    for (int p = 0; p < nnz; p++) {
      deg[std::size_t(cols[p])] += 1;
    }
    double avg = 2.0 * double(nnz) / double(n);
    std::vector<std::pair<long long, int>> cand;
    for (int i = 0; i < n; i++) {
      if (double(deg[std::size_t(i)]) > ratio * avg) {
        cand.push_back({deg[std::size_t(i)], i});
      }
    }
    std::sort(cand.rbegin(), cand.rend());
    if (int(cand.size()) > max_hubs) {
      cand.resize(std::size_t(max_hubs));
    }
    num_hubs = int(cand.size());
    hubs.resize(std::size_t(num_hubs));
    for (int t = 0; t < num_hubs; t++) {
      hubs[std::size_t(t)] = cand[std::size_t(t)].second;
    }
    std::sort(hubs.begin(), hubs.end());

    // Hub lookup hub_of[i] as the border index or -1
    std::vector<int> hub_of(std::size_t(n), -1);
    for (int t = 0; t < num_hubs; t++) {
      hub_of[std::size_t(hubs[std::size_t(t)])] = t;
    }

    // Old to new row mapping for the kept rows
    n_b = n - num_hubs;
    std::vector<int> keep(std::size_t(n), -1);
    std::vector<int> keep_rows_h(std::size_t(n_b), 0);
    for (int i = 0, ptr = 0; i < n; i++) {
      if (hub_of[std::size_t(i)] < 0) {
        keep[std::size_t(i)] = ptr;
        keep_rows_h[std::size_t(ptr)] = i;
        ptr++;
      }
    }

    // K_b pattern, value gather map, and border entry maps
    int* kb_rowp = new int[n_b + 1];
    std::vector<int> kb_cols_v;
    std::vector<int> kb_src_h;
    std::vector<int> b_src_h, b_dst_h;
    std::vector<int> d_src_h, d_dst_h;
    kb_rowp[0] = 0;
    for (int i = 0; i < n; i++) {
      int ti = hub_of[std::size_t(i)];
      for (int p = rowp[i]; p < rowp[i + 1]; p++) {
        int j = cols[p];
        int tj = hub_of[std::size_t(j)];
        if (ti < 0 && tj < 0) {
          kb_cols_v.push_back(keep[std::size_t(j)]);
          kb_src_h.push_back(p);
        } else if (ti >= 0 && tj >= 0) {
          // Hub to hub coupling goes to the D block, symmetrized on the host
          d_src_h.push_back(p);
          d_dst_h.push_back(ti * num_hubs + tj);
        } else {
          int t = (ti >= 0) ? ti : tj;
          int other = (ti >= 0) ? j : i;
          b_src_h.push_back(p);
          b_dst_h.push_back(t * n_b + keep[std::size_t(other)]);
        }
      }
      if (ti < 0) {
        kb_rowp[keep[std::size_t(i)] + 1] = int(kb_cols_v.size());
      }
    }
    nnz_b = int(kb_cols_v.size());
    nb_ent = int(b_src_h.size());
    nd_ent = int(d_src_h.size());
    int* kb_cols = new int[nnz_b];
    std::copy(kb_cols_v.begin(), kb_cols_v.end(), kb_cols);

    // The reduced matrix owns its host pattern and CSRMat allocates the device
    kb_mat = std::make_shared<CSRMat<double>>(n_b, n_b, nnz_b, kb_rowp,
                                              kb_cols);
    kb_backend = new CudssFactorBackend(kb_mat, pivot_tol);

    // Device-side maps and work storage
    auto upload = [](const std::vector<int>& v, int** d) {
      AMIGO_CHECK_CUDA(cudaMalloc(d, v.size() * sizeof(int)));
      AMIGO_CHECK_CUDA(cudaMemcpy(*d, v.data(), v.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
    };
    upload(kb_src_h, &d_kb_src);
    if (nb_ent > 0) {
      upload(b_src_h, &d_b_src);
      upload(b_dst_h, &d_b_dst);
    }
    if (nd_ent > 0) {
      upload(d_src_h, &d_d_src);
      upload(d_dst_h, &d_d_dst);
    }
    upload(keep_rows_h, &d_keep_rows);
    upload(hubs, &d_hubs);

    // AMIGO_BORDER_VIR refines v and AMIGO_BORDER_IR refines the bordered solve
    border_vir = 2;
    border_ir = 1;
    if (const char* e = std::getenv("AMIGO_BORDER_VIR")) {
      border_vir = std::atoi(e);
    }
    if (const char* e = std::getenv("AMIGO_BORDER_IR")) {
      border_ir = std::atoi(e);
    }

    std::size_t nbk = std::size_t(n_b) * num_hubs;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_Bcols, std::max(nbk, std::size_t(1)) *
                                              sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_V, std::max(nbk, std::size_t(1)) *
                                          sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_w, n_b * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_rb, n_b * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_res_full, std::size_t(n) * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_corr_full, std::size_t(n) * sizeof(double)));
    AMIGO_CHECK_CUDA(
        cudaMalloc(&d_Dblk, std::max(num_hubs * num_hubs, 1) * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_z, std::max(num_hubs, 1) * sizeof(double)));
    AMIGO_CHECK_CUDA(cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS
                         ? cudaSuccess
                         : cudaErrorUnknown);

    S.resize(std::size_t(num_hubs) * num_hubs, 0.0);
    Sd.resize(std::size_t(num_hubs), 0.0);
#endif  // AMIGO_USE_CUDSS
  }

  ~BorderedCudssBackend() {
#ifdef AMIGO_USE_CUDSS
    delete kb_backend;
    cudaFree(d_kb_src);
    if (nb_ent > 0) {
      cudaFree(d_b_src);
      cudaFree(d_b_dst);
    }
    if (nd_ent > 0) {
      cudaFree(d_d_src);
      cudaFree(d_d_dst);
    }
    cudaFree(d_keep_rows);
    cudaFree(d_hubs);
    cudaFree(d_Bcols);
    cudaFree(d_V);
    cudaFree(d_w);
    cudaFree(d_rb);
    cudaFree(d_res_full);
    cudaFree(d_corr_full);
    cudaFree(d_Dblk);
    cudaFree(d_z);
    cublasDestroy(blas);
#endif
  }

  int num_bordered() const { return num_hubs; }

  void factor() {
#ifdef AMIGO_USE_CUDSS
    // Gather the full matrix values with the IPM diagonal into K_b, B, and D
    double* d_full = nullptr;
    full->get_device_data(nullptr, nullptr, &d_full);
    double* d_kb = nullptr;
    kb_mat->get_device_data(nullptr, nullptr, &d_kb);

    int bn = 256;
    k_gather<<<(nnz_b + bn - 1) / bn, bn>>>(nnz_b, d_kb_src, d_full, d_kb);
    if (nb_ent > 0) {
      AMIGO_CHECK_CUDA(cudaMemset(
          d_Bcols, 0, std::size_t(n_b) * num_hubs * sizeof(double)));
      k_scatter_border<<<(nb_ent + bn - 1) / bn, bn>>>(nb_ent, d_b_src,
                                                       d_b_dst, d_full,
                                                       d_Bcols);
    }
    AMIGO_CHECK_CUDA(
        cudaMemset(d_Dblk, 0, num_hubs * num_hubs * sizeof(double)));
    if (nd_ent > 0) {
      k_scatter_border<<<(nd_ent + bn - 1) / bn, bn>>>(nd_ent, d_d_src,
                                                       d_d_dst, d_full,
                                                       d_Dblk);
    }

    // Factor the hub-free block
    kb_backend->factor();

    // Border solves V_t = K_b^{-1} B_t and S = D - B^T V
    std::vector<double> Dh(std::size_t(num_hubs) * num_hubs, 0.0);
    AMIGO_CHECK_CUDA(cudaMemcpy(Dh.data(), d_Dblk,
                                num_hubs * num_hubs * sizeof(double),
                                cudaMemcpyDeviceToHost));
    // Symmetrize the lower stored hub to hub entries
    for (int t1 = 0; t1 < num_hubs; t1++) {
      for (int t2 = 0; t2 < t1; t2++) {
        double v = Dh[std::size_t(t1) * num_hubs + t2] +
                   Dh[std::size_t(t2) * num_hubs + t1];
        Dh[std::size_t(t1) * num_hubs + t2] = v;
        Dh[std::size_t(t2) * num_hubs + t1] = v;
      }
    }
    // Solve v = K_b^{-1} b with border_vir refinement passes against K_b
    int* d_kb_rowp = nullptr;
    int* d_kb_cols = nullptr;
    double* d_kb_vals = nullptr;
    kb_mat->get_device_data(&d_kb_rowp, &d_kb_cols, &d_kb_vals);
    const double minus_one = -1.0, one = 1.0;
    for (int t = 0; t < num_hubs; t++) {
      double* d_vt = d_V + std::size_t(t) * n_b;
      double* d_bt = d_Bcols + std::size_t(t) * n_b;
      kb_backend->solve_raw(d_bt, d_vt);
      for (int pass = 0; pass < border_vir; pass++) {
        // Refinement step v += K_b^{-1} (b - K_b v)
        k_csr_residual<<<(n_b + bn - 1) / bn, bn>>>(n_b, d_kb_rowp, d_kb_cols,
                                                    d_kb_vals, d_vt, d_bt,
                                                    d_rb);
        cublasDscal(blas, n_b, &minus_one, d_rb, 1);
        kb_backend->solve_raw(d_rb, d_w);
        cublasDaxpy(blas, n_b, &one, d_w, 1, d_vt, 1);
      }
    }
    for (int t1 = 0; t1 < num_hubs; t1++) {
      for (int t2 = 0; t2 <= t1; t2++) {
        double dot = 0.0;
        cublasDdot(blas, n_b, d_Bcols + std::size_t(t1) * n_b, 1,
                   d_V + std::size_t(t2) * n_b, 1, &dot);
        cudaDeviceSynchronize();
        double sv = Dh[std::size_t(t1) * num_hubs + t2] - dot;
        S[std::size_t(t1) * num_hubs + t2] = sv;
        S[std::size_t(t2) * num_hubs + t1] = sv;
      }
    }
    // Small dense unpivoted LDL^T of S
    Sl = S;
    for (int j = 0; j < num_hubs; j++) {
      double djj = Sl[std::size_t(j) * num_hubs + j];
      for (int p = 0; p < j; p++) {
        double l = Sl[std::size_t(j) * num_hubs + p];
        djj -= l * l * Sd[std::size_t(p)];
      }
      Sd[std::size_t(j)] = djj;
      for (int i = j + 1; i < num_hubs; i++) {
        double v = Sl[std::size_t(i) * num_hubs + j];
        for (int p = 0; p < j; p++) {
          v -= Sl[std::size_t(i) * num_hubs + p] *
               Sl[std::size_t(j) * num_hubs + p] * Sd[std::size_t(p)];
        }
        Sl[std::size_t(i) * num_hubs + j] = (djj != 0.0) ? v / djj : 0.0;
      }
    }
#endif
  }

  void solve_raw(const double* d_r, double* d_x) {
#ifdef AMIGO_USE_CUDSS
    solve_once(d_r, d_x);

    // Refine against the full matrix to remove the composed border error
    if (border_ir > 0) {
      int* d_rowp = nullptr;
      int* d_cols = nullptr;
      double* d_vals = nullptr;
      full->get_device_data(&d_rowp, &d_cols, &d_vals);
      const double minus_one = -1.0, one = 1.0;
      int bn = 256;
      for (int pass = 0; pass < border_ir; pass++) {
        k_csr_residual<<<(n + bn - 1) / bn, bn>>>(n, d_rowp, d_cols, d_vals,
                                                  d_x, d_r, d_res_full);
        cublasDscal(blas, n, &minus_one, d_res_full, 1);
        solve_once(d_res_full, d_corr_full);
        cublasDaxpy(blas, n, &one, d_corr_full, 1, d_x, 1);
      }
    }
#endif
  }

 private:
  void solve_once(const double* d_r, double* d_x) {
#ifdef AMIGO_USE_CUDSS
    int bn = 256;
    // Solve w = K_b^{-1} r[keep]
    k_gather_rows<<<(n_b + bn - 1) / bn, bn>>>(n_b, d_keep_rows, d_r, d_rb);
    kb_backend->solve_raw(d_rb, d_w);

    // Form y_t = r[hub_t] - B_t . w and solve S z = y on the host
    std::vector<double> y(std::size_t(num_hubs), 0.0);
    for (int t = 0; t < num_hubs; t++) {
      double rt = 0.0;
      AMIGO_CHECK_CUDA(cudaMemcpy(&rt, d_r + hubs[std::size_t(t)],
                                  sizeof(double), cudaMemcpyDeviceToHost));
      double dot = 0.0;
      cublasDdot(blas, n_b, d_Bcols + std::size_t(t) * n_b, 1, d_w, 1, &dot);
      cudaDeviceSynchronize();
      y[std::size_t(t)] = rt - dot;
    }
    // Forward, diagonal, and backward sweeps with the small LDL
    for (int i = 0; i < num_hubs; i++) {
      for (int p = 0; p < i; p++) {
        y[std::size_t(i)] -=
            Sl[std::size_t(i) * num_hubs + p] * y[std::size_t(p)];
      }
    }
    for (int i = 0; i < num_hubs; i++) {
      y[std::size_t(i)] =
          (Sd[std::size_t(i)] != 0.0) ? y[std::size_t(i)] / Sd[std::size_t(i)]
                                      : 0.0;
    }
    for (int i = num_hubs - 1; i >= 0; i--) {
      for (int p = i + 1; p < num_hubs; p++) {
        y[std::size_t(i)] -=
            Sl[std::size_t(p) * num_hubs + i] * y[std::size_t(p)];
      }
    }

    // Recover x_b = w - sum_t z_t V_t and x[hub_t] = z_t
    AMIGO_CHECK_CUDA(cudaMemcpy(d_z, y.data(), num_hubs * sizeof(double),
                                cudaMemcpyHostToDevice));
    k_combine_scatter<<<(n_b + bn - 1) / bn, bn>>>(n_b, num_hubs, d_keep_rows,
                                                   d_w, d_V, d_z, d_x);
    for (int t = 0; t < num_hubs; t++) {
      AMIGO_CHECK_CUDA(cudaMemcpy(d_x + hubs[std::size_t(t)],
                                  &y[std::size_t(t)], sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
#endif
  }

 public:
  void get_inertia(int* pos, int* neg) {
    *pos = 0;
    *neg = 0;
#ifdef AMIGO_USE_CUDSS
    kb_backend->get_inertia(pos, neg);
    // Haynsworth adds the inertia of the Schur complement
    for (int t = 0; t < num_hubs; t++) {
      if (Sd[std::size_t(t)] > 0.0) {
        (*pos)++;
      } else if (Sd[std::size_t(t)] < 0.0) {
        (*neg)++;
      }
    }
#endif
  }

  int num_perturbed_pivots() {
#ifdef AMIGO_USE_CUDSS
    return kb_backend->num_perturbed_pivots();
#else
    return 0;
#endif
  }

  void set_pivot_epsilon(double eps) {
#ifdef AMIGO_USE_CUDSS
    kb_backend->set_pivot_epsilon(eps);
#endif
  }

  void set_ir_steps(int steps) {
#ifdef AMIGO_USE_CUDSS
    kb_backend->set_ir_steps(steps);
#endif
  }

 private:
  std::shared_ptr<CSRMat<double>> full;
  std::shared_ptr<CSRMat<double>> kb_mat;
  CudssFactorBackend* kb_backend;

  int num_hubs, n, nnz, n_b, nnz_b, nb_ent, nd_ent;
  std::vector<int> hubs;

  // Device maps and work storage
  int *d_kb_src = nullptr, *d_b_src = nullptr, *d_b_dst = nullptr;
  int *d_d_src = nullptr, *d_d_dst = nullptr;
  int *d_keep_rows = nullptr, *d_hubs = nullptr;
  double *d_Bcols = nullptr, *d_V = nullptr, *d_w = nullptr, *d_rb = nullptr;
  double *d_Dblk = nullptr, *d_z = nullptr;
  double *d_res_full = nullptr, *d_corr_full = nullptr;

  // Border refinement pass counts
  int border_vir = 2;
  int border_ir = 1;
#ifdef AMIGO_USE_CUDSS
  cublasHandle_t blas;
#endif

  // Host k x k Schur block and its LDL factors
  std::vector<double> S, Sl, Sd;
};

CSRMatFactorCudaBordered::CSRMatFactorCudaBordered(
    std::shared_ptr<CSRMat<double>> m, double pivot_tol)
    : mat(m) {
  obj = new BorderedCudssBackend(m, pivot_tol);
}

CSRMatFactorCudaBordered::~CSRMatFactorCudaBordered() { delete obj; }

void CSRMatFactorCudaBordered::factor() { obj->factor(); }

void CSRMatFactorCudaBordered::solve(std::shared_ptr<Vector<double>> b,
                                     std::shared_ptr<Vector<double>> x) {
  obj->solve_raw(b->get_device_array(), x->get_device_array());
}

void CSRMatFactorCudaBordered::get_inertia(int* pos, int* neg) {
  obj->get_inertia(pos, neg);
}

int CSRMatFactorCudaBordered::num_perturbed_pivots() {
  return obj->num_perturbed_pivots();
}

void CSRMatFactorCudaBordered::set_pivot_epsilon(double eps) {
  obj->set_pivot_epsilon(eps);
}

void CSRMatFactorCudaBordered::set_ir_steps(int steps) {
  obj->set_ir_steps(steps);
}

int CSRMatFactorCudaBordered::num_bordered() { return obj->num_bordered(); }

void CSRMatFactorCudaBordered::residual(std::shared_ptr<Vector<double>> b,
                                        std::shared_ptr<Vector<double>> x,
                                        std::shared_ptr<Vector<double>> r) {
  // Full matrix residual r = K x - b over the whole bordered elimination
  int n = 0;
  mat->get_data(&n, nullptr, nullptr, nullptr, nullptr, nullptr);
  int* d_rowp = nullptr;
  int* d_cols = nullptr;
  double* d_data = nullptr;
  mat->get_device_data(&d_rowp, &d_cols, &d_data);
  int bn = 256;
  k_csr_residual<<<(n + bn - 1) / bn, bn>>>(n, d_rowp, d_cols, d_data,
                                            x->get_device_array(),
                                            b->get_device_array(),
                                            r->get_device_array());
}

}  // namespace amigo
