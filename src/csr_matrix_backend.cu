#ifndef AMIGO_CSR_MATRIX_CUDA_BACKEND_H
#define AMIGO_CSR_MATRIX_CUDA_BACKEND_H

#include "amigo.h"

namespace amigo {

namespace detail {

template <typename T>
AMIGO_KERNEL void fill_values(int n, T value, T* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    array[i] = value;
  }
}

template <typename T>
void fill_values_cuda(int n, T value, T* array, cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (n + TPB - 1) / TPB;
  fill_values<T><<<grid, TPB, 0, stream>>>(n, value, array);
}

template <typename T>
AMIGO_KERNEL void add_scalar(int n, T value, T* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    array[i] += value;
  }
}

template <typename T>
void add_scalar_cuda(int n, T value, T* array, cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (n + TPB - 1) / TPB;
  add_scalar<T><<<grid, TPB, 0, stream>>>(n, value, array);
}

template <typename T>
AMIGO_KERNEL void add_array_values(int num_variables, const int* indices,
                                   const T* d_src, T* d_dest) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  int idx = indices[i];
  if (idx >= 0) {
    d_dest[idx] += d_src[i];
  }
}

template <typename T>
void add_diagonal_cuda(int nrows, const int* d_indices, const T* d_values,
                       T* d_data, cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nrows + TPB - 1) / TPB;
  add_array_values<T>
      <<<grid, TPB, 0, stream>>>(nrows, d_indices, d_values, d_data);
}

template <typename T>
AMIGO_KERNEL void zero_at_indices(int nentries, const int* d_indices,
                                  T* d_array) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nentries;
       i += blockDim.x * gridDim.x) {
    d_array[d_indices[i]] = 0.0;
  }
}

template <typename T>
void zero_at_indices_cuda(int nentries, const int* d_indices, T* d_array,
                          cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nentries + TPB - 1) / TPB;
  zero_at_indices<T><<<grid, TPB, 0, stream>>>(nentries, d_indices, d_array);
}

template <typename T>
AMIGO_KERNEL void set_value_at_indices(const T value, int nentries,
                                       const int* d_indices, T* d_array) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nentries;
       i += blockDim.x * gridDim.x) {
    d_array[d_indices[i]] = value;
  }
}

template <typename T>
void set_value_at_indices_cuda(const T value, int nentries,
                               const int* d_indices, T* d_array,
                               cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nentries + TPB - 1) / TPB;
  set_value_at_indices<T>
      <<<grid, TPB, 0, stream>>>(value, nentries, d_indices, d_array);
}

template <typename T>
AMIGO_KERNEL void row_maxabs_kernel(int nrows, const int* rowp, const T* data,
                                    T* out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    T m = 0.0;
    for (int jp = rowp[row]; jp < rowp[row + 1]; jp++) {
      T a = data[jp] < 0 ? -data[jp] : data[jp];
      if (a > m) {
        m = a;
      }
    }
    out[row] = m;
  }
}

template <typename T>
void row_maxabs_cuda(int nrows, const int* d_rowp, const T* d_data, T* d_out,
                     cudaStream_t stream) {
  constexpr int TPB = 256;
  int grid = (nrows + TPB - 1) / TPB;
  row_maxabs_kernel<T><<<grid, TPB, 0, stream>>>(nrows, d_rowp, d_data, d_out);
}

template <typename T>
AMIGO_KERNEL void scale_symmetric_kernel(int nrows, const int* rowp,
                                         const int* cols, T* data,
                                         const T* d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    T dr = d[row];
    for (int jp = rowp[row]; jp < rowp[row + 1]; jp++) {
      data[jp] *= dr * d[cols[jp]];
    }
  }
}

template <typename T>
void scale_symmetric_cuda(int nrows, const int* d_rowp, const int* d_cols,
                          T* d_data, const T* d_diagvals,
                          cudaStream_t stream) {
  constexpr int TPB = 256;
  int grid = (nrows + TPB - 1) / TPB;
  scale_symmetric_kernel<T>
      <<<grid, TPB, 0, stream>>>(nrows, d_rowp, d_cols, d_data, d_diagvals);
}

template void fill_values_cuda<double>(int n, double value, double* array,
                                       cudaStream_t stream);

template void fill_values_cuda<float>(int n, float value, float* array,
                                      cudaStream_t stream);

template void add_scalar_cuda<double>(int n, double value, double* array,
                                      cudaStream_t stream);

template void add_scalar_cuda<float>(int n, float value, float* array,
                                     cudaStream_t stream);

template void add_diagonal_cuda<double>(int nrows, const int* d_indices,
                                        const double* d_values, double* d_data,
                                        cudaStream_t stream);

template void row_maxabs_cuda<double>(int nrows, const int* d_rowp,
                                      const double* d_data, double* d_out,
                                      cudaStream_t stream);

template void row_maxabs_cuda<float>(int nrows, const int* d_rowp,
                                     const float* d_data, float* d_out,
                                     cudaStream_t stream);

template void scale_symmetric_cuda<double>(int nrows, const int* d_rowp,
                                           const int* d_cols, double* d_data,
                                           const double* d_diagvals,
                                           cudaStream_t stream);

template void scale_symmetric_cuda<float>(int nrows, const int* d_rowp,
                                          const int* d_cols, float* d_data,
                                          const float* d_diagvals,
                                          cudaStream_t stream);

template void add_diagonal_cuda<float>(int nrows, const int* d_indices,
                                       const float* d_values, float* d_data,
                                       cudaStream_t stream);

template void zero_at_indices_cuda<double>(int nentries, const int* d_indices,
                                           double* d_array,
                                           cudaStream_t stream);

template void zero_at_indices_cuda<float>(int nentries, const int* d_indices,
                                          float* d_array, cudaStream_t stream);

template void set_value_at_indices_cuda<double>(double value, int nentries,
                                                const int* d_indices,
                                                double* d_array,
                                                cudaStream_t stream);

template void set_value_at_indices_cuda<float>(float value, int nentries,
                                               const int* d_indices,
                                               float* d_array,
                                               cudaStream_t stream);

}  // namespace detail

}  // namespace amigo

#endif