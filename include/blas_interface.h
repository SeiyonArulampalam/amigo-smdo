#ifndef AMIGO_BLAS_INTERFACE_H
#define AMIGO_BLAS_INTERFACE_H

#include <complex>

extern "C" {

// Find the argmax
extern int idamax_(const int* n, const double* a, const int* inc);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void dsyrk_(const char* uplo, const char* trans, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* beta, double* c,
                   const int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void dgemm_(const char* ta, const char* tb, const int* m, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc);

// Compute y = alpha*op(A)*x + beta*y
extern void dgemv_(const char* trans, const int* m, const int* n,
                   const double* alpha, const double* A, const int* lda,
                   const double* x, const int* incx, const double* beta,
                   double* y, const int* incy);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtpsv_(const char* uplo, const char* transa, const char* diag,
                   const int* n, const double* a, double* x, const int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtptrs_(const char* uplo, const char* transa, const char* diag,
                    const int* n, const int* nrhs, const double* a, double* b,
                    const int* ldb, int* info);

// Factorization of packed storage matrices
extern void dpptrf_(const char* c, const int* n, double* ap, int* info);

// Find the argmax
extern int izamax_(const int* n, const std::complex<double>* a, const int* inc);

extern void dsytrf_(const char* uplo, const int* n, double* A, const int* lda,
                    int* ipiv, double* work, const int* lwork, int* info);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void zsyrk_(const char* uplo, const char* trans, const int* n,
                   const int* k, const std::complex<double>* alpha,
                   const std::complex<double>* a, const int* lda,
                   const std::complex<double>* beta, std::complex<double>* c,
                   const int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void zgemm_(const char* ta, const char* tb, const int* m, const int* n,
                   const int* k, const std::complex<double>* alpha,
                   const std::complex<double>* a, const int* lda,
                   const std::complex<double>* b, const int* ldb,
                   const std::complex<double>* beta, std::complex<double>* c,
                   const int* ldc);

// Compute y = alpha*op(A)*x + beta*y
extern void zgemv_(const char* trans, const int* m, const int* n,
                   const std::complex<double>* alpha,
                   const std::complex<double>* A, const int* lda,
                   const double* x, const int* incx,
                   const std::complex<double>* beta, std::complex<double>* y,
                   const int* incy);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztpsv_(const char* uplo, const char* transa, const char* diag,
                   const int* n, const std::complex<double>* a,
                   std::complex<double>* x, const int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztptrs_(const char* uplo, const char* transa, const char* diag,
                    const int* n, const int* nrhs,
                    const std::complex<double>* a, std::complex<double>* b,
                    const int* ldb, int* info);

// Factorization of packed storage matrices
extern void zpptrf_(const char* c, const int* n, std::complex<double>* ap,
                    int* info);
}

namespace amigo {

template <typenameT>
int blas_imax(const int* n, const T* a, const int* inc) {
  if constexpr (std::is_same<T, double>::value) {
    return idmax_(n, a, inc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return izmax_(n, a, inc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_imax only supports double and std::complex<double>");
    return 0;
  }
}

template <typename T>
void blas_syrk(const char* uplo, const char* trans, const int* n, const int* k,
               const T* alpha, const T* a, const int* lda, const T* beta, T* c,
               const int* ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_syrk only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_gemm(const char* ta, const char* tb, const int* m, const int* n,
               const int* k, const T* alpha, const T* a, const int* lda,
               const T* b, const int* ldb, const T* beta, T* c,
               const int* ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemm only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_gemv(const char* ta, const int* m, const int* n, const T* alpha,
               const T* a, const int* lda, const T* x, const int* incx,
               const T* beta, T* y, const int* incy) {
  if constexpr (std::is_same<T, double>::value) {
    dgemv_(ta, m, n, alpha, a, lda, x, incx, beta, y, incy);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemv_(ta, m, n, alpha, a, lda, x, incx, beta, y, incy);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemv only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_tpsv(const char* uplo, const char* transa, const char* diag, int* n,
               const T* a, T* x, const int* incx) {
  if constexpr (std::is_same<T, double>::value) {
    dtpsv_(uplo, transa, diag, n, a, x, incx);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztpsv_(uplo, transa, diag, n, a, x, incx);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tpsv only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_tptrs(const char* uplo, const char* transa, const char* diag,
                const int* n, const int* nrhs, const T* a, T* x, const int* ldx,
                int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dtptrs_(uplo, transa, diag, n, nrhs, a, x, ldx, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztptrs_(uplo, transa, diag, n, nrhs, a, x, ldx, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tptrs only supports double and std::complex<double>");
  }
}

template <typename T>
void lapack_pptrf(const char* c, const int* n, T* ap, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dpptrf_(c, n, ap, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zpptrf_(c, n, ap, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_pptrf only supports double and std::complex<double>");
  }
}

}  // namespace amigo

#endif  // AMIGO_BLAS_INTERFACE_H