#ifndef AMIGO_SPARSE_LDL_H
#define AMIGO_SPARSE_LDL_H

#include "blas_interface.h"
#include "csr_matrix.h"

namespace amigo {

template <typename T>
class SparseLDL {
 public:
  SparseLDL(std::shared_ptr<CSRMat<T>> mat) : mat(mat) {
    // Get the non-zero pattern
    int nrows, ncols;
    const int *mat_rowp, *mat_cols;
    mat->get_data(&nrows, &ncols, nullptr, &mat_rowp, &mat_cols, nullptr);

    // Perform a symbolic analysis to determine the size of the factorization
    int* parent = new int[size];  // Space for the etree
    int* Lnz = new int[size];     // Nonzeros below the diagonal
    build_tree(ncols, mat_rowp, mat_cols, parent, Lnz);

    // Find the supernodes in the matrix
    var_to_snode = new int[size];
    num_snodes = init_snodes(parent, Lnz, var_to_snode);
  }

  int num_snodes;
  int* var_to_snode;

  /**
   * @brief Perform the multifrontal factorization
   *
   *
   * Factor children that this front depends on
   * for children in front:
   *   factor_child(children);
   *
   * Find all the variables in this front
   *
   * Assemble the frontal matrix
   *
   * Pivot based on the fully summed nodes
   *
   * Compute the contribution blocks
   *

   *
   */

  void factor() {
    //
  }

 void assemble_contrib(T* F, f)

     private :
     /**
       Build the elimination tree and compute the number of non-zeros in
       each column.

       @param ncols The number of columns in the matrix
       @param Acolp The pointer into each column
       @param Arows The row indices for each matrix entry
       @param parent The elimination tree/forest
       @param Lnz The number of non-zeros in each column
     */
     void build_tree(const int ncols, const int Acolp[], const int Arows[],
                     int parent[], int Lnz[]) {
    int* flag = new int[ncols];

    for (int k = 0; k < ncols; k++) {
      parent[k] = -1;
      flag[k] = k;
      Lnz[k] = 0;

      // Loop over the k-th column of the original matrix
      int ip_end = Acolp[k + 1];
      for (int ip = Acolp[k]; ip < ip_end; ip++) {
        int i = Arows[ip];

        if (i < k) {
          // Scan up the etree
          for (; flag[i] != k; i = parent[i]) {
            if (parent[i] == -1) {
              parent[i] = k;
            }

            // L[k, i] is non-zero
            Lnz[i]++;
            flag[i] = k;
          }
        }
      }
    }

    delete[] flag;
  }

  /**
    Initialize the supernodes in the matrix

    The supernodes share the same column non-zero pattern

    @param ncols The number of columns in the matrix
    @param parent The elimination tree data
    @param Lnz The number of non-zeros per variable
    @param vtn The array of supernodes for each variable
  */
  int init_snodes(int ncols, const int parent[], const int Lnz[], int vtn[]) {
    int snode = 0;

    // First find the supernodes
    int i = 0;
    while (i < ncols) {
      vtn[i] = snode;
      i++;

      while ((parent[i - 1] == i) && (Lnz[i] == Lnz[i - 1] - 1)) {
        vtn[i] = snode;
        i++;
      }
      snode++;
    }

    return snode;
  }

  // The matrix
  std::shared_ptr<CSRMat<T>> mat;
};

}  // namespace amigo
#endif AMIGO_SPARSE_LDL_H