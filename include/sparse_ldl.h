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
    int nrows;
    const int *rowp, *cols;
    mat->get_data(&nrows, nullptr, nullptr, &rowp, &cols, nullptr);

    // Perform the symbolic anallysis based on the input pattern
    symbolic_analysis(nrows, rowp, cols);
  }
  ~SparseLDL() {
    delete[] snode_size;
    delete[] var_to_snode;
    delete[] snode_to_var;
    delete[] num_children;
  }

  int factor() {
    // Get the non-zero pattern
    int nrows, ncols, nnz;
    const int *rowp, *cols;
    const T* data;
    mat->get_data(&nrows, &ncols, &nnz, &rowp, &cols, &data);

    // TODO: fix this
    // Find the diagonal elements in the matrix - this should be fixed
    int* diag = new int[nrows];
    for (int i = 0; i < nrows; i++) {
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        if (cols[jp] == i) {
          diag[i] = jp;
          break;
        }
      }
    }

    // Perform the numerical factorization
    int flag = factor_numeric(nrows, diag, rowp, cols, data);

    delete[] diag;

    return flag;
  }

 private:
  // The matrix
  std::shared_ptr<CSRMat<T>> mat;

  // Number of non-zeros in the Choleksy factorization
  int cholesky_nnz;

  // Number of super nodes in the matrix
  int num_snodes;

  // Size of the super nodes
  int* snode_size;

  // Go from var to super node or super node to variable
  int* var_to_snode;
  int* snode_to_var;

  // Number of children for each super node
  int* num_children;

  // Compute the contribution blocks sizes (without delayed pivots)
  int* contrib_ptr;
  int* contrib_rows;

  class ContributionStack {
   public:
    ContributionStack(int max_index, int max_work) {
      stack_size = 0;
      idx_top = 0;
      idx = new int[max_index];
      work_top = 0;
      work = new T[max_work];
    }
    ~ContributionStack() {
      delete[] idx;
      delete[] work;
    }

   private:
    int stack_size;
    int idx_top = 0;
    int* idx;
    int work_top = 0;
    T* work;
  };

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
   * Compute the contribution block
   */
  int factor_numeric(const int ncols, const int diag[], const int colp[],
                     const int rows[], const T data[]) {
    // Create an array that indexes into the local front
    int* front_indices = new int[ncols];
    std::fill(front_indices, front_indices + ncols, -1);

    // Pick very large values for now...
    int size = 10 * cholesky_nnz;
    ContributionStack stack(size, size);

    // Another silly large guess right here...
    int max_frontal_size = 10 * cholesky_nnz;
    T* F = new T[max_frontal_size];

    for (int ks = 0, k = 0; ks < num_snodes; k += snode_size[ks], ks++) {
      // Size of the super node
      int ns = snode_size[ks];

      // Number of children for this super node
      int nchildren = num_children[ks];

      // Set the ordering of the degrees of freeom in the front
      // Number of fully summed contributions (supernode pivots + delayed
      // pivots)
      for (int j = 0; j < ns; j++) {
        int var = snode_to_var[k + j];
        front_indices[var] = j;
      }

      // Merge the fully summed rows
      int fully_summed = ns;

      // Add the additional contributions from the delayed pivots
      // .....

      // Get the entries predicted from Cholesky
      int start = contrib_ptr[ks];
      int cbsize = contrib_ptr[ks + 1] - start;
      for (int j = 0, *row = &contrib_rows[start]; j < cbsize; j++, row++) {
        front_indices[*row] = fully_summed + j;
      }

      // Get the size of the front
      int front_size = fully_summed + cbsize;

      // Zero the frontal matrix
      std::fill(F, F + front_size * front_size, 0.0);

      // Assemble the front contributions from the matrix
      // assemble_from_matrix(ks, front_size, front_indices, ncols, diag, colp,
      //  rows, data, F);

      // Assemble the front contributions from the stack
      // assemble_from_stack(ks, front_size, front_indices, stack, nchildren,
      // F);

      // Factor the frontal matrix
      // Need to return data here about the stack
      // factor_frontal(fully_summed, front_size, F);

      // Save the contribution to the factor
      //

      // Add the contribution block to the stack
      // add_contribution(F);
    }

    // Clean up the data
    delete[] front_indices;

    return 0;
  }

  /**
   * @brief Perform the symbolic analysis phase on the non-zero matrix pattern
   *
   * This performs a post-order of the elimination tree, identifies super nodes
   * based on the post-ordering and performs a count of the numbers of non-zero
   * entries in the matrices.
   *
   * @param ncols Number of columns (equal to number of rows) in the matrix
   * @param colp Pointer into each column of the matrix
   * @param rows Row indices within each column of the matrix
   */
  void symbolic_analysis(const int ncols, const int colp[], const int rows[]) {
    // Allocate storage that we'll need
    int* work = new int[3 * ncols];

    // Compute the elimination tree
    int* parent = new int[ncols];
    compute_etree(ncols, colp, rows, parent, work);

    // Find the post-ordering for the elimination tree
    int* ipost = new int[ncols];
    post_order_etree(ncols, colp, rows, parent, ipost, work);

    // Count the column non-zeros in the post-ordering
    int* Lnz = new int[ncols];
    count_column_nonzeros(ncols, colp, rows, ipost, parent, Lnz, work);

    // Count up the total number of non-zeros in the Cholesky factorization
    cholesky_nnz = 0;
    for (int i = 0; i < ncols; i++) {
      cholesky_nnz += Lnz[i];
    }

    // Use the work array as a temporary here
    int* post = work;
    for (int i = 0; i < ncols; i++) {
      post[ipost[i]] = i;
    }

    // Initialize the super nodes
    var_to_snode = new int[ncols];
    snode_to_var = new int[ncols];
    num_snodes =
        init_super_nodes(ncols, post, parent, Lnz, var_to_snode, snode_to_var);

    // Count up the size of each snode
    snode_size = new int[num_snodes];
    std::fill(snode_size, snode_size + num_snodes, 0);
    for (int i = 0; i < ncols; i++) {
      snode_size[var_to_snode[i]]++;
    }

    // Count the children of supernodes within the post-ordered elimination tree
    num_children = new int[num_snodes];
    count_super_node_children(ncols, parent, num_snodes, var_to_snode,
                              num_children, work);

    // Count up the sizes of the contribution blocks
    contrib_ptr = new int[num_snodes + 1];
    contrib_ptr[0] = 0;
    for (int is = 0, i = 0; is < num_snodes; i += snode_size[is], is++) {
      int var = snode_to_var[i + snode_size[is] - 1];
      contrib_ptr[is + 1] = Lnz[var];
    }

    // Count up the contribution block pointer
    for (int i = 0; i < num_snodes; i++) {
      contrib_ptr[i + 1] += contrib_ptr[i];
    }

    // Fill in the rows in the contribution blocks
    contrib_rows = new int[contrib_ptr[num_snodes]];
    build_nonzero_pattern(ncols, colp, rows, parent, num_snodes, snode_size,
                          var_to_snode, snode_to_var, contrib_ptr, contrib_rows,
                          work);

    delete[] work;
    delete[] parent;
    delete[] ipost;
    delete[] Lnz;
  }

  /**
   * @brief Compute the elimination tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ancestor Largest ancestor of each node
   */
  void compute_etree(const int ncols, const int colp[], const int rows[],
                     int parent[], int ancestor[]) {
    // Initialize the parent and ancestor arrays
    std::fill(parent, parent + ncols, -1);
    std::fill(ancestor, ancestor + ncols, -1);

    for (int k = 0; k < ncols; k++) {
      // Loop over the column of k
      int start = colp[k];
      int end = colp[k + 1];
      for (int ip = start; ip < end; ip++) {
        int i = rows[ip];

        while (i < k) {
          int tmp = ancestor[i];

          // Update the largest ancestor of i
          ancestor[i] = k;

          // We've reached the root of the previous tree,
          // set the parent of i to k
          if (tmp == -1) {
            parent[i] = k;
            break;
          }

          i = tmp;
        }
      }
    }
  }

  /**
   * @brief Post-order the elimination tree
   *
   * ipost[i] = j
   *
   * means that node i of the original tree is the j-th node of the
   * post-ordered tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ipost The computed post order
   * @param work Work array of size 3 * ncols
   */
  void post_order_etree(const int ncols, const int colp[], const int rows[],
                        const int parent[], int ipost[], int work[]) {
    int* head = work;
    int* next = &work[ncols];
    int* stack = &work[2 * ncols];

    std::fill(head, head + ncols, -1);
    std::fill(next, next + ncols, -1);

    // Initialize the heads of each linked list
    for (int j = ncols - 1; j >= 0; j--) {
      if (parent[j] != -1) {
        next[j] = head[parent[j]];
        head[parent[j]] = j;
      }
    }

    for (int j = 0, k = 0; j < ncols; j++) {
      if (parent[j] == -1) {
        // Perform a depth first search starting from j which is a root
        // in the etree
        k = depth_first_search(j, k, head, next, ipost, stack);
      }
    }
  }

  /**
   * @brief Perform a depth first search from node j
   *
   * @param j The root node to start from
   * @param k The post-order index
   * @param head The head of each linked list
   * @param next The next child in the linked lists
   * @param ipost The post order ipost[origin node i] = post node j
   * @param stack The stack for the depth first search
   * @return int The final post-order index
   */
  int depth_first_search(int j, int k, int head[], const int next[],
                         int ipost[], int stack[]) {
    int last = 0;     // Last position on the tack
    stack[last] = j;  // Put node j on the stack

    while (last >= 0) {
      // Look at the top of the stack and find the top node p and
      // its child i
      int p = stack[last];
      int i = head[p];

      if (i == -1) {
        // No unordered children of p left in the list
        ipost[p] = k;
        k++;
        last--;
      } else {
        // Remove i from the children of p and add i to the
        // stack to continue the depth first search
        head[p] = next[i];
        last++;
        stack[last] = i;
      }
    }

    return k;
  }

  /**
   * @brief Build the elimination tree and compute the number of non-zeros in
   * each column.
   *
   * @param ncols The number of columns in the matrix
   * @param colp The pointer into each column
   * @param rows The row indices for each matrix entry
   * @param parent The elimination tree/forest
   * @param Lnz The number of non-zeros in each column
   * @param work Work array of size ncols
   */
  void count_column_nonzeros(const int ncols, const int colp[],
                             const int rows[], const int ipost[],
                             const int parent[], int Lnz[], int work[]) {
    int* flag = work;
    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            Lnz[i]++;
            flag[i] = k;

            // Set the next parent
            i = parent[i];
          }
        }
      }
    }
  }

  /**
   * @brief Initialize the supernodes in the matrix
   *
   * The supernodes share the same column non-zero pattern
   *
   * @param ncols The number of columns in the matrix
   * @param post The etree post ordering
   * @param parent The etree parents
   * @param Lnz The number of non-zeros per variable
   * @param vtosn Variable to super node
   * @param sntov Super node to variable
   * @return int The number of super nodes
   */
  int init_super_nodes(int ncols, const int post[], const int parent[],
                       const int Lnz[], int vtosn[], int sntov[]) {
    // First find the supernodes
    int snode = 0;

    // Loop over subsequent numbers in the post-ordering
    for (int i = 0; i < ncols;) {
      int var = post[i];  // Get the original variable number

      // Set the super node
      vtosn[var] = snode;
      sntov[i] = var;
      i++;

      int next_var = post[i];
      while (i < ncols && parent[var] == next_var &&
             (Lnz[next_var] == Lnz[var] - 1)) {
        vtosn[next_var] = snode;
        var = next_var;
        sntov[i] = var;
        i++;
        if (i < ncols) {
          next_var = post[i];
        }
      }

      snode++;
    }

    return snode;
  }

  /**
   * @brief Count up the number of children for each super node
   *
   * @param ncols Number of columns
   * @param parent Parent pointer for the elimination tree
   * @param ns Number of super nodes
   * @param vtosn Variable to super node array
   * @param nchild Number of children (output)
   * @param work Work array - at least number of super nodes
   */
  void count_super_node_children(const int ncols, const int parent[],
                                 const int ns, const int vtosn[], int nchild[],
                                 int work[]) {
    int* snode_parent = work;
    std::fill(nchild, nchild + ns, 0);
    std::fill(snode_parent, snode_parent + ns, -1);

    // Set up the snode parents first
    for (int j = 0; j < ncols; j++) {
      int pj = parent[j];

      if (pj != -1) {
        int js = vtosn[j];
        int pjs = vtosn[pj];

        if (pjs != js) {
          snode_parent[js] = pjs;
        }
      }
    }

    // Count up the children within the post-ordered elmination tree
    for (int i = 0; i < ns; i++) {
      if (snode_parent[i] != -1) {
        nchild[snode_parent[i]]++;
      }
    }
  }

  /**
   * @brief Find the non-zero rows in the post-ordered column using the
   * elimination tree - this does not included delayed pivots
   *
   * Find the non-zero rows below the diagonal in a column of L.
   *
   * This utilizes the elimination tree
   *
   * @param colp Column pointer
   * @param rows Rows for each column
   * @param parent Parent in the elimination tree
   * @param row_count The number of indices found so far
   * @param row_indices The array of row indices
   * @param tag Tag for the visited nodes
   * @param flag Array of flags (tag must not be contained in flag initially)
   */
  void build_nonzero_pattern(const int ncols, const int colp[],
                             const int rows[], const int parent[], int sn,
                             int snsize[], const int vtosn[], const int sntov[],
                             const int cptr[], int cvars[], int work[]) {
    int* Lnz = work;
    int* flag = &work[ncols];
    int* snvar = &work[2 * ncols];

    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Find the last variable in each super node
    for (int ks = 0, k = 0; ks < sn; k += snsize[ks], ks++) {
      snvar[ks] = sntov[k + snsize[ks] - 1];
    }

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int iptr_end = colp[k + 1];
      for (int iptr = colp[k]; iptr < iptr_end; iptr++) {
        int i = rows[iptr];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            // Get the super node from the variable
            int is = vtosn[i];

            // Find the last variable in this super node
            int ivar = snvar[is];

            // If this is the last variable, add this to the row
            if (ivar == i) {
              cvars[cptr[is] + Lnz[is]] = k;
              Lnz[is]++;
            }

            // Flag the node
            flag[i] = k;

            // Set the next parent
            i = parent[i];
          }
        }
      }
    }
  }
};

}  // namespace amigo

#endif  // AMIGO_SPARSE_LDL_H