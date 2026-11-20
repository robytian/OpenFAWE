/* OpenFAWE no-GSL subset (OF-0013 revision 1, section 3).
 *
 * An independent implementation of the part of the GNU Scientific Library's *interface* that Makani's
 * common/c_math/linalg.c and gsl_linalg_extra.c call, so that the Makani controller library can be linked without GSL.
 * Written for OpenFAWE from GSL's public documentation (the interface: types, function names, arguments, the meaning of
 * the outputs) and from the algorithms of the numerical-linear-algebra literature; no GSL source code was consulted.
 *
 *   Householder reflector     LAPACK DLARFG's construction (beta = -sign(alpha) |x|, tau = (beta - alpha) / beta)
 *   QR                        Householder QR, Golub & Van Loan, Matrix Computations, 4th ed., algorithm 5.2.1
 *   QR with column pivoting   Businger & Golub (1965) as in LAPACK DGEQPF, with its column-norm downdating and
 *                             recomputation test
 *   SVD                       one-sided Jacobi (Hestenes 1958; Golub & Van Loan section 8.6.3) with LAPACK DGESVJ's
 *                             convergence tolerance sqrt(m) eps, columns sorted by decreasing singular value
 *   2-norm                    the scaled sum of squares of the reference BLAS DNRM2
 *   triangular solve          forward or backward substitution for every side, triangle, transpose and diagonal case
 *
 * LAPACK and the reference BLAS are distributed under the modified BSD licence; only their published algorithms are
 * followed here. Results agree with GSL's to rounding (checked by OF-0013 criterion N3), not bit for bit.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_permute_vector.h"
#include "gsl/gsl_vector.h"

#define NOGSL_ERROR(reason, code)              \
  do {                                         \
    gsl_error(reason, __FILE__, __LINE__, code); \
    return code;                               \
  } while (0)

/* ---------------------------------------------------------------- the error hook (Makani defines its own on ARM) */
#if !defined(__arm__)
void gsl_error(const char *reason, const char *file, int line, int gsl_errno) {
  fprintf(stderr, "openfawe-nogsl: %s:%d: ERROR %d: %s\n", file, line, gsl_errno, reason);
  fflush(stderr);
  abort();
}
#endif

/* ---------------------------------------------------------------- element access */
#define VEC(v, i) ((v)->data[(i) * (v)->stride])
#define MAT(m, i, j) ((m)->data[(i) * (m)->tda + (j)])

gsl_vector_view gsl_vector_view_array(double *base, size_t n) {
  gsl_vector_view view;
  view.vector.size = n;
  view.vector.stride = 1;
  view.vector.data = base;
  view.vector.block = NULL;
  view.vector.owner = 0;
  return view;
}

double gsl_vector_get(const gsl_vector *v, size_t i) {
  if (i >= v->size) {
    gsl_error("index out of range", __FILE__, __LINE__, GSL_EINVAL);
    return 0.0;
  }
  return VEC(v, i);
}

void gsl_vector_set(gsl_vector *v, size_t i, double x) {
  if (i >= v->size) {
    gsl_error("index out of range", __FILE__, __LINE__, GSL_EINVAL);
    return;
  }
  VEC(v, i) = x;
}

int gsl_vector_swap_elements(gsl_vector *v, size_t i, size_t j) {
  if (i >= v->size || j >= v->size) NOGSL_ERROR("index out of range", GSL_EINVAL);
  if (i != j) {
    double t = VEC(v, i);
    VEC(v, i) = VEC(v, j);
    VEC(v, j) = t;
  }
  return GSL_SUCCESS;
}

static gsl_matrix make_matrix(double *data, size_t n1, size_t n2, size_t tda) {
  gsl_matrix m;
  m.size1 = n1;
  m.size2 = n2;
  m.tda = tda;
  m.data = data;
  m.block = NULL;
  m.owner = 0;
  return m;
}

static gsl_vector make_vector(double *data, size_t n, size_t stride) {
  gsl_vector v;
  v.size = n;
  v.stride = stride;
  v.data = data;
  v.block = NULL;
  v.owner = 0;
  return v;
}

gsl_matrix_view gsl_matrix_view_array(double *base, size_t n1, size_t n2) {
  gsl_matrix_view view;
  view.matrix = make_matrix(base, n1, n2, n2);
  return view;
}

_gsl_matrix_const_view gsl_matrix_const_view_array(const double *base, size_t n1, size_t n2) {
  _gsl_matrix_const_view view;
  view.matrix = make_matrix((double *)base, n1, n2, n2);
  return view;
}

static int submatrix_ok(const gsl_matrix *m, size_t i, size_t j, size_t n1, size_t n2) {
  return i < m->size1 && j < m->size2 && n1 > 0 && n2 > 0 && i + n1 <= m->size1 && j + n2 <= m->size2;
}

gsl_matrix_view gsl_matrix_submatrix(gsl_matrix *m, size_t i, size_t j, size_t n1, size_t n2) {
  gsl_matrix_view view;
  if (!submatrix_ok(m, i, j, n1, n2)) {
    gsl_error("submatrix out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.matrix = make_matrix(NULL, 0, 0, 0);
    return view;
  }
  view.matrix = make_matrix(m->data + i * m->tda + j, n1, n2, m->tda);
  return view;
}

_gsl_matrix_const_view gsl_matrix_const_submatrix(const gsl_matrix *m, size_t i, size_t j, size_t n1, size_t n2) {
  _gsl_matrix_const_view view;
  if (!submatrix_ok(m, i, j, n1, n2)) {
    gsl_error("submatrix out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.matrix = make_matrix(NULL, 0, 0, 0);
    return view;
  }
  view.matrix = make_matrix(m->data + i * m->tda + j, n1, n2, m->tda);
  return view;
}

gsl_vector_view gsl_matrix_row(gsl_matrix *m, size_t i) {
  gsl_vector_view view;
  if (i >= m->size1) {
    gsl_error("row index out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + i * m->tda, m->size2, 1);
  return view;
}

gsl_vector_view gsl_matrix_column(gsl_matrix *m, size_t j) {
  gsl_vector_view view;
  if (j >= m->size2) {
    gsl_error("column index out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + j, m->size1, m->tda);
  return view;
}

gsl_vector_view gsl_matrix_subrow(gsl_matrix *m, size_t i, size_t offset, size_t n) {
  gsl_vector_view view;
  if (i >= m->size1 || n == 0 || offset + n > m->size2) {
    gsl_error("subrow out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + i * m->tda + offset, n, 1);
  return view;
}

gsl_vector_view gsl_matrix_subcolumn(gsl_matrix *m, size_t j, size_t offset, size_t n) {
  gsl_vector_view view;
  if (j >= m->size2 || n == 0 || offset + n > m->size1) {
    gsl_error("subcolumn out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + offset * m->tda + j, n, m->tda);
  return view;
}

_gsl_vector_const_view gsl_matrix_const_row(const gsl_matrix *m, size_t i) {
  _gsl_vector_const_view view;
  if (i >= m->size1) {
    gsl_error("row index out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + i * m->tda, m->size2, 1);
  return view;
}

_gsl_vector_const_view gsl_matrix_const_subrow(const gsl_matrix *m, size_t i, size_t offset, size_t n) {
  _gsl_vector_const_view view;
  if (i >= m->size1 || n == 0 || offset + n > m->size2) {
    gsl_error("subrow out of range", __FILE__, __LINE__, GSL_EINVAL);
    view.vector = make_vector(NULL, 0, 1);
    return view;
  }
  view.vector = make_vector(m->data + i * m->tda + offset, n, 1);
  return view;
}

double gsl_matrix_get(const gsl_matrix *m, size_t i, size_t j) {
  if (i >= m->size1 || j >= m->size2) {
    gsl_error("index out of range", __FILE__, __LINE__, GSL_EINVAL);
    return 0.0;
  }
  return MAT(m, i, j);
}

void gsl_matrix_set(gsl_matrix *m, size_t i, size_t j, double x) {
  if (i >= m->size1 || j >= m->size2) {
    gsl_error("index out of range", __FILE__, __LINE__, GSL_EINVAL);
    return;
  }
  MAT(m, i, j) = x;
}

void gsl_matrix_set_zero(gsl_matrix *m) {
  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j) MAT(m, i, j) = 0.0;
}

int gsl_matrix_memcpy(gsl_matrix *dest, const gsl_matrix *src) {
  if (dest->size1 != src->size1 || dest->size2 != src->size2) NOGSL_ERROR("matrix sizes are different", GSL_EBADLEN);
  for (size_t i = 0; i < src->size1; ++i)
    for (size_t j = 0; j < src->size2; ++j) MAT(dest, i, j) = MAT(src, i, j);
  return GSL_SUCCESS;
}

int gsl_matrix_transpose_memcpy(gsl_matrix *dest, const gsl_matrix *src) {
  if (dest->size1 != src->size2 || dest->size2 != src->size1)
    NOGSL_ERROR("dimensions of dest must be transpose of src", GSL_EBADLEN);
  for (size_t i = 0; i < dest->size1; ++i)
    for (size_t j = 0; j < dest->size2; ++j) MAT(dest, i, j) = MAT(src, j, i);
  return GSL_SUCCESS;
}

int gsl_matrix_set_row(gsl_matrix *m, size_t i, const gsl_vector *v) {
  if (i >= m->size1) NOGSL_ERROR("row index out of range", GSL_EINVAL);
  if (v->size != m->size2) NOGSL_ERROR("matrix row size and vector length are not equal", GSL_EBADLEN);
  for (size_t j = 0; j < m->size2; ++j) MAT(m, i, j) = VEC(v, j);
  return GSL_SUCCESS;
}

int gsl_permute_vector_inverse(const gsl_permutation *p, gsl_vector *v) {
  size_t n = p->size;
  if (v->size != n) NOGSL_ERROR("vector and permutation must be the same length", GSL_EBADLEN);
  double stack[256];
  double *copy = n <= 256 ? stack : (double *)malloc(n * sizeof(double));
  if (copy == NULL) NOGSL_ERROR("out of memory", GSL_FAILURE);
  for (size_t i = 0; i < n; ++i) copy[i] = VEC(v, i);
  for (size_t i = 0; i < n; ++i) VEC(v, p->data[i]) = copy[i];
  if (copy != stack) free(copy);
  return GSL_SUCCESS;
}

/* ---------------------------------------------------------------- BLAS */
/* The reference BLAS DNRM2's scaled sum of squares over x[offset], x[offset + stride], ... (n elements). */
static double nrm2(const double *x, size_t n, size_t stride) {
  double scale = 0.0, ssq = 1.0;
  for (size_t i = 0; i < n; ++i) {
    double xi = x[i * stride];
    if (xi != 0.0) {
      double a = fabs(xi);
      if (scale < a) {
        double r = scale / a;
        ssq = 1.0 + ssq * r * r;
        scale = a;
      } else {
        double r = a / scale;
        ssq += r * r;
      }
    }
  }
  return scale * sqrt(ssq);
}

int gsl_blas_ddot(const gsl_vector *x, const gsl_vector *y, double *result) {
  if (x->size != y->size) NOGSL_ERROR("invalid length", GSL_EBADLEN);
  double s = 0.0;
  for (size_t i = 0; i < x->size; ++i) s += VEC(x, i) * VEC(y, i);
  *result = s;
  return GSL_SUCCESS;
}

int gsl_blas_daxpy(double alpha, const gsl_vector *x, gsl_vector *y) {
  if (x->size != y->size) NOGSL_ERROR("invalid length", GSL_EBADLEN);
  for (size_t i = 0; i < x->size; ++i) VEC(y, i) += alpha * VEC(x, i);
  return GSL_SUCCESS;
}

/* op(A)(i, j) for op = A or A^T. */
static double opa(const gsl_matrix *a, int trans, size_t i, size_t j) { return trans ? MAT(a, j, i) : MAT(a, i, j); }

int gsl_blas_dtrsm(CBLAS_SIDE_t side, CBLAS_UPLO_t uplo, CBLAS_TRANSPOSE_t trans_a, CBLAS_DIAG_t diag, double alpha,
                   const gsl_matrix *a, gsl_matrix *b) {
  if (a->size1 != a->size2) NOGSL_ERROR("matrix must be square", GSL_ENOTSQR);
  if ((side == CblasLeft && a->size1 != b->size1) || (side == CblasRight && a->size1 != b->size2))
    NOGSL_ERROR("invalid length", GSL_EBADLEN);
  const size_t n = a->size1;
  const int trans = trans_a != CblasNoTrans;
  const int unit = diag == CblasUnit;
  /* op(A) is upper triangular when (upper and not transposed) or (lower and transposed). */
  const int upper = (uplo == CblasUpper) != trans;
  if (alpha != 1.0)
    for (size_t i = 0; i < b->size1; ++i)
      for (size_t j = 0; j < b->size2; ++j) MAT(b, i, j) *= alpha;
  if (n == 0) return GSL_SUCCESS;
  if (side == CblasLeft) {
    /* op(A) X = B, column by column of B. */
    for (size_t c = 0; c < b->size2; ++c) {
      if (upper) {
        for (size_t ii = n; ii-- > 0;) {
          double s = MAT(b, ii, c);
          for (size_t k = ii + 1; k < n; ++k) s -= opa(a, trans, ii, k) * MAT(b, k, c);
          MAT(b, ii, c) = unit ? s : s / opa(a, trans, ii, ii);
        }
      } else {
        for (size_t ii = 0; ii < n; ++ii) {
          double s = MAT(b, ii, c);
          for (size_t k = 0; k < ii; ++k) s -= opa(a, trans, ii, k) * MAT(b, k, c);
          MAT(b, ii, c) = unit ? s : s / opa(a, trans, ii, ii);
        }
      }
    }
  } else {
    /* X op(A) = B, row by row of B: x_j op(A)(j, j) = b_j - sum over k != j of x_k op(A)(k, j). */
    for (size_t r = 0; r < b->size1; ++r) {
      if (upper) {
        for (size_t jj = 0; jj < n; ++jj) {
          double s = MAT(b, r, jj);
          for (size_t k = 0; k < jj; ++k) s -= MAT(b, r, k) * opa(a, trans, k, jj);
          MAT(b, r, jj) = unit ? s : s / opa(a, trans, jj, jj);
        }
      } else {
        for (size_t jj = n; jj-- > 0;) {
          double s = MAT(b, r, jj);
          for (size_t k = jj + 1; k < n; ++k) s -= MAT(b, r, k) * opa(a, trans, k, jj);
          MAT(b, r, jj) = unit ? s : s / opa(a, trans, jj, jj);
        }
      }
    }
  }
  return GSL_SUCCESS;
}

/* ---------------------------------------------------------------- Householder reflectors */
/* H = I - tau u u^T with u = (1, x[1..n-1]) after the call; H x_in = (beta, 0, ..., 0), x[0] = beta. */
static double householder(double *x, size_t n, size_t stride) {
  if (n < 2) return 0.0;
  double xnorm = nrm2(x + stride, n - 1, stride);
  if (xnorm == 0.0) return 0.0;
  double alpha = x[0];
  double norm = hypot(alpha, xnorm);
  double beta = alpha >= 0.0 ? -norm : norm;
  double tau = (beta - alpha) / beta;
  double scale = 1.0 / (alpha - beta);
  for (size_t i = 1; i < n; ++i) x[i * stride] *= scale;
  x[0] = beta;
  return tau;
}

double gsl_linalg_householder_transform(gsl_vector *v) { return householder(v->data, v->size, v->stride); }

/* Applies H = I - tau u u^T (u = (1, a(r0+1..m-1, col)) stored in column `col` below row r0) from the left to the
 * columns c0..c1-1 of `target`, rows r0..m-1. */
static void apply_householder(const gsl_matrix *qr, size_t r0, size_t col, double tau, gsl_matrix *target, size_t c0,
                              size_t c1) {
  if (tau == 0.0) return;
  const size_t m = qr->size1;
  for (size_t j = c0; j < c1; ++j) {
    double w = MAT(target, r0, j);
    for (size_t r = r0 + 1; r < m; ++r) w += MAT(qr, r, col) * MAT(target, r, j);
    w *= tau;
    MAT(target, r0, j) -= w;
    for (size_t r = r0 + 1; r < m; ++r) MAT(target, r, j) -= w * MAT(qr, r, col);
  }
}

int gsl_linalg_QR_decomp(gsl_matrix *a, gsl_vector *tau) {
  const size_t m = a->size1, n = a->size2, k = m < n ? m : n;
  if (tau->size != k) NOGSL_ERROR("size of tau must be MIN(M,N)", GSL_EBADLEN);
  for (size_t i = 0; i < k; ++i) {
    double t = householder(&MAT(a, i, i), m - i, a->tda);
    VEC(tau, i) = t;
    if (i + 1 < n) apply_householder(a, i, i, t, a, i + 1, n);
  }
  return GSL_SUCCESS;
}

int gsl_linalg_QR_unpack(const gsl_matrix *qr, const gsl_vector *tau, gsl_matrix *q, gsl_matrix *r) {
  const size_t m = qr->size1, n = qr->size2, k = m < n ? m : n;
  if (q->size1 != m || q->size2 != m) NOGSL_ERROR("Q matrix must be M x M", GSL_ENOTSQR);
  if (r->size1 != m || r->size2 != n) NOGSL_ERROR("R matrix must be M x N", GSL_ENOTSQR);
  if (tau->size != k) NOGSL_ERROR("size of tau must be MIN(M,N)", GSL_EBADLEN);
  /* Q = H_0 H_1 ... H_{k-1} I, accumulated from the last reflector. */
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < m; ++j) MAT(q, i, j) = i == j ? 1.0 : 0.0;
  for (size_t i = k; i-- > 0;) apply_householder(qr, i, i, VEC(tau, i), q, i, m);
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) MAT(r, i, j) = j >= i ? MAT(qr, i, j) : 0.0;
  return GSL_SUCCESS;
}

int gsl_linalg_QR_QTmat(const gsl_matrix *qr, const gsl_vector *tau, gsl_matrix *a) {
  const size_t m = qr->size1, n = qr->size2, k = m < n ? m : n;
  if (tau->size != k) NOGSL_ERROR("size of tau must be MIN(M,N)", GSL_EBADLEN);
  if (a->size1 != m) NOGSL_ERROR("matrix must have M rows", GSL_EBADLEN);
  /* Q^T A = H_{k-1} ... H_1 H_0 A. */
  for (size_t i = 0; i < k; ++i) apply_householder(qr, i, i, VEC(tau, i), a, 0, a->size2);
  return GSL_SUCCESS;
}

static void swap_columns(gsl_matrix *a, size_t i, size_t j) {
  for (size_t r = 0; r < a->size1; ++r) {
    double t = MAT(a, r, i);
    MAT(a, r, i) = MAT(a, r, j);
    MAT(a, r, j) = t;
  }
}

int gsl_linalg_QRPT_decomp(gsl_matrix *a, gsl_vector *tau, gsl_permutation *p, int *signum, gsl_vector *norm) {
  const size_t m = a->size1, n = a->size2, k = m < n ? m : n;
  if (tau->size != k) NOGSL_ERROR("size of tau must be MIN(M,N)", GSL_EBADLEN);
  if (p->size != n) NOGSL_ERROR("permutation size must be N", GSL_EBADLEN);
  if (norm->size != n) NOGSL_ERROR("norm size must be N", GSL_EBADLEN);
  double stack[256];
  double *orig = n <= 256 ? stack : (double *)malloc(n * sizeof(double));
  if (orig == NULL) NOGSL_ERROR("out of memory", GSL_FAILURE);
  *signum = 1;
  for (size_t j = 0; j < n; ++j) {
    p->data[j] = j;
    VEC(norm, j) = nrm2(&MAT(a, 0, j), m, a->tda);
    orig[j] = VEC(norm, j);
  }
  for (size_t i = 0; i < k; ++i) {
    /* Pivot: the first column of largest remaining norm (as LAPACK's IDAMAX). */
    size_t pvt = i;
    double best = VEC(norm, i);
    for (size_t j = i + 1; j < n; ++j) {
      if (VEC(norm, j) > best) {
        best = VEC(norm, j);
        pvt = j;
      }
    }
    if (pvt != i) {
      swap_columns(a, i, pvt);
      double t = VEC(norm, i);
      VEC(norm, i) = VEC(norm, pvt);
      VEC(norm, pvt) = t;
      t = orig[i];
      orig[i] = orig[pvt];
      orig[pvt] = t;
      size_t s = p->data[i];
      p->data[i] = p->data[pvt];
      p->data[pvt] = s;
      *signum = -*signum;
    }
    double t = householder(&MAT(a, i, i), m - i, a->tda);
    VEC(tau, i) = t;
    if (i + 1 < n) apply_householder(a, i, i, t, a, i + 1, n);
    /* Downdate the remaining column norms (DGEQPF), recomputing them when cancellation makes the update unreliable. */
    for (size_t j = i + 1; j < n; ++j) {
      double nj = VEC(norm, j);
      if (nj == 0.0) continue;
      double r = fabs(MAT(a, i, j)) / nj;
      double temp = 1.0 - r * r;
      if (temp < 0.0) temp = 0.0;
      double ratio = nj / orig[j];
      double temp2 = 1.0 + 0.05 * temp * ratio * ratio;
      if (temp2 == 1.0) {
        if (i + 1 < m) {
          VEC(norm, j) = nrm2(&MAT(a, i + 1, j), m - i - 1, a->tda);
        } else {
          VEC(norm, j) = 0.0;
        }
        orig[j] = VEC(norm, j);
      } else {
        VEC(norm, j) = nj * sqrt(temp);
      }
    }
  }
  if (orig != stack) free(orig);
  return GSL_SUCCESS;
}

/* ---------------------------------------------------------------- SVD (one-sided Jacobi) */
int gsl_linalg_SV_decomp(gsl_matrix *a, gsl_matrix *v, gsl_vector *s, gsl_vector *work) {
  const size_t m = a->size1, n = a->size2;
  if (m < n) NOGSL_ERROR("svd of MxN matrix, M<N, is not implemented", GSL_EBADLEN);
  if (v->size1 != n || v->size2 != n) NOGSL_ERROR("square matrix V must match second dimension of matrix A",
                                                  GSL_EBADLEN);
  if (s->size != n) NOGSL_ERROR("length of vector S must match second dimension of matrix A", GSL_EBADLEN);
  if (work->size != n) NOGSL_ERROR("length of workspace must match second dimension of matrix A", GSL_EBADLEN);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) MAT(v, i, j) = i == j ? 1.0 : 0.0;
  const double eps = 2.220446049250313e-16;
  const int max_sweeps = 100;
  /* A column whose norm is below eps |A|_F is numerically zero: rotating against it only stirs rounding noise (its
   * cosine with any column stays O(1)), so such pairs are skipped. */
  double fro2 = 0.0;
  for (size_t r = 0; r < m; ++r)
    for (size_t j = 0; j < n; ++j) fro2 += MAT(a, r, j) * MAT(a, r, j);
  const double negligible = eps * eps * fro2;
  const double tol = sqrt((double)m) * eps;
  int converged = n < 2;
  for (int sweep = 0; sweep < max_sweeps && !converged; ++sweep) {
    size_t rotations = 0;
    for (size_t j = 0; j + 1 < n; ++j) {
      for (size_t k = j + 1; k < n; ++k) {
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (size_t r = 0; r < m; ++r) {
          double uj = MAT(a, r, j), uk = MAT(a, r, k);
          alpha += uj * uj;
          beta += uk * uk;
          gamma += uj * uk;
        }
        /* Converged pair: the cosine below sqrt(m) eps, LAPACK DGESVJ's tolerance (a bound of eps alone sits under the
         * rounding noise of an m-term dot product and never settles). */
        if (gamma == 0.0 || alpha <= negligible || beta <= negligible || fabs(gamma) <= tol * sqrt(alpha * beta))
          continue;
        ++rotations;
        double zeta = (beta - alpha) / (2.0 * gamma);
        double t = (zeta >= 0.0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
        double c = 1.0 / sqrt(1.0 + t * t);
        double sn = c * t;
        for (size_t r = 0; r < m; ++r) {
          double uj = MAT(a, r, j), uk = MAT(a, r, k);
          MAT(a, r, j) = c * uj - sn * uk;
          MAT(a, r, k) = sn * uj + c * uk;
        }
        for (size_t r = 0; r < n; ++r) {
          double vj = MAT(v, r, j), vk = MAT(v, r, k);
          MAT(v, r, j) = c * vj - sn * vk;
          MAT(v, r, k) = sn * vj + c * vk;
        }
      }
    }
    if (rotations == 0) converged = 1;
  }
  for (size_t j = 0; j < n; ++j) {
    double sj = nrm2(&MAT(a, 0, j), m, a->tda);
    VEC(s, j) = sj;
    if (sj > 0.0)
      for (size_t r = 0; r < m; ++r) MAT(a, r, j) /= sj;
  }
  /* Sort by decreasing singular value (selection sort; columns of U and V move with them). */
  for (size_t i = 0; i < n; ++i) {
    size_t best = i;
    for (size_t j = i + 1; j < n; ++j)
      if (VEC(s, j) > VEC(s, best)) best = j;
    if (best != i) {
      double t = VEC(s, i);
      VEC(s, i) = VEC(s, best);
      VEC(s, best) = t;
      swap_columns(a, i, best);
      swap_columns(v, i, best);
    }
  }
  for (size_t j = 0; j < n; ++j) VEC(work, j) = 0.0;
  if (!converged) NOGSL_ERROR("Jacobi SVD did not converge", GSL_EMAXITER);
  return GSL_SUCCESS;
}
