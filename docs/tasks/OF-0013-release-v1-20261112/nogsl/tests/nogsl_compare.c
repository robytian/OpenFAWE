/* OF-0013 criterion N3: one test program, built twice from this source - linked with GSL, and with OpenFAWE's no-GSL
 * subset - running Makani's own common/c_math functions (unchanged) on the same cases. Reads CASES (text), writes one
 * line per case (key value...) with %.17g numbers; nogsl_check.py compares the two outputs.
 *
 * Case line: KIND ID M N K TOL, then M*N numbers (A, row-major), then the right-hand side: left M numbers, right N
 * numbers, matleft M*K numbers. KIND: qr qrpt left right matleft rank svd. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_linalg.h>

#include "common/c_math/linalg.h"

static void put(FILE *out, const char *key, const double *d, int n) {
  fprintf(out, " %s %d", key, n);
  for (int i = 0; i < n; ++i) fprintf(out, " %.17g", d[i]);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: nogsl_compare CASES OUT\n");
    return 2;
  }
#if defined(NOGSL_WITH_REAL_GSL)
  gsl_set_error_handler_off();
#endif
  FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
  if (in == NULL || out == NULL) return 2;
  char kind[16];
  long id;
  int m, n, k;
  double tol;
  while (fscanf(in, "%15s %ld %d %d %d %lf", kind, &id, &m, &n, &k, &tol) == 6) {
    static double a[MAT_MAX_ELEMENTS], rhs[MAT_MAX_ELEMENTS];
    for (int i = 0; i < m * n; ++i)
      if (fscanf(in, "%lf", &a[i]) != 1) return 3;
    int nr = !strcmp(kind, "left") ? m : !strcmp(kind, "right") ? n : !strcmp(kind, "matleft") ? m * k : 0;
    for (int i = 0; i < nr; ++i)
      if (fscanf(in, "%lf", &rhs[i]) != 1) return 3;
    Mat A = {m, n, a, 0, m * n};
    fprintf(out, "%s %ld", kind, id);
    if (!strcmp(kind, "qr")) {
      MAT(m, m, Q);
      MAT(m, n, R);
      MatQrDecomp(&A, &Q, &R);
      put(out, "Q", Q.d, m * m);
      put(out, "R", R.d, m * n);
    } else if (!strcmp(kind, "qrpt")) {
      int kk = m < n ? m : n, signum = 0;
      double qr[MAT_MAX_ELEMENTS], tau[VEC_MAX_ELEMENTS], norm[VEC_MAX_ELEMENTS];
      size_t perm[VEC_MAX_ELEMENTS];
      memcpy(qr, a, sizeof(double) * (size_t)(m * n));
      gsl_matrix_view QR = gsl_matrix_view_array(qr, (size_t)m, (size_t)n);
      gsl_vector_view T = gsl_vector_view_array(tau, (size_t)kk);
      gsl_vector_view N = gsl_vector_view_array(norm, (size_t)n);
      gsl_permutation p = {(size_t)n, perm};
      int ret = gsl_linalg_QRPT_decomp(&QR.matrix, &T.vector, &p, &signum, &N.vector);
      double pd[VEC_MAX_ELEMENTS], rdiag[VEC_MAX_ELEMENTS], st[2] = {(double)ret, (double)signum};
      for (int j = 0; j < n; ++j) pd[j] = (double)perm[j];
      for (int j = 0; j < kk; ++j) rdiag[j] = qr[j * n + j];
      put(out, "status_signum", st, 2);
      put(out, "perm", pd, n);
      put(out, "rdiag", rdiag, kk);
      put(out, "QR", qr, m * n);
    } else if (!strcmp(kind, "left")) {
      Vec b = {m, rhs, 0, m};
      VEC(n, x);
      double ret = (double)MatVecLeftDivide(&A, &b, &x);
      put(out, "ret", &ret, 1);
      put(out, "x", x.d, n);
    } else if (!strcmp(kind, "right")) {
      Vec b = {n, rhs, 0, n};
      VEC(m, x);
      double ret = (double)MatVecRightDivide(&A, &b, &x);
      put(out, "ret", &ret, 1);
      put(out, "x", x.d, m);
    } else if (!strcmp(kind, "matleft")) {
      Mat B = {m, k, rhs, 0, m * k};
      MAT(n, k, X);
      double ret = (double)MatMatLeftDivide(&A, &B, &X);
      put(out, "ret", &ret, 1);
      put(out, "X", X.d, n * k);
    } else if (!strcmp(kind, "rank")) {
      double r = (double)MatRank(&A, tol);
      put(out, "rank", &r, 1);
    } else if (!strcmp(kind, "svd")) {
      MAT(m, n, U);
      VEC(n, s);
      MAT(n, n, V);
      double ret = (double)MatThinSvDecomp(&A, &U, &s, &V);
      put(out, "ret", &ret, 1);
      put(out, "U", U.d, m * n);
      put(out, "s", s.d, n);
      put(out, "V", V.d, n * n);
    }
    fprintf(out, "\n");
    fflush(out);
  }
  fclose(out);
  return 0;
}
