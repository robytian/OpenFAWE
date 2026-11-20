/* OF-0013 N3: records the matrices the Makani controller passes to MatVecLeftDivide, MatVecRightDivide and MatQrDecomp
 * (the only GSL-backed calls it makes, from common/c_math/optim.c) while a flight runs. Loaded with LD_PRELOAD; the
 * library's calls go through its PLT, so these definitions are found first; each logs its inputs as one JSON line to
 * $NOGSL_CAPTURE (at most $NOGSL_CAPTURE_MAX lines, default 20000) and calls the library's own function, found in the
 * library $NOGSL_LIBRARY itself (ctypes loads it RTLD_LOCAL, so RTLD_NEXT would not see it). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "common/c_math/linalg.h"

static FILE *out;

static void *own(const char *name) {
  static void *handle;
  if (!handle) handle = dlopen(getenv("NOGSL_LIBRARY"), RTLD_NOLOAD | RTLD_LAZY);
  void *f = handle ? dlsym(handle, name) : NULL;
  if (!f) {
    fprintf(stderr, "capture: %s not found in $NOGSL_LIBRARY\n", name);
    abort();
  }
  return f;
}
static long written, limit = -1;

static int open_log(void) {
  if (limit < 0) {
    const char *path = getenv("NOGSL_CAPTURE");
    const char *max = getenv("NOGSL_CAPTURE_MAX");
    limit = max ? atol(max) : 20000;
    out = path ? fopen(path, "w") : NULL;
  }
  return out != NULL && written < limit;
}

static void log_mat(const char *call, const Mat *A, const double *rhs, int nrhs) {
  if (!open_log()) return;
  fprintf(out, "{\"call\":\"%s\",\"m\":%d,\"n\":%d,\"A\":[", call, A->nr, A->nc);
  for (int i = 0; i < A->nr * A->nc; ++i) fprintf(out, "%s%.17g", i ? "," : "", A->d[i]);
  fprintf(out, "],\"rhs\":[");
  for (int i = 0; i < nrhs; ++i) fprintf(out, "%s%.17g", i ? "," : "", rhs[i]);
  fprintf(out, "]}\n");
  if (++written == limit) fflush(out);
}

int32_t MatVecLeftDivide(const Mat *A, const Vec *b, Vec *x) {
  static int32_t (*real)(const Mat *, const Vec *, Vec *);
  if (!real) real = (int32_t (*)(const Mat *, const Vec *, Vec *))own("MatVecLeftDivide");
  log_mat("left", A, b->d, b->length);
  return real(A, b, x);
}

int32_t MatVecRightDivide(const Mat *A, const Vec *b, Vec *x) {
  static int32_t (*real)(const Mat *, const Vec *, Vec *);
  if (!real) real = (int32_t (*)(const Mat *, const Vec *, Vec *))own("MatVecRightDivide");
  log_mat("right", A, b->d, b->length);
  return real(A, b, x);
}

void MatQrDecomp(const Mat *A, Mat *Q, Mat *R) {
  static void (*real)(const Mat *, Mat *, Mat *);
  if (!real) real = (void (*)(const Mat *, Mat *, Mat *))own("MatQrDecomp");
  log_mat("qr", A, NULL, 0);
  real(A, Q, R);
}

__attribute__((destructor)) static void close_log(void) {
  if (out) fclose(out);
}
