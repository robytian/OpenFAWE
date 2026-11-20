/* OpenFAWE no-GSL subset: the status codes of the GSL interface (documented values) and the error hook. */
#ifndef OPENFAWE_NOGSL_ERRNO_H_
#define OPENFAWE_NOGSL_ERRNO_H_

enum {
  GSL_SUCCESS = 0,
  GSL_FAILURE = -1,
  GSL_EDOM = 1,
  GSL_EINVAL = 4,
  GSL_EMAXITER = 11,
  GSL_EBADLEN = 19,
  GSL_ENOTSQR = 20,
  GSL_ESING = 21
};

/* Called on an error before the status is returned, as the GSL interface does; Makani defines its own on ARM. */
void gsl_error(const char *reason, const char *file, int line, int gsl_errno);

#endif
