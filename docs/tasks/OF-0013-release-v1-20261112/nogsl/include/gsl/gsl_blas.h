/* OpenFAWE no-GSL subset: the level 1 and level 3 BLAS operations Makani uses. */
#ifndef OPENFAWE_NOGSL_BLAS_H_
#define OPENFAWE_NOGSL_BLAS_H_

#include "gsl_blas_types.h"
#include "gsl_matrix.h"
#include "gsl_vector.h"

int gsl_blas_ddot(const gsl_vector *x, const gsl_vector *y, double *result);
int gsl_blas_daxpy(double alpha, const gsl_vector *x, gsl_vector *y);
int gsl_blas_dtrsm(CBLAS_SIDE_t side, CBLAS_UPLO_t uplo, CBLAS_TRANSPOSE_t trans_a, CBLAS_DIAG_t diag, double alpha,
                   const gsl_matrix *a, gsl_matrix *b);

#endif
