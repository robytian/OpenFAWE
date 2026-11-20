/* OpenFAWE no-GSL subset: the decompositions Makani uses (Householder QR, QR with column pivoting, SVD). */
#ifndef OPENFAWE_NOGSL_LINALG_H_
#define OPENFAWE_NOGSL_LINALG_H_

#include "gsl_matrix.h"
#include "gsl_permute_vector.h"
#include "gsl_vector.h"
#include "nogsl_types.h"

double gsl_linalg_householder_transform(gsl_vector *v);
int gsl_linalg_QR_decomp(gsl_matrix *a, gsl_vector *tau);
int gsl_linalg_QR_unpack(const gsl_matrix *qr, const gsl_vector *tau, gsl_matrix *q, gsl_matrix *r);
int gsl_linalg_QR_QTmat(const gsl_matrix *qr, const gsl_vector *tau, gsl_matrix *a);
int gsl_linalg_QRPT_decomp(gsl_matrix *a, gsl_vector *tau, gsl_permutation *p, int *signum, gsl_vector *norm);
int gsl_linalg_SV_decomp(gsl_matrix *a, gsl_matrix *v, gsl_vector *s, gsl_vector *work);

#endif
