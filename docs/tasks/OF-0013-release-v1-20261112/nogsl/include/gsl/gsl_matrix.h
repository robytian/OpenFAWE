/* OpenFAWE no-GSL subset: matrices and views. */
#ifndef OPENFAWE_NOGSL_MATRIX_H_
#define OPENFAWE_NOGSL_MATRIX_H_

#include "gsl_vector.h"
#include "nogsl_types.h"

gsl_matrix_view gsl_matrix_view_array(double *base, size_t n1, size_t n2);
_gsl_matrix_const_view gsl_matrix_const_view_array(const double *base, size_t n1, size_t n2);
gsl_matrix_view gsl_matrix_submatrix(gsl_matrix *m, size_t i, size_t j, size_t n1, size_t n2);
_gsl_matrix_const_view gsl_matrix_const_submatrix(const gsl_matrix *m, size_t i, size_t j, size_t n1, size_t n2);
gsl_vector_view gsl_matrix_row(gsl_matrix *m, size_t i);
gsl_vector_view gsl_matrix_column(gsl_matrix *m, size_t j);
gsl_vector_view gsl_matrix_subrow(gsl_matrix *m, size_t i, size_t offset, size_t n);
gsl_vector_view gsl_matrix_subcolumn(gsl_matrix *m, size_t j, size_t offset, size_t n);
_gsl_vector_const_view gsl_matrix_const_row(const gsl_matrix *m, size_t i);
_gsl_vector_const_view gsl_matrix_const_subrow(const gsl_matrix *m, size_t i, size_t offset, size_t n);
double gsl_matrix_get(const gsl_matrix *m, size_t i, size_t j);
void gsl_matrix_set(gsl_matrix *m, size_t i, size_t j, double x);
void gsl_matrix_set_zero(gsl_matrix *m);
int gsl_matrix_memcpy(gsl_matrix *dest, const gsl_matrix *src);
int gsl_matrix_transpose_memcpy(gsl_matrix *dest, const gsl_matrix *src);
int gsl_matrix_set_row(gsl_matrix *m, size_t i, const gsl_vector *v);

#endif
