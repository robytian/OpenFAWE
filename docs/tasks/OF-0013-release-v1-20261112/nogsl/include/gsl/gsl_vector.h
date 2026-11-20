/* OpenFAWE no-GSL subset: vectors and views. */
#ifndef OPENFAWE_NOGSL_VECTOR_H_
#define OPENFAWE_NOGSL_VECTOR_H_

#include "nogsl_types.h"

gsl_vector_view gsl_vector_view_array(double *base, size_t n);
double gsl_vector_get(const gsl_vector *v, size_t i);
void gsl_vector_set(gsl_vector *v, size_t i, double x);
int gsl_vector_swap_elements(gsl_vector *v, size_t i, size_t j);

#endif
