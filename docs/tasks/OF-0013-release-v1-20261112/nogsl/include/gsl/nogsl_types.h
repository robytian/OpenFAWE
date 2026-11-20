/* OpenFAWE no-GSL subset (OF-0013 revision 1, section 3): the types of the GSL interface that Makani's common/c_math
 * uses, declared from GSL's public documentation (the interface only). Written for OpenFAWE; no GSL source used. */
#ifndef OPENFAWE_NOGSL_TYPES_H_
#define OPENFAWE_NOGSL_TYPES_H_

#include <stddef.h>

typedef struct {
  size_t size;
  double *data;
} gsl_block;

typedef struct {
  size_t size;
  size_t stride;
  double *data;
  gsl_block *block;
  int owner;
} gsl_vector;

typedef struct {
  gsl_vector vector;
} _gsl_vector_view;
typedef _gsl_vector_view gsl_vector_view;

typedef struct {
  gsl_vector vector;
} _gsl_vector_const_view;
typedef const _gsl_vector_const_view gsl_vector_const_view;

typedef struct {
  size_t size1;
  size_t size2;
  size_t tda;
  double *data;
  gsl_block *block;
  int owner;
} gsl_matrix;

typedef struct {
  gsl_matrix matrix;
} _gsl_matrix_view;
typedef _gsl_matrix_view gsl_matrix_view;

typedef struct {
  gsl_matrix matrix;
} _gsl_matrix_const_view;
typedef const _gsl_matrix_const_view gsl_matrix_const_view;

typedef struct {
  size_t size;
  size_t *data;
} gsl_permutation;

#endif
