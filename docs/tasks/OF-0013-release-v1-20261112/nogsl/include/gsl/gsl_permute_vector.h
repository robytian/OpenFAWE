/* OpenFAWE no-GSL subset: permutations of vectors. */
#ifndef OPENFAWE_NOGSL_PERMUTE_VECTOR_H_
#define OPENFAWE_NOGSL_PERMUTE_VECTOR_H_

#include "nogsl_types.h"

/* v'[p[i]] = v[i]: the inverse of the permutation p applied to v. */
int gsl_permute_vector_inverse(const gsl_permutation *p, gsl_vector *v);

#endif
