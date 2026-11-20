/* OpenFAWE no-GSL subset: the CBLAS enumerations (values of the CBLAS standard, netlib cblas.h). */
#ifndef OPENFAWE_NOGSL_CBLAS_H_
#define OPENFAWE_NOGSL_CBLAS_H_

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };

#endif
