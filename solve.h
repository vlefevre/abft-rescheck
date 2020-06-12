#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "cblas.h"
#include "lapacke.h"

#define DEBUG 1

void printMatrix(int m, int n, double* M);
void solveRerun(int k, double* M, int ldm, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc, int row_count, int* row_loc, int col_count, int* col_loc);
void solveSystem(double* M, int ldm, double* OCK, int ldock, double* CK, int ldck, double* CM, int ldcm, int row_count, int* row_loc, int col_count, int* col_loc);
int hasFailed(int nbOp, double rate);
