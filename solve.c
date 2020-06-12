#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "cblas.h"
#include "solve.h"

void printMatrix(int m, int n, double* M)
//print a matrix M of size mxn
{
	int i,j;
	for (i=0; i<m; i++)
	{
		for (j=0; j<n; j++)
			printf("%3f ", M[i+j*m]);
		printf("\n");
	}
}

void solveRerun(int k, double* M, int ldm, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc, int row_count, int* row_loc, int col_count, int* col_loc)
/*
int k: size of scalar product
double* M: matrix to be corrected (ColMajor)
int ldm: leading dimension of M
double alpha: scale factor for A*B
double* A: left matrix of product (ColMajor)
int lda: leading dimension of A
double* B: right matrix of product (ColMajor)
int ldb: leading dimension of B
double beta: scale factor of C
double* C: original matrix (ColMajor)
int ldc: leading dimension of C
int row_count: number of erroneous rows
int* row_loc: indexes of erroneous rows
int col_count: number of erroneous columns
int* col_loc: indexes of erroneous columns
*/
{
	if (DEBUG)
		printf("Solving...\n");
//	setenv("MKL_NUM_THREADS","1",1);
	//run in k * row_count * col_count
	int i,j;
	#pragma omp parallel for collapse(2)
	for(i = 0; i < row_count ; i++){
		for(j = 0; j < col_count ; j++){
			//Recomputing the erroneous elements from original C0 copy
			M[ row_loc[i] + col_loc[j] * ldm ] = beta * C[ row_loc[i] + col_loc[j] * ldc ] + alpha * cblas_ddot( k, &(A[row_loc[i]]), lda, &(B[col_loc[j] * ldb]), 1 );
		}
	}
//	setenv("MKL_NUM_THREADS",getenv("OMP_NUM_THREADS"),1);
	if (DEBUG)
		printf("Solved.\n");
}

void solveSystem(double* M, int ldm, double* OCK, int ldock, double* CK, int ldck, double* CM, int ldcm, int row_count, int* row_loc, int col_count, int* col_loc)
/*
double* M: matrix to be corrected (ColMajor)
int ldm: leading dimension of M
double* OCK: original checksum (ColMajor)
int ldock: leading dimension of OCK
double* CK: current checksum (ColMajor)
int ldck: leading dimension of CK
double* CM: checksum matrix (ColMajor)
int ldcm: leading dimension of CM
int row_count: number of erroneous rows
int* row_loc: indexes of erroneous rows
int col_count: number of erroneous columns
int* col_loc: indexes of erroneous columns
*/
{
	int i,j,k;
	double *RHS, *syst;
	int *piv;

	RHS = (double*) malloc(col_count * sizeof(double));
	syst = (double*) malloc(col_count * col_count * sizeof(double));
	piv = (int*) malloc(col_count * sizeof(int));
	for (k = 0; k < row_count; k++)
	{
		//Create a system of size col_count to solve
		#pragma omp parallel for
		for (i = 0; i < col_count; i++)
		{
			// select first col_count checksums
			RHS[i] = OCK[row_loc[k] + i * ldock] - CK[row_loc[k] + i * ldck];
			for (j = 0; j < col_count; j++)
			{
				RHS[i] += M[row_loc[k] + col_loc[j] * ldm] * CM[col_loc[j] + i * ldcm];
				syst[i + j * col_count] = CM[col_loc[j] + i * ldcm];
			}
		}

		//Solving the system syst * X = RHS
		LAPACKE_dgesv(CblasColMajor, col_count, 1, syst, col_count, piv, RHS, col_count);

		//Correcting the values
		#pragma omp parallel for
		for (i = 0; i < col_count; i++)
			M[row_loc[k] + col_loc[i] * ldm] = RHS[i];

	}

	free(RHS);
	free(syst);
	free(piv);
}

/*
Compute the probability that one cell is erroneous

int nbOp: number of operations done to compute *src
double rate: probability of one operation producing wrong result
*/
int hasFailed(int nbOp, double rate)
{
/*	printf("hasFailed: %d %f\n",nbOp,rate);
	fflush(stdout);*/

	double randVal = (double)rand()/(double)RAND_MAX;
	double proba = 1.0- pow(1.0-rate,(double)nbOp);
	return randVal < proba;
}
