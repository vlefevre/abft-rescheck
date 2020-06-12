#include "solve.h"

int main(int argc, char ** argv)
{
	int i, j, n;
	double *A, *A2, *B, *B2, *C0, *C1, *C2, *C2tmp;
	double *x1, *x2;
	double *CK, *xCK;
	double *xc, *yc, *zc;
	double res1, res2, tmp;

	int row_count, col_count, nb_error;
	int *row_loc, *col_loc;
//	int *ix, *jx;
	double error_rate;
	double tol;

	int iit, nb_iter, reexec, maxreexec, nb_fail=0;

	double elapsed,initTime,checkTime,dgemmTime,cleanTime,solveTime;
	struct timeval tp;
	
	gettimeofday(&tp, NULL);
	srand((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

    	n = 4;
	char nstring[10] = "4";
	nb_error = 2;
	tol = 1e-10;
	nb_iter = 1;
	error_rate = 1e-9;
	char error_ratestring[100] = "1e-9";

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			strcpy(nstring, *(argv + i + 1));
			i++;
		}
		if( strcmp( *(argv + i), "-k") == 0) {
			nb_error = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-x") == 0) {
			maxreexec = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-i") == 0) {
			nb_iter = atoi( *(argv + i + 1) );
			i++;
		}
		if ( strcmp( *(argv + i), "-r") == 0) {
			error_rate = atof(*(argv + i + 1));
			strcpy(error_ratestring, *(argv + i + 1));
			i++;
		}
	}

	char filename[1000] = "failures/dgemm_abft_rerun_";
	strcat(filename,nstring);
	strcat(filename,"_");
	strcat(filename,error_ratestring);
	FILE* failures = fopen(filename, "w");

	A = (double *) malloc( n * n * sizeof(double));
	B = (double *) malloc( n * n * sizeof(double));
	C0 = (double *) malloc( n * n * sizeof(double));

	C1 = (double *) malloc( n * n * sizeof(double));

	//Initialize A,B,C random
 	for(i = 0; i < n * n; i++)
		A[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

 	for(i = 0; i < n * n; i++)
		B[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

 	for(i = 0; i < n * n; i++)
		C0[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

	
//////////////////////////
/*	OLD STUFF
	//Find coordinates of errors
//	ix goes from 0 to n-1, jx goes from 0 to n-1,
	ix = (int *) malloc( nb_error * sizeof(int));
	jx = (int *) malloc( nb_error * sizeof(int));

	for(i = 0; i < nb_error ; i++) { ix[i] = ( rand() % n ); jx[i] = ( rand() % n ); }

	if (DEBUG) {
	printf("nb_error  = %3d: error_loc = [", nb_error);
	for(i = 0; i < nb_error ; i++) printf(" (%3d,%3d) ", ix[i], jx[i]);
	printf("];\n");
	}*/
////////////////////////// CALL TO GEMM WITHOUT FAILURES

 	for(i = 0; i < n * n; i++)
		C1[i] = C0[i];

//	reference call to GEMM	
//	compute C1 = C0 - A * B

	gettimeofday(&tp, NULL);
	elapsed=-((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C1, n);
	gettimeofday(&tp, NULL);
	elapsed+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

//	check quality with residual checking
	xc = (double *) malloc( n * sizeof(double));
	yc = (double *) malloc( n * sizeof(double));
	zc = (double *) malloc( n * sizeof(double));

	for(i = 0; i < n; i++)
		xc[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

	// zc = C0 * xc
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, C0, n, xc, 1, 0.0e+00, zc, 1 );
	// yc = B * xc
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, B, n, xc, 1, 0.0e+00, yc, 1 );
	// zc = zc - A * yc
	// i.e. zc = C0 * xc - A * B * xc
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, -1.0e+00, A, n, yc, 1, 1.0e+00, zc, 1 );

	// yc = zc
	cblas_dcopy( n, zc, 1, yc, 1 );
	// yc = C1 * xc - yc
	// i.e. yc = C1 * xc - (C0*xc - A*B*xc) = ((C0 - A*B) - (C0 - A*B))*xc (SHOULD BE ~0)
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, C1, n, xc, 1, -1.0e+00, yc, 1 );
	res1 = 0.e+00; for(i = 0; i < n ; i++) { tmp = fabs(yc[i]); if ( res1 < tmp ){ res1 = tmp; } }

	//printf("Call to GEMM:\nn = %3d\telapsed = %7.4f (sec)\t%e\n", n, elapsed, res1);
	//printf("%7.4f %e ",elapsed,res1);

//	reference call to GEMM, followed by residual checking and "naive" recovery

////////////////////////// CALL TO GEMM + RES CHECK NAIVE
	C2 = (double*) malloc( n * n * sizeof(double));
	elapsed = 0;
	for (iit = 0; iit < nb_iter; iit++)
	{
	
	reexec = 0;
	fprintf(failures, "%d ", iit);
	gettimeofday(&tp, NULL);
	elapsed-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

//	we assume that nbmax_error is less than n
//	if nbmax_error is greater than n, than we should allocate min( nbmax_error, n )
	gettimeofday(&tp, NULL);
	initTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	#pragma omp parallel for
	for (i = 0; i < n; i++)
		memcpy(C2+i*n,C0+i*n,n*sizeof(double));

	row_loc = (int *) malloc( n * sizeof(int));
	col_loc = (int *) malloc( n * sizeof(int));

	x1 = (double *) malloc( n * sizeof(double));
	x2 = (double *) malloc( n * sizeof(double));

	CK = (double*) malloc(n * sizeof(double));
	xCK = (double*) malloc(n * sizeof(double));

	#pragma omp parallel for
 	for(i = 0; i < n; i++)
	{
		x1[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;
		x2[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;
	}
	
////////////

	//Creating copies

	C2tmp = (double*) malloc((n+1) * (n+1) * sizeof(double)); //Extend C0 by 1 row and 1 col
	A2 = (double*) malloc((n+1) * n * sizeof(double)); //Extend A by 1 row
	B2 = (double*) malloc(n * (n+1) * sizeof(double)); //Extend B by 1 col

 	for(i = 0; i < n; i++)
	{
		memcpy(C2tmp+i*(n+1),C2+i*n,n*sizeof(double));
		memcpy(A2+i*(n+1),A+i*n,n*sizeof(double));
		memcpy(B2+i*n,B+i*n,n*sizeof(double));
	}

	gettimeofday(&tp, NULL);
	initTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	//Compute the initial checksums
	
	gettimeofday(&tp, NULL);
	checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, A2, n+1, x2, 1, 0.0e+00, &(A2[n]), n+1 ); //A2[n] is beginning of n+1-th row
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, B2, n, x1, 1, 0.0e+00, &(B2[n*n]), 1); //B2[n*n] is beginning of n+1-th col
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, C2tmp, n+1, x2, 1, 0.0e+00, &(C2tmp[n]), n+1 ); //C2[n] is beginning of n+1-th row
	cblas_dgemv( CblasColMajor, CblasNoTrans, n+1, n, 1.0e+00, C2tmp, n+1, x1, 1, 0.0e+00, &(C2tmp[n*(n+1)]), 1); //same (but with already computed 1st checksum!)
	gettimeofday(&tp, NULL);
	checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

//	compute C2 = C0 - A * B
	gettimeofday(&tp, NULL);
	dgemmTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n+1, n+1, n, -1.0e+00, A2, n+1, B2, n, +1.0e+00, C2tmp, n+1);
	gettimeofday(&tp, NULL);
	dgemmTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	#pragma omp parallel for collapse(2)
	nb_error = 0;
	for (i = 0; i < n; i++ )
	{
		for (j = 0; j < n; j++ )
		{
			if (hasFailed(2*n-1,error_rate))
			{
				if (DEBUG)
				{
					printf("Cell %d,%d  corrupted.\n",i,j);
					fflush(stdout);
				}
				nb_error++;
				C2tmp[i+j*(n+1)] = C2tmp[i+j*(n+1)] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
			}
		}
	}
	fprintf(failures,"%d ",nb_error);
	/*//modifiying the erroneous elements (should not be part of time computation but should be negligible)
	for(i = 0; i < nb_error ; i++) { 
		C2tmp[ ix[i] + jx[i] * (n+1) ] = C2tmp[ ix[i] + jx[i] * (n+1) ] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 ) ;
	}*/

	check:
	gettimeofday(&tp, NULL);
	checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, C2tmp, n+1, x2, 1, 0.0e+00, xCK, 1); //column checksum
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, C2tmp, n+1, x1, 1, 0.0e+00, CK, 1); //row checksum (no need to checksum the colum checksum)


	res2 = 0.e+00; for(i = 0; i < n ; i++) { tmp = fabs(C2tmp[i+n*(n+1)]-CK[i]); if ( res2 < tmp ){ res2 = tmp; } } //looking at n+1-th col for the checksums

	//Margin for floating-point value rounding
	if( res2 > tol ){

	row_count = 0; 
	for(i = 0; i < n ; i++) { 
		if( fabs(C2tmp[i+n*(n+1)]-CK[i]) > tol ){ //row i is corrupted 
			row_loc[row_count] = i; 
			row_count++;
		}
	}

	if (DEBUG) {
	printf("row_count = %3d: row_loc = [", row_count);
	for(i = 0; i < row_count ; i++) printf(" %3d ", row_loc[i]);
	printf("];\n");
	}
	col_count = 0; 
	for(j = 0; j < n ; j++) { 
		if( fabs(C2tmp[n + j*(n+1)]-xCK[j]) > tol ){ 
			col_loc[col_count] = j; 
			col_count++;
		}
	}

	gettimeofday(&tp, NULL);
	checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	if (DEBUG) {
	printf("col_count = %3d: col_loc = [", col_count);
	for(j = 0; j < col_count ; j++) printf(" %3d ", col_loc[j]);
	printf("];\n");
	}
/*
	for(i = 0; i < row_count ; i++){
		for(j = 0; j < col_count ; j++){
			//Recomputing the erroneous elements from original C0 copy
			C2[ row_loc[i] + col_loc[j] * n ] = C0[ row_loc[i] + col_loc[j] * n ] - cblas_ddot( n, &(A[row_loc[i]]), n, &(B[col_loc[j] * n]), 1 );

		}
	}
*/
	gettimeofday(&tp, NULL);
	solveTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	solveRerun(n,C2tmp,n+1,-1.0e+00,A,n,B,n,1.0e+00,C0,n,row_count,row_loc,col_count,col_loc);
	gettimeofday(&tp, NULL);
	solveTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	//New computation can be corrupted too
	nb_error = 0;
	for (int i=0; i<row_count; i++)
	{
		for (int j=0; j<col_count; j++)
		{
			if (hasFailed(2*n-1,error_rate))
			{
				if (DEBUG)
				{
					printf("Cell %d,%d  corrupted.\n",row_loc[i],col_loc[j]);
					fflush(stdout);
				}
				nb_error++;
				C2tmp[row_loc[i]+col_loc[j]*(n+1)] = C2tmp[row_loc[i]+col_loc[j]*(n+1)] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
			}
		}
	}
	fprintf(failures,"%d ",nb_error);
	reexec++;
	if (reexec <= maxreexec)
	{
		if (DEBUG)
			printf("Moving to check\n");
		goto check;
	} else {
		if (DEBUG)
			printf("Max number of executions reached %d %d\n",reexec, maxreexec);
		if (nb_error > 0)
			nb_fail++;
		fprintf(failures,"\n");
	}

	} else {
		fprintf(failures,"\n");
		gettimeofday(&tp, NULL);
		checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	}

	gettimeofday(&tp, NULL);
	cleanTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	#pragma omp parallel for
	for (j = 0; j < n; j++)
		memcpy(C2+j*n,C2tmp+j*(n+1),n*sizeof(double));

	free(A2);
	free(B2);
	free(C2tmp);

	free( col_loc );
	free( row_loc );

	free(x1);
	free(x2);
	free(xCK);
	free(CK);

	gettimeofday(&tp, NULL);
	cleanTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	gettimeofday(&tp, NULL);
	elapsed+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	}
	cblas_dcopy( n, zc, 1, yc, 1 );
	//Again checking that C2 = C0-A*B by a simple check (C2*xc vs yc)
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, C2, n, xc, 1, -1.0e+00, yc, 1 );
	res2 = 0.e+00; for(i = 0; i < n ; i++) { tmp = fabs(yc[i]); if ( res2 < tmp ){ res2 = tmp; } }

	//printf("Call to GEMM and recovery (recomputing %3d values):\nn = %3d\telapsed = %7.4f (sec)\t%e\n", col_count*row_count, n, elapsed, res2);
	if (nb_fail > 0)
		printf("FAILED %d TIMES",nb_fail);
	else
		printf("%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %e ",elapsed/nb_iter,initTime/nb_iter,dgemmTime/nb_iter,checkTime/nb_iter,solveTime/nb_iter,cleanTime/nb_iter,res2);

////////////////////CLEANING

	/*free(ix);
	free(jx);*/
	
	fclose(failures);

	free(zc);
	free(yc);
	free(xc);

	free(C2);
	free(C1);
	free(C0);
	free(B);
	free(A);

	return 0;

}
