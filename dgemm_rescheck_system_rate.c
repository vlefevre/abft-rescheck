#include "solve.h"

int main(int argc, char ** argv) {

	int i, j, k, n;
	double *A, *B, *C0, *C1, *C2;
	double *x1, *x2, *y, *z;
	double *M, *MC, *Mtemp;
	double *xc, *yc, *zc;
	double res1, res2, tmp;

	int row_count, col_count, nb_error;
	int *row_loc, *col_loc;
	//int *ix, *jx;
	double error_rate;
	double tol;

	int nb_iter, iit, reexec, maxreexec, nb_fail=0;

	double elapsed,initTime,dgemmTime,solveTime,checkTime,cleanTime;
	struct timeval tp;
	
	gettimeofday(&tp, NULL);
	srand((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

    	n = 4;
	char nstring[10] = "4";
	nb_error = 3;
	tol = 1e-10;
	nb_iter = 1;
	error_rate = 1e-9;
	maxreexec = 3;
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
		if( strcmp( *(argv + i), "-i") == 0) {
			nb_iter = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-x") == 0) {
			maxreexec = atoi( *(argv + i + 1) );
			i++;
		}
		if ( strcmp( *(argv + i), "-r") == 0) {
			error_rate = atof(*(argv + i + 1));
			strcpy(error_ratestring, *(argv + i + 1));
			i++;
		}
	}

	char filename[1000] = "failures/dgemm_rescheck_system_";
	strcat(filename,nstring);
	strcat(filename,"_");
	strcat(filename,error_ratestring);
	FILE* failures = fopen(filename, "w");

	A = (double *) malloc( n * n * sizeof(double));
	B = (double *) malloc( n * n * sizeof(double));
	C0 = (double *) malloc( n * n * sizeof(double));
	C1 = (double *) malloc( n * n * sizeof(double));
	C2 = (double *) malloc( n * n * sizeof(double));

	//Initialize A,B,C random
 	for(i = 0; i < n * n; i++)
		A[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

 	for(i = 0; i < n * n; i++)
		B[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

 	for(i = 0; i < n * n; i++)
		C0[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

//////////////////////////
/*	OLD STUFF
//	Find coordinates of errors
//	ix goes from 0 to n-1, jx goes from 0 to n-1,
	ix = (int *) malloc( nb_error * sizeof(int));
	jx = (int *) malloc( nb_error * sizeof(int));

	for(i = 0; i < nb_error ; i++) { ix[i] = ( rand() % n ); jx[i] = ( rand() % n ); }

	if (DEBUG)
	{
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

	//printMatrix(n,n,C1);

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

////////////////////////// CALL TO GEMM + RES CHECK RECOVERY O(nberror*(nberror^3 + n))
	elapsed=0;
	initTime=0;
	dgemmTime=0;
	checkTime=0;
	solveTime=0;
	cleanTime=0;
	for (iit = 0; iit < nb_iter; iit++)
	{

	reexec = 0;
	fprintf(failures, "%d ", iit);
	gettimeofday(&tp, NULL);
	elapsed-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

//	we assume that nbmax_error is less than n
	gettimeofday(&tp,NULL);
	initTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
 	
	#pragma omp parallel for
	for(i = 0; i < n; i++)
		memcpy(C2+i*n,C0+i*n,n*sizeof(double));

	row_loc = (int *) malloc( n * sizeof(int));
	col_loc = (int *) malloc( n * sizeof(int));

	//column checksum to locate
	x2 = (double *) malloc( n * sizeof(double));
	y = (double *) malloc( n * sizeof(double));
	z = (double *) malloc( n * sizeof(double));

	#pragma omp parallel for
 	for(i = 0; i < n; i++)
		x2[i] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;
	gettimeofday(&tp,NULL);
	initTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

//	compute C3 = C0 - A * B
	gettimeofday(&tp,NULL);
	dgemmTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C2, n);
	gettimeofday(&tp,NULL);
	dgemmTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	//printMatrix(n,n,C3);

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
				C2[i+j*n] = C2[i+j*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
			}
		}
	}
	fprintf(failures,"%d ",nb_error);
	/* OLD STUFF
	//modifiying the erroneous elements (should not be part of time computation but should be negligible)
	for(i = 0; i < nb_error ; i++) { 
		C2[ ix[i] + jx[i] * n ] = C2[ ix[i] + jx[i] * n ] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 ) ;
	}*/

	//printMatrix(n,n,C3);

//	compute ( x2 * C3 ) - ( ( x2 * C0 ) -  ( x2 * A ) * B )
//	compute ( C3^T * x2 ) - ( ( C0^T * x2 ) - B^T * ( A^T * x2 ) )

	check:
	gettimeofday(&tp,NULL);
	checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, C0, n, x2, 1, 0.0e+00, z, 1 );
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, A, n, x2, 1, 0.0e+00, y, 1 );
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, -1.0e+00, B, n, y, 1, 1.0e+00, z, 1 );
	cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0e+00, C2, n, x2, 1, -1.0e+00, z, 1 );

	res2 = 0.e+00; for(i = 0; i < n ; i++) { tmp = fabs(z[i]); if ( res2 < tmp ){ res2 = tmp; } }

	x1 = NULL;
	MC = NULL;
	Mtemp = NULL;
	M = NULL;
	//Margin for floating-point value rounding
	if( res2 > tol ){

	col_count = 0; 
	for(j = 0; j < n ; j++) { 
		if( fabs(z[j]) > tol ){ 
			col_loc[col_count] = j; 
			col_count++;
		}
	}

	if (DEBUG)
	{
	printf("col_count = %3d: col_loc = [", col_count);
	for(j = 0; j < col_count ; j++) printf(" %3d ", col_loc[j]);
	printf("];\n");
	}

	//Checksum matrix
	if (col_count != 0)
		x1 = (double *) malloc(col_count *  n * sizeof(double)); //nbmax_error row checksums

	#pragma omp parallel for
 	for(i = 0; i < n; i++)
	{
		for (j = 0; j < col_count; j++)
			x1[i + j * n] = (double)rand() / (double)(RAND_MAX) - 0.5e+00;
	}

	if (col_count != 0)
	{
		MC = (double *) malloc(col_count * n * sizeof(double)); //row checksums of computation without errors (will contain C0*x1 - A*(B*x1))
		Mtemp = (double *) malloc(col_count * n * sizeof(double)); //temp to store A*B*x1
		M = (double *) malloc(col_count * n * sizeof(double)); //row checksums of actual computation (will contain C3*x1)
	//	compute M = C3 * x1 and MC = C0 * x1 - A * ( B * x1 )
	//	just as before but C3 is corrupted (not C1)
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, col_count, n, 1.0e+00, C0, n, x1, n, 0.0e+00, MC, n );
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, col_count, n, 1.0e+00, B, n, x1, n, 0.0e+00, Mtemp, n );
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, col_count, n, -1.0e+00, A, n, Mtemp, n, 1.0e+00, MC, n );
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, col_count, n, 1.0e+00, C2, n, x1, n, 0.0e+00, M, n );

		row_count = 0; 
		for(i = 0; i < n ; i++) { 
			if( fabs(M[i]-MC[i]) > tol ){ //row i is corrupted (comparing first column)
		// This part is useless as we can tolerate up to n errors in the same column
		//		if( nbmax_error <= row_count ){
		//			row_count++;
		//			printf("we have detected at least %3d errors, we provision to tolerate at most %3d => abort\n", row_count, nbmax_error ); 
		//			return 0;
		//		}
				row_loc[row_count] = i; 
				row_count++;
			}
		}
		gettimeofday(&tp,NULL);
		checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		if (DEBUG)
		{
		printf("row_count = %3d: row_loc = [", row_count);
		for(i = 0; i < row_count ; i++) printf(" %3d ", row_loc[i]);
		printf("];\n");
		}
		
		gettimeofday(&tp, NULL);
		solveTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		solveSystem(C2,n,MC,n,M,n,x1,n,row_count,row_loc,col_count,col_loc);
		gettimeofday(&tp, NULL);
		solveTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

		//New computation can be corrupted too
		nb_error = 0;
		for (int i=0; i<row_count; i++)
		{
			for (int j=0; j<col_count; j++)
			{
				if (hasFailed(col_count*col_count,error_rate))
				{
					if (DEBUG)
					{
						printf("Cell %d,%d  corrupted.\n",row_loc[i],col_loc[j]);
						fflush(stdout);
					}
					nb_error++;
					C2[row_loc[i]+col_loc[j]*n] = C2[row_loc[i]+col_loc[j]*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
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
	}
	} else {
		fprintf(failures,"\n");
		gettimeofday(&tp, NULL);
		checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	}
	
	gettimeofday(&tp, NULL);
	cleanTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	if (x1 != NULL)
		free(x1);
	free(x2);
	
	free(y);
	free(z);

	if (M != NULL)
		free(M);
	if (MC != NULL)
		free(MC);
	if (Mtemp != NULL)
		free(Mtemp);

	free( col_loc );
	free( row_loc );
	gettimeofday(&tp, NULL);
	cleanTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	gettimeofday(&tp, NULL);
	elapsed+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	}
	//printMatrix(n,n,C2);

	cblas_dcopy( n, zc, 1, yc, 1 );
	//Again checking that C2 = C0-A*B by a simple check (C2*xc vs yc)
	cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0e+00, C2, n, xc, 1, -1.0e+00, yc, 1 );
	res2 = 0.e+00; for(i = 0; i < n ; i++) { tmp = fabs(yc[i]); if ( res2 < tmp ){ res2 = tmp; } }

	//printf("Call to GEMM and recovery (roll-forwarding %3d values):\nn = %3d\telapsed = %7.4f (sec)\t%e\n", col_count*row_count, n, elapsed, res2);
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
