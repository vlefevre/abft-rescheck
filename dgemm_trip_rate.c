#include "solve.h"

int main(int argc, char ** argv) {

	int i, j, n;
	double *A, *B, *C0, *C1, *C2, *C3, *C4, *C5;
	double *x1, *x2, *y, *z;
	double *xc, *yc, *zc;
	double res1, res2, tmp;

	int row_count, col_count, nb_error;
	int *row_loc, *col_loc;
//	int *ix, *jx;
	double error_rate;
	double tol;

	int iit, nb_iter, reexec, maxreexec, nb_fail=0;
	int isDone;

	double elapsed,initTime,dgemmTime,solveTime,checkTime,cleanTime;
	struct timeval tp;
	
	gettimeofday(&tp, NULL);
	srand((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

    	n = 4;
	char nstring[10] = "4";
//	strcpy(nstring,"4");
//	printf("%s ",nstring);
	nb_error = 3;
	tol = 1e-10;
	nb_iter = 1;
	error_rate = 1e-9;
	char error_ratestring[100] = "1e-9";
//	strcpy(error_ratestring, "1e-9");

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			strcpy(nstring, *(argv + i + 1));
			i++;
		}
		if( strcmp( *(argv + i), "-i") == 0) {
			nb_iter = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-k") == 0) {
			nb_error = atoi( *(argv + i + 1) );
			i++;
		}
		if ( strcmp( *(argv + i), "-r") == 0) {
			error_rate = atof(*(argv + i + 1));
			strcpy(error_ratestring, *(argv + i + 1));
			i++;
		}
	}

	char filename[1000] = "failures/dgemm_trip_";
	strcat(filename,nstring);
	strcat(filename,"_");
	strcat(filename,error_ratestring);
	FILE* failures = fopen(filename, "w");

	//printf("%s\n",filename);

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
	//Find coordinates of errors
	/* OLD STUFF
//	ix goes from 0 to n-1, jx goes from 0 to n-1,
	ix = (int *) malloc( nb_error * sizeof(int));
	jx = (int *) malloc( nb_error * sizeof(int));

	for(i = 0; i < nb_error ; i++) { ix[i] = ( rand() % n ); jx[i] = ( rand() % n ); }
	if (DEBUG)
	{
	printf("nb_error  = %3d: error_loc = [", nb_error);
	for(i = 0; i < nb_error ; i++) printf(" (%3d,%3d) ", ix[i], jx[i]);
	printf("];\n");
	}
	*/
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

	elapsed=0;
	dgemmTime=0;
	checkTime=0;
	solveTime=0;
	initTime=0;
	cleanTime=0;
	for (iit = 0; iit < nb_iter; iit++)
	{

	fprintf(failures, "%d ", iit);
	gettimeofday(&tp, NULL);
	elapsed-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	initTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	//take copy into account as original data is needed to correct

	#pragma omp parallel for
 	for(i = 0; i < n; i++)
		memcpy(C2+i*n, C0+i*n, n*sizeof(double));
	C3 = (double *) malloc( n * n * sizeof(double));
	#pragma omp parallel for
 	for(i = 0; i < n; i++)
		memcpy(C3+i*n, C0+i*n, n*sizeof(double));


	/*if (DEBUG)
	{
		for (i=0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%f ",C2[i+j*n]);
			printf("\n");
		}
	}*/

	gettimeofday(&tp, NULL);
	initTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

////////////

	gettimeofday(&tp, NULL);
	dgemmTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
//	compute C2 = C0 - A * B
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C2, n);
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C3, n);
	gettimeofday(&tp, NULL);
	dgemmTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	if (DEBUG)
	{
		for (i = 0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%f ",C2[i+j*n]);
			printf("\n");
		}
	}
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
					printf("Cell C2 %d,%d  corrupted.\n",i,j);
					fflush(stdout);
				}
				nb_error++;
				C2[i+j*n] = C2[i+j*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
			}
		}
	}
	fprintf(failures,"%d ",nb_error);
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
					printf("Cell C3 %d,%d  corrupted.\n",i,j);
					fflush(stdout);
				}
				nb_error++;
				C3[i+j*n] = C3[i+j*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
			}
		}
	}
	fprintf(failures,"%d ",nb_error);
	/*  ===OLD WAY===
	//modifiying the erroneous elements (should not be part of time computation but should be negligible)
	for(i = 0; i < nb_error ; i++) { 
		C2[ ix[i] + jx[i] * n ] = C2[ ix[i] + jx[i] * n ] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 ) ;
	}
	*/

	if (DEBUG)
	{
		for (i = 0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%f ",C2[i+j*n]);
			printf("\n");
		}
	}
	if (DEBUG)
	{
		for (i = 0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%f ",C3[i+j*n]);
			printf("\n");
		}
	}

	check:
	gettimeofday(&tp, NULL);
	checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	isDone = 1;
	#pragma omp parallel for collapse(2)
	for (i = 0; i < n; i++)
	{
		for (j=0; j<n; j++)
		{
			if (C2[i+j*n] != C3[i+j*n])
			{
				isDone = 0;
			}
		}
	}
	
	gettimeofday(&tp, NULL);
	checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	if (!isDone)
	{
		
		if (DEBUG)
			printf("3rd attempt\n");
		gettimeofday(&tp, NULL);
		initTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		C4 = (double *) malloc( n * n * sizeof(double));
		#pragma omp parallel for
		for(i = 0; i < n; i++)
			memcpy(C4+i*n, C0+i*n, n*sizeof(double));
		gettimeofday(&tp, NULL);
		initTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		gettimeofday(&tp, NULL);
		dgemmTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C4, n);
		gettimeofday(&tp, NULL);
		dgemmTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		#pragma omp parallel for collapse(2)
		nb_error = 0;
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++ )
			{
				if (hasFailed(2*n-1,error_rate))
				{
					if (DEBUG)
					{
						printf("Cell C4 %d,%d  corrupted.\n",i,j);
						fflush(stdout);
					}
					nb_error++;
					C4[i+j*n] = C4[i+j*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
				}
			}
		}
		fprintf(failures,"%d ",nb_error);
		
		if (DEBUG)
		{
			for (i = 0; i<n; i++)
			{
				for (j=0; j<n; j++)
					printf("%f ",C4[i+j*n]);
				printf("\n");
			}
		}
		gettimeofday(&tp, NULL);
		checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		isDone = 1;
		#pragma omp parallel for collapse(2)
		for (i = 0; i < n; i++)
		{
			for (j=0; j<n; j++)
			{
				if (C2[i+j*n] != C3[i+j*n])
				{
					if (C2[i+j*n] != C4[i+j*n])
					{
						if (C3[i+j*n] != C4[i+j*n])
						{
							isDone = 0;
						} else {
							//Correction of C2
							C2[i+j*n] = C3[i+j*n];
						}
					}
				}
			}
		}
		gettimeofday(&tp, NULL);
		checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

		if (!isDone)
		{
			if (DEBUG)
				printf("4th attempt\n");
			gettimeofday(&tp, NULL);
			initTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
			C5 = (double *) malloc( n * n * sizeof(double));
			#pragma omp parallel for
			for(i = 0; i < n; i++)
				memcpy(C5+i*n, C0+i*n, n*sizeof(double));
			gettimeofday(&tp, NULL);
			initTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
			
			gettimeofday(&tp, NULL);
			dgemmTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
			cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0e+00, A, n, B, n, +1.0e+00, C5, n);
			gettimeofday(&tp, NULL);
			dgemmTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

			#pragma omp parallel for collapse(2)
			nb_error = 0;
			for (i = 0; i < n; i++)
			{
				for (j = 0; j < n; j++ )
				{
					if (hasFailed(2*n-1,error_rate))
					{
						if (DEBUG)
						{
							printf("Cell C5 %d,%d  corrupted.\n",i,j);
							fflush(stdout);
						}
						nb_error++;
						C5[i+j*n] = C5[i+j*n] * ( (double)rand() / (double)(RAND_MAX) + 0.5e+00 );
					}
				}
			}
			fprintf(failures,"%d ",nb_error);
			
			if (DEBUG)
			{
				for (i = 0; i<n; i++)
				{
					for (j=0; j<n; j++)
						printf("%f ",C5[i+j*n]);
					printf("\n");
				}
			}
			gettimeofday(&tp, NULL);
			checkTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
			isDone = 1;
			#pragma omp parallel for collapse(2)
			for (i = 0; i < n; i++)
			{
				for (j=0; j<n; j++)
				{
					if (C2[i+j*n] != C3[i+j*n])
					{
						if (C2[i+j*n] != C4[i+j*n])
						{
							if (C2[i+j*n] != C4[i+j*n])
							{
								if (C2[i+j*n] != C5[i+j*n])
								{
									if (C3[i+j*n] != C4[i+j*n])
									{
										if (C3[i+j*n] != C5[i+j*n])
										{
											if (C4[i+j*n] != C5[i+j*n])
												isDone = 0;
											else
												C2[i+j*n] = C4[i+j*n];
										} else {
											C2[i+j*n] = C3[i+j*n];
										}
									} else {
										C2[i+j*n] = C3[i+j*n];
									}
								}
							}
						}
					}
				}
			}
			gettimeofday(&tp, NULL);
			checkTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

			if (!isDone)
				nb_fail++;

			gettimeofday(&tp, NULL);
			cleanTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
			free( C5 );
			gettimeofday(&tp, NULL);
			cleanTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		}
		gettimeofday(&tp, NULL);
		cleanTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
		free( C4 );
		gettimeofday(&tp, NULL);
		cleanTime+=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
	}
/*	if (DEBUG)
	{
		for (i=0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%f ",C2[i+j*n]);
			printf("\n");
		}
	}*/

	gettimeofday(&tp, NULL);
	cleanTime-=((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

	free( C3 );

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


/*	free(ix);
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
