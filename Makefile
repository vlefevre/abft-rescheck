
all: dgemm_basis dgemm_abft_rerun_rate dgemm_abft_system_rate dgemm_rescheck_rerun_rate dgemm_rescheck_system_rate dgemm_abft_rerun dgemm_abft_system dgemm_rescheck_rerun dgemm_rescheck_system dgemm_trip_rate

xmain_dtrsm.exe: main_dtrsm.o
	gfortran -o $@ main_dtrsm.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

xmain_dgemm.exe: main_dgemm.o solve.o
	gfortran -o $@ main_dgemm.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

xmain_dgemm_v2.exe: main_dgemm_v2.o solve.o
	gfortran -o $@ main_dgemm_v2.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_basis: dgemm_basis.o solve.o
	gfortran -o $@ dgemm_basis.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_abft_rerun: dgemm_abft_rerun.o solve.o
	gfortran -o $@ dgemm_abft_rerun.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_abft_rerun_rate: dgemm_abft_rerun_rate.o solve.o
	gfortran -o $@ dgemm_abft_rerun_rate.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_abft_system: dgemm_abft_system.o solve.o
	gfortran -o $@ dgemm_abft_system.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_abft_system_rate: dgemm_abft_system_rate.o solve.o
	gfortran -o $@ dgemm_abft_system_rate.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_rescheck_rerun: dgemm_rescheck_rerun.o solve.o
	gfortran -o $@ dgemm_rescheck_rerun.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_rescheck_rerun_rate: dgemm_rescheck_rerun_rate.o solve.o
	gfortran -o $@ dgemm_rescheck_rerun_rate.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_rescheck_system: dgemm_rescheck_system.o solve.o
	gfortran -o $@ dgemm_rescheck_system.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_rescheck_system_rate: dgemm_rescheck_system_rate.o solve.o
	gfortran -o $@ dgemm_rescheck_system_rate.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

dgemm_trip_rate: dgemm_trip_rate.o solve.o
	gfortran -o $@ dgemm_trip_rate.o solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

test: test_solve.o
	gfortran -o $@ test_solve.o /home/valentin/Documents/lapack/liblapacke.a /home/valentin/Documents/lapack/liblapack.a /home/valentin/Documents/lapack/libcblas.a /home/valentin/Documents/lapack/librefblas.a

.c.o:
	gcc -g -I/home/valentin/Documents/lapack/LAPACKE/include -I/home/valentin/Documents/lapack/CBLAS/include -c -o $@ $<

clean:
	rm -f *o *exe
