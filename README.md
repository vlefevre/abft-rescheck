# abft-rescheck
A repository for the code of "A comparison of several fault-tolerance methods for error detection and correction of floating-point errors  in matrix multiplication" by Valentin Le Fèvre, Thomas Hérault, Julien Langou and Yves Robert

/!\ Please adapt the Makefile to use your blas library /!\

You might also need to create a directory "failures" to store the information about the number of errors in the matrices at each step

6 programs can be compiled (and their multi-threaded version) and they correspond to one of the implementations described in the paper:\
dgemm_basis is just a GEMM without fault tolerance\
dgemm_abft_rerun is the implementation ABFT-Recomp\
dgemm_abft_system is the implementation ABFT-Solve\
dgemm_rescheck_rerun is the implementation RC-Recomp\
dgemm_rescheck_system is the implementation RC-Solve\
dgemm_trip is the implementation Replication

The three options are:\
-x val: allows val+1 rounds of checking (and val rounds of correction)\
-i val: runs val iterations of the program\
-n val: sets the matrix size to val\
-r val: sets the error rate to val\
For ABFT-Solve a fourth parameter -m val allows to define the number of checksums added
