all: ContractDataDataScalar

ContractDataDataScalar: ContractDataDataScalar.cu
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03/include -o ContractDataDataScalar ContractDataDataScalar.cu -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

clean:
	rm -f ContractDataDataScalar
