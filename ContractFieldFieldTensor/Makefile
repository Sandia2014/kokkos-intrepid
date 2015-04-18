all: ContractFieldFieldTensor

ContractFieldFieldTensor:  ContractFieldFieldTensor.cpp CFFT_AdaptiveSlicing_Cuda.hpp CFFT_AdaptiveSlicing_Kokkos.hpp CFFT_Independent_Cuda.hpp CFFT_Independent_Kokkos.hpp CFFT_Slicing_Cuda.hpp CFFT_Slicing_Kokkos.hpp CFFT_TeamReduction.hpp CFFT_Tiling_Kokkos.hpp
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03/include -o ContractFieldFieldTensor ContractFieldFieldTensor.cpp -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

clean:
	rm -f ContractFieldFieldTensor
