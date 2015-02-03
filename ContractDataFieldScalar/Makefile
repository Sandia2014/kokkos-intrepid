all: ContractDataFieldScalar

ContractDataFieldScalar: ContractDataFieldScalar.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o ContractDataFieldScalar ContractDataFieldScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install-shadowfax/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractDataDataTensor: ContractDataDataTensor.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -std=c++11 -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o ContractDataDataTensor ContractDataDataTensor.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install-shadowfax/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

clean:
	rm -f ContractFieldFieldScalar
