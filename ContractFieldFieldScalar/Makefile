# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # compiler
  CC              := /opt/local/bin/g++-mp-4.8
  NVCCSCC         := /opt/local/bin/g++-mp-4.7
  # cuda
  CUDA_PATH       ?= /usr/local/cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib/
  LD_FLAGS        += -Xlinker -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
  # tbb
  TBB_DIR         += /opt/tbb-4.2
  LD_FLAGS        += -L$(TBB_DIR)/lib-gcc48 -ltbb
  # kokkos
  #KOKKOS_DIR      := /opt/kokkos
  # eigen
  EIGEN_DIR       := /opt/eigen
  # avx vectorization support
  #CC_FLAGS        += -mavx -Wa,-q
else
  # compiler
  #CC              := g++
  #CUDA_PATH       ?= /usr/local/cuda-6.5
  #KOKKOS_DIR      := /clinic/2014/sandia14/trilinos-install/
  CC              := g++-4.7.3
  CUDA_PATH       ?= /opt/cuda
  #KOKKOS_DIR      := /clinic/2014/sandia14/trilinos-install-shadowfax
  KOKKOS_DIR      := /clinic/2014/sandia14/TrilinosInstall-2014-02-03
  NVCCSCC         := $(CC)
  # cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64/
  ifeq ($(OS_SIZE),32)
    CC_FLAGS      += -m32
  else
    CC_FLAGS      += -m64
  endif
  LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
  # tbb
  TBB_DIR         += /clinic/2014/sandia14/tbb-4.3
  LD_FLAGS        += -L$(TBB_DIR)/lib -ltbb
  # eigen
  EIGEN_DIR       := /clinic/2014/sandia14/eigen
  # avx vectorization support
  CC_FLAGS        += -mavx
  # enable kokkos
  CC_FLAGS        += -DENABLE_KOKKOS
  NVCC_FLAGS      += -DENABLE_KOKKOS -L$(KOKKOS_DIR)/lib -lkokkoscore
  CC_INCLUDE      += -I$(KOKKOS_DIR)/include
  NVCC_INCLUDE    += -I$(KOKKOS_DIR)/include
  LD_FLAGS        += -L$(KOKKOS_DIR)/lib -lkokkoscore
  NVCC_LD_FLAGS   += -lhwloc
endif

# eigen
CC_INCLUDE        += -I$(EIGEN_DIR)

# tbb
CC_INCLUDE        += -I$(TBB_DIR)/include

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCC_FLAGS += -m32
else
      NVCC_FLAGS += -m64
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

CC_FLAGS        += -std=c++11 -fopenmp
NVCC_FLAGS      += -std=c++11
NVCC_LD_FLAGS   += -lgomp

TARGETS = CFFS

all: $(TARGETS)

GraphMaker: GraphMaker.cu
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ $< $(NVCC_LD_FLAGS) -Xcompiler -fopenmp,-O3
	#$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ $< $(NVCC_LD_FLAGS) -Xcompiler -fopenmp,-O0,-g


ContractFieldFieldScalar: ContractFieldFieldScalar.cu
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ $< $(NVCC_LD_FLAGS) -Xcompiler -fopenmp,-O3
	#$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ $< $(NVCC_LD_FLAGS) -Xcompiler -fopenmp,-O0,-g

CFFS_teams: CFFS_teams.cpp
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03//include -o CFFS_teams CFFS_teams.cpp -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

TeamKokkos: GraphMaker.cpp
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03//include -o TeamKokkos GraphMaker.cpp -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas



CFFS: CFFS.cpp
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03//include -o CFFS CFFS.cpp -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas





Example: nested_parallel_for.cpp
	/clinic/2014/sandia14/bin/nvcc_wrapper -std=c++11 -m64 -O3 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/TrilinosInstall-2014-02-03//include -o example nested_parallel_for.cpp -fopenmp -L/clinic/2014/sandia14/TrilinosInstall-2014-02-03/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
