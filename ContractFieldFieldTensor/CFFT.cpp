// -*- C++ -*-
// ArrayOfTensors.cc
// a huge comparison of different ways of doing an array of dot products
// Jeff Amelang, 2014

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>

// c++ junk
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <array>
#include <fstream>
using std::string;
using std::vector;
using std::array;

// header file for openmp
#include <omp.h>
#include "CFFT_Tiling.hpp"
#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif // ENABLE_KOKKOS

enum CudaStyle {CudaStyle_Independent,
                CudaStyle_Reduction,
                CudaStyle_Slicing,
                CudaStyle_AdaptiveSlicing};

enum KokkosStyle {
  KokkosStyle_Independent,
  KokkosStyle_Tiling
};

enum ClearCacheStyle {ClearCacheAfterEveryRepeat,
                      DontClearCacheAfterEveryRepeat};

string
convertCudaStyleToString(const CudaStyle cudaStyle) {
  switch (cudaStyle) {
  case CudaStyle_Independent:
    return string("CudaStyle_Independent");
  case CudaStyle_Reduction:
    return string("CudaStyle_Reduction");
  case CudaStyle_Slicing:
    return string("CudaStyle_Slicing");
  case CudaStyle_AdaptiveSlicing:
    return string("CudaStyle_AdaptiveSlicing");
  default:
    fprintf(stderr, "invalid cuda style\n");
    exit(1);
  };
}

// stolen from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline
void
gpuAssert(const cudaError_t code, const char *file, const int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort == true) {
      exit(code);
    }
  }
}

timespec
getTimePoint() {
  timespec timepoint;
  clock_gettime(CLOCK_MONOTONIC, &timepoint);
  return timepoint;
}

// yay for having to use pre-c++11 timing because of nvcc
double
getElapsedTime(const timespec & start, const timespec & end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}





__global__
void
doCudaClearCache_kernel(const unsigned int junkDataSize,
                        const int * const __restrict__ dev_junkDataToClearTheCache,
                        int * dev_result) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  int partialSum = 0;
  while (index < junkDataSize) {
    partialSum += dev_junkDataToClearTheCache[index];
    index += blockDim.x * gridDim.x;
  }
  atomicAdd(dev_result, partialSum);
}

__global__
void
doCudaTensors_Independent_kernel(const unsigned int numberOfTensors,
                                 const unsigned int numLeftFields,
                                 const unsigned int numRightFields,
                                 const unsigned int numPoints,
                                 const unsigned int tens1,
                                 const unsigned int tens2,
                                 const float * const __restrict__ dev_tensorData_Left,
                                 const float * const __restrict__ dev_tensorData_Right,
                                 float * dev_tensorResults) {

  unsigned int myID = blockIdx.x * blockDim.x + threadIdx.x;
  while (myID < (numberOfTensors * numLeftFields * numRightFields)) {
    float sum = 0;
    int myCell = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);
    int lbf = matrixIndex / numRightFields;
    int rbf = matrixIndex % numRightFields;

    int clOff = numLeftFields*numPoints*tens1*tens2;
    int crOff = numRightFields*numPoints*tens1*tens2;
    int cOut = numLeftFields*numRightFields;
    int lOff = numPoints*tens1*tens2;
    int lOut = numRightFields;
    //int rOff = numPoints*tens1*tens2;
    int pOff = tens1*tens2;
    int tenOff = tens2;

    for (int qp = 0; qp < numPoints; qp++) {
      for (int iTens1 = 0; iTens1 < tens1; iTens1++) {
        for (int iTens2 = 0; iTens2 < tens2; iTens2++) {
          sum += dev_tensorData_Left[myCell*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
          dev_tensorData_Right[myCell*crOff+qp*numRightFields*tens1*tens2+
                              iTens1*numRightFields*tens2+iTens2*numRightFields
                              +rbf];
        }
      }
    }

    dev_tensorResults[myCell*cOut+lbf*lOut+rbf] = sum;

    myID += blockDim.x * gridDim.x;
  }
}

__global__
void
doCudaContractions_Slicing_kernel(const unsigned int numberOfTensors,
                                 const unsigned int numLeftFields,
                                 const unsigned int numRightFields,
                                 const unsigned int numPoints,
                                 const unsigned int tens1,
                                 const unsigned int tens2,
                                 const float * const __restrict__ dev_tensorData_Left,
                                 const float * const __restrict__ dev_tensorData_Right,
                                 float * dev_tensorResults) {

  extern __shared__ float sliceStorage[];

  const unsigned int col = threadIdx.x;

  unsigned int currentBlock = blockIdx.x;
  const unsigned int numBlocks = numRightFields*numberOfTensors;
  const unsigned int contractionSize = numPoints * tens1 * tens2;

  while (currentBlock < numBlocks) {
    syncthreads();
    const unsigned int cell = currentBlock / numLeftFields;
    const unsigned int row = currentBlock - cell * numLeftFields;

    for (unsigned int p = col; p < contractionSize; p += blockDim.x) {
      sliceStorage[p] = dev_tensorData_Left[cell*numLeftFields*contractionSize +
        row*contractionSize + p];
    }
    //dev_contractionResults[cell*numRightFields*numLeftFields + row*numRightFields + col] = -1;
    syncthreads();

    float sum = 0;
    for (int p = 0; p < contractionSize; ++p) {
      sum += sliceStorage[p] * dev_tensorData_Right[cell*numRightFields*contractionSize +
        p*numRightFields + col];
    }

    dev_tensorResults[cell*numRightFields*numLeftFields + row*numRightFields + col] = sum;

    currentBlock += gridDim.x;
  }
}

__global__
void
doCudaContractions_AdaptiveSlicing_kernel(const unsigned int numberOfTensors,
                                 const unsigned int numLeftFields,
                                 const unsigned int numRightFields,
                                 const unsigned int numPoints,
                                 const unsigned int tens1,
                                 const unsigned int tens2,
                                 const float * const __restrict__ dev_tensorData_Left,
                                 const float * const __restrict__ dev_tensorData_Right,
                                 float * dev_tensorResults) {

  extern __shared__ float sliceStorage[];

  //const unsigned int blockSize = blockDim.x;
  const unsigned int contractionSize = numPoints * tens1 * tens2;
  const unsigned int threadRow = threadIdx.x / contractionSize;
  const unsigned int col = threadIdx.x - (threadRow * contractionSize);

  unsigned int currentBlock = blockIdx.x;
  const unsigned int numBlocks = gridDim.x;

  while (currentBlock < numBlocks) {
    syncthreads();
    const unsigned int cell = (currentBlock*2) / numLeftFields;
    const unsigned int row = (currentBlock*2) - cell * numLeftFields;

    if((cell < numberOfTensors) && (row < numLeftFields)) {
      for (unsigned int p = col; p < contractionSize; p += blockDim.x) {
        sliceStorage[p] = dev_tensorData_Left[cell*numLeftFields*contractionSize +
          row*contractionSize + p];
        }
      //dev_contractionResults[cell*numRightFields*numLeftFields + row*numRightFields + col] = -1;
      syncthreads();
      float sum = 0;
      for (int p = 0; p < contractionSize; ++p) {
        sum += sliceStorage[p + (threadRow*contractionSize)] * dev_tensorData_Right[cell*numRightFields*contractionSize +
          p*numRightFields + col];
        }

        dev_tensorResults[cell*numRightFields*numLeftFields + row*numRightFields + col] = sum;
    }
    currentBlock += gridDim.x;
  }
}
__global__
void
doCudaTensors_Reduction_kernel(const unsigned int numberOfTensors,
                                   const unsigned int tensorSize,
                                   const float * const __restrict__ dev_tensorData_LayoutRight_A,
                                   const float * const __restrict__ dev_tensorData_LayoutRight_B,
                                   float * dev_tensorResults) {

  extern __shared__ float sharedMemory[];

  unsigned int tensorIndex = blockIdx.x;
  while (tensorIndex < numberOfTensors) {

    // goal: compute the contribution to the dot product from this thread
    const unsigned int shortcutIndex = tensorIndex * tensorSize;
    float partialSum = 0;
    unsigned int entryIndex = threadIdx.x;
    while (entryIndex < tensorSize) {
      const unsigned int index = shortcutIndex + entryIndex;
      partialSum +=
        dev_tensorData_LayoutRight_A[index] *
        dev_tensorData_LayoutRight_B[index];
      entryIndex += blockDim.x;
    }
    // set this thread's value
    sharedMemory[threadIdx.x] = partialSum;

    // goal: reduce the warp's contribution to one number and add it to the
    //  dot product.

    // each warp does its own reduction
    const unsigned int warpIndex = threadIdx.x / 32;
    const unsigned int indexWithinWarp = threadIdx.x - warpIndex * 32;
    const unsigned int warpStartingIndexInSharedMemory = warpIndex * 32;
    // the first quarter of the threads in the warp make small partial sums
    if (indexWithinWarp < 8) {
      const int i = indexWithinWarp * 4;
      partialSum  = sharedMemory[warpStartingIndexInSharedMemory + i];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + i + 1];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + i + 2];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + i + 3];
      sharedMemory[warpStartingIndexInSharedMemory + i] = partialSum;
    }
    // the first thread in the warp reduces the 8 partial sums
    if (indexWithinWarp == 0) {
      partialSum += sharedMemory[warpStartingIndexInSharedMemory +  4];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory +  8];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + 12];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + 16];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + 20];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + 24];
      partialSum += sharedMemory[warpStartingIndexInSharedMemory + 28];
      // and adds it to the global sum
      atomicAdd(&dev_tensorResults[tensorIndex], partialSum);
    }

    // move on to the next dot product
    tensorIndex += gridDim.x;
  }
}

void
writeTimesMatrixToFile(const vector<vector<float> > & times,
                       const string filename) {

  const unsigned int numberOfTensorSizes = times.size();
  // yeah, yeah, kinda unsafe
  const unsigned int numberOfMemorySizes = times[0].size();
  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%s.csv", filename.c_str());
  FILE* file = fopen(sprintfBuffer, "w");
  for (unsigned int tensorSizeIndex = 0;
       tensorSizeIndex < numberOfTensorSizes;
       ++tensorSizeIndex) {
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      if (memorySizeIndex > 0) {
        fprintf(file, ", ");
      }
      fprintf(file, "%10.4e", times[tensorSizeIndex][memorySizeIndex]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

void
checkAnswer(const vector<float> & correctResults,
            const vector<float> & tensorResults,
            const unsigned int tensorSize,
            const unsigned int memorySize,
            const string flavorName) {
  for (unsigned int tensorIndex = 0;
       tensorIndex < correctResults.size();
       ++tensorIndex) {
    if (std::abs(correctResults[tensorIndex] -
                 tensorResults[tensorIndex]) /
        std::abs(correctResults[tensorIndex]) > 1e-4) {
      fprintf(stderr, "invalid answer for dot product index %u for "
              "flavor %s, "
              "should be %e but we have %e, "
              "tensorSize = %u, memorySize = %8.2e\n",
              tensorIndex, flavorName.c_str(),
              correctResults[tensorIndex],
              tensorResults[tensorIndex],
              tensorSize, float(memorySize));
      exit(1);
    }
  }
}

double
runCudaTest(const CudaStyle cudaStyle,
            const unsigned int numberOfThreadsPerBlock,
            const unsigned int numberOfRepeats,
            const unsigned int maxNumberOfCudaBlocks,
            const unsigned int numberOfTensors,
            const unsigned int numLeftFields,
            const unsigned int numRightFields,
            const unsigned int numPoints,
            const unsigned int tens1,
            const unsigned int tens2,
            const unsigned int maxNumberOfTensors,
            const unsigned int tensorSize,
            const unsigned int memorySize,
            const vector<float> & correctResults,
            const ClearCacheStyle clearCacheStyle,
            const int * const dev_junkDataToClearTheCache,
            const unsigned int junkDataSize,
            const vector<float> & tensorData_Right,
            const vector<float> & tensorData_Left,
            int * const dev_junkDataCounter,
            unsigned int * const totalNumberOfRepeats,
            float * const dev_tensorResults,
            vector<float> * const tensorResults) {

  const unsigned int numberOfBlocks =
    min(maxNumberOfCudaBlocks,
        (unsigned int)ceil(numberOfTensors*numRightFields*numLeftFields/float(numberOfThreadsPerBlock)));

  // Format the data the way we want and then copy it to the GPU
  vector<float> contractionData_GPURight(tensorData_Right.size());
  vector<float> contractionData_GPULeft(tensorData_Left.size());

  int cLOff = numLeftFields*numPoints*tens1*tens2;
  int cROff = numRightFields*numPoints*tens1*tens2;
  int basisOff = numPoints*tens1*tens2;
  int pLOff = tens1*tens2;
  int pROff = tens1*tens2;
  int tROff = tens2;
  int t2ROff = 1;
  int tOff = tens2;

  for (int cl = 0; cl < numberOfTensors; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for (int iTens1 = 0; iTens1 < tens1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < tens2; ++iTens2) {
          for(int rbf = 0; rbf < numRightFields; ++rbf) {
            contractionData_GPURight[cl*cROff + qp*numRightFields*pROff + iTens1*numRightFields*tROff +
            iTens2*numRightFields + rbf] =
            tensorData_Right[cl*cROff + rbf*basisOff + qp*pROff +
            iTens1*tROff + iTens2*t2ROff];
          }
          for(int lbf = 0; lbf < numLeftFields; ++lbf) {
            contractionData_GPULeft[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2] =
            tensorData_Left[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2];
          }
        }
      }
    }
  }
  /*
  for (int cl = 0; cl < numberOfTensors; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for (int iTens1 = 0; iTens1 < tens1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < tens2; ++iTens2) {
          for(int rbf = 0; rbf < numRightFields; ++rbf) {
            contractionData_GPURight[cl*numPoints*numRightFields*tens1*tens2 + qp*numRightFields*tens1*tens2+
            iTens1*numRightFields*tens1+ numRightFields*iTens2+rbf] =
            tensorData_Right[cl*cROff + rbf*basisOff + qp*pROff +
            iTens1*tROff + iTens2*t2ROff];
          }
          for(int lbf = 0; lbf < numLeftFields; ++lbf) {
            contractionData_GPULeft[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2] =
            tensorData_Left[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2];
          }
        }
      }
    }
  }
  */

  // Then copy it over
  float * dev_contractionData_Right;
  checkCudaError(cudaMalloc((void **) &dev_contractionData_Right,
   numberOfTensors * numPoints * tens1 * tens2 * numRightFields * sizeof(float)));

  checkCudaError(cudaMemcpy(dev_contractionData_Right,
    &contractionData_GPURight[0], numberOfTensors * numPoints * tens1 * tens2 *
    numRightFields * sizeof(float), cudaMemcpyHostToDevice));

  float * dev_contractionData_Left;
  checkCudaError(cudaMalloc((void **) &dev_contractionData_Left, numberOfTensors
  * numPoints * tens1 * tens2 * numLeftFields * sizeof(float)));

  checkCudaError(cudaMemcpy(dev_contractionData_Left, &contractionData_GPULeft[0],
  numberOfTensors * numPoints * tens1 * tens2 * numLeftFields * sizeof(float),
  cudaMemcpyHostToDevice));



  timespec tic;
  double totalElapsedTime = 0;
  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
    *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
    if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
         repeatIndex == 1) ||
        clearCacheStyle == ClearCacheAfterEveryRepeat) {
      tic = getTimePoint();
    }

    // do the actual calculation
    if (cudaStyle == CudaStyle_Independent) {
      doCudaTensors_Independent_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(numberOfTensors,
                                   numLeftFields,
                                   numRightFields,
                                   numPoints,
                                   tens1,
                                   tens2,
                                   dev_contractionData_Left,
                                   dev_contractionData_Right,
                                   dev_tensorResults);
    } else if (cudaStyle == CudaStyle_Reduction) {
      doCudaTensors_Reduction_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        numberOfThreadsPerBlock * sizeof(float)>>>(numberOfTensors,
                                                   tensorSize,
                                                   dev_contractionData_Left,
                                                   dev_contractionData_Right,
                                                   dev_tensorResults);
    } else {
      fprintf(stderr, "unknown cuda style\n");
      exit(1);
    }

    // wait for the kernel launch
    checkCudaError(cudaPeekAtLastError());
    checkCudaError(cudaDeviceSynchronize());
    if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
      const timespec toc = getTimePoint();
      const float elapsedTime = getElapsedTime(tic, toc);
      totalElapsedTime += elapsedTime;

      const unsigned int junkNumberOfBlocks =
        min(maxNumberOfCudaBlocks,
            (unsigned int)ceil(junkDataSize/float(numberOfThreadsPerBlock)));
      doCudaClearCache_kernel<<<junkNumberOfBlocks,
        numberOfThreadsPerBlock>>>(junkDataSize,
                                   dev_junkDataToClearTheCache,
                                   dev_junkDataCounter);
      // wait for the kernel launch
      checkCudaError(cudaPeekAtLastError());
      checkCudaError(cudaDeviceSynchronize());
    }
  }
  if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
    const timespec toc = getTimePoint();
    const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
    totalElapsedTime = elapsedTime;
  }
  // copy over the results from the gpu to the cpu
  checkCudaError(cudaMemcpy(&tensorResults->at(0), dev_tensorResults,
                            numberOfTensors *numLeftFields*numRightFields* sizeof(float),
                            cudaMemcpyDeviceToHost));
  // check the results
  checkAnswer(correctResults, *tensorResults,
              tensorSize, memorySize,
              convertCudaStyleToString(cudaStyle));

  // scrub the results
  std::fill(tensorResults->begin(),
            tensorResults->end(),
            std::numeric_limits<float>::quiet_NaN());
  checkCudaError(cudaMemcpy(dev_tensorResults, &tensorResults->at(0),
                            numberOfTensors * numLeftFields*numRightFields*sizeof(float),
                            cudaMemcpyHostToDevice));
  checkCudaError(cudaFree(dev_contractionData_Right));
  checkCudaError(cudaFree(dev_contractionData_Left));
  return totalElapsedTime;
}

double
runCudaTeamTest(const CudaStyle cudaStyle,
            const unsigned int numberOfThreadsPerBlock,
            const unsigned int numberOfRepeats,
            const unsigned int maxNumberOfCudaBlocks,
            const unsigned int numberOfTensors,
            const unsigned int numLeftFields,
            const unsigned int numRightFields,
            const unsigned int numPoints,
            const unsigned int tens1,
            const unsigned int tens2,
            const unsigned int maxNumberOfTensors,
            const unsigned int tensorSize,
            const unsigned int memorySize,
            const vector<float> & correctResults,
            const ClearCacheStyle clearCacheStyle,
            const int * const dev_junkDataToClearTheCache,
            const unsigned int junkDataSize,
            const vector<float> & tensorData_Right,
            const vector<float> & tensorData_Left,
            int * const dev_junkDataCounter,
            unsigned int * const totalNumberOfRepeats,
            float * const dev_tensorResults,
            vector<float> * const tensorResults) {


  unsigned int numberOfSlicingBlocks;
  if(cudaStyle == CudaStyle_Slicing) {
   numberOfSlicingBlocks =
                min(maxNumberOfCudaBlocks, numberOfTensors*numRightFields);
  } else {
    numberOfSlicingBlocks =
                 min(maxNumberOfCudaBlocks, (unsigned) ceil((numberOfTensors*numRightFields)/2));
  }
  const unsigned int contractionSize = numPoints * tens1 * tens2;
  // Format the data the way we want and then copy it to the GPU
  vector<float> contractionData_GPURight(tensorData_Right.size());
  vector<float> contractionData_GPULeft(tensorData_Left.size());

  int cLOff = numLeftFields*numPoints*tens1*tens2;
  int cROff = numRightFields*numPoints*tens1*tens2;
  int basisOff = numPoints*tens1*tens2;
  int pLOff = tens1*tens2;
  int pROff = tens1*tens2;
  int tROff = tens2;
  int t2ROff = 1;
  int tOff = tens2;

  for (int cl = 0; cl < numberOfTensors; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for (int iTens1 = 0; iTens1 < tens1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < tens2; ++iTens2) {
          for(int rbf = 0; rbf < numRightFields; ++rbf) {
            contractionData_GPURight[cl*cROff + qp*numRightFields*pROff + iTens1*numRightFields*tROff +
            iTens2*numRightFields + rbf] =
            tensorData_Right[cl*cROff + rbf*basisOff + qp*pROff +
            iTens1*tROff + iTens2*t2ROff];
          }
          for(int lbf = 0; lbf < numLeftFields; ++lbf) {
            contractionData_GPULeft[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2] =
            tensorData_Left[cl*cLOff + lbf*basisOff + qp*pLOff +
            iTens1*tOff + iTens2];
          }
        }
      }
    }
  }

  // Then copy it over
  float * dev_contractionData_Right;
  checkCudaError(cudaMalloc((void **) &dev_contractionData_Right,
   numberOfTensors * numPoints * tens1 * tens2 * numRightFields * sizeof(float)));

  checkCudaError(cudaMemcpy(dev_contractionData_Right,
    &contractionData_GPURight[0], numberOfTensors * numPoints * tens1 * tens2 *
    numRightFields * sizeof(float), cudaMemcpyHostToDevice));

  float * dev_contractionData_Left;
  checkCudaError(cudaMalloc((void **) &dev_contractionData_Left, numberOfTensors
  * numPoints * tens1 * tens2 * numLeftFields * sizeof(float)));

  checkCudaError(cudaMemcpy(dev_contractionData_Left, &contractionData_GPULeft[0],
  numberOfTensors * numPoints * tens1 * tens2 * numLeftFields * sizeof(float),
  cudaMemcpyHostToDevice));



  timespec tic;
  double totalElapsedTime = 0;
  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
    *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
    if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
         repeatIndex == 1) ||
        clearCacheStyle == ClearCacheAfterEveryRepeat) {
      tic = getTimePoint();
    }

    // do the actual calculation
    if (cudaStyle == CudaStyle_Slicing) {
      doCudaContractions_Slicing_kernel<<<numberOfSlicingBlocks,
        numberOfThreadsPerBlock,
        contractionSize * sizeof(float)>>>(numberOfTensors,
                                   numLeftFields,
                                   numRightFields,
                                   numPoints,
                                   tens1,
                                   tens2,
                                   dev_contractionData_Left,
                                   dev_contractionData_Right,
                                   dev_tensorResults);
    } else if (cudaStyle == CudaStyle_AdaptiveSlicing) {

      //THIS IS ALL WRONG RIGHT NOW
      doCudaContractions_AdaptiveSlicing_kernel<<<numberOfSlicingBlocks/2,
        numberOfThreadsPerBlock*2,
        contractionSize * sizeof(float) * 2>>>(numberOfTensors,
                                   numLeftFields,
                                   numRightFields,
                                   numPoints,
                                   tens1,
                                   tens2,
                                   dev_contractionData_Left,
                                   dev_contractionData_Right,
                                   dev_tensorResults);
    } else {
      fprintf(stderr, "unknown cuda style\n");
      exit(1);
    }

    // wait for the kernel launch
    checkCudaError(cudaPeekAtLastError());
    checkCudaError(cudaDeviceSynchronize());
    if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
      const timespec toc = getTimePoint();
      const float elapsedTime = getElapsedTime(tic, toc);
      totalElapsedTime += elapsedTime;

      const unsigned int junkNumberOfBlocks =
        min(maxNumberOfCudaBlocks,
            (unsigned int)ceil(junkDataSize/float(numberOfThreadsPerBlock)));
      doCudaClearCache_kernel<<<junkNumberOfBlocks,
        numberOfThreadsPerBlock>>>(junkDataSize,
                                   dev_junkDataToClearTheCache,
                                   dev_junkDataCounter);
      // wait for the kernel launch
      checkCudaError(cudaPeekAtLastError());
      checkCudaError(cudaDeviceSynchronize());
    }
  }
  if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
    const timespec toc = getTimePoint();
    const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
    totalElapsedTime = elapsedTime;
  }
  // copy over the results from the gpu to the cpu
  checkCudaError(cudaMemcpy(&tensorResults->at(0), dev_tensorResults,
                            numberOfTensors *numLeftFields*numRightFields* sizeof(float),
                            cudaMemcpyDeviceToHost));
  // check the results
  checkAnswer(correctResults, *tensorResults,
              tensorSize, memorySize,
              convertCudaStyleToString(cudaStyle));

  // scrub the results
  std::fill(tensorResults->begin(),
            tensorResults->end(),
            std::numeric_limits<float>::quiet_NaN());
  checkCudaError(cudaMemcpy(dev_tensorResults, &tensorResults->at(0),
                            numberOfTensors * numLeftFields*numRightFields*sizeof(float),
                            cudaMemcpyHostToDevice));
  checkCudaError(cudaFree(dev_contractionData_Right));
  checkCudaError(cudaFree(dev_contractionData_Left));
  return totalElapsedTime;
}

double
runSwitchingCudaTest(const unsigned int numberOfRepeats,
                     const unsigned int maxNumberOfCudaBlocks,
                     const unsigned int numberOfTensors,
                     const unsigned int numLeftFields,
                     const unsigned int numRightFields,
                     const unsigned int numPoints,
                     const unsigned int tens1,
                     const unsigned int tens2,
                     const unsigned int maxNumberOfTensors,
                     const unsigned int tensorSize,
                     const unsigned int memorySize,
                     const vector<float> & correctResults,
                     const ClearCacheStyle clearCacheStyle,
                     const int * const dev_junkDataToClearTheCache,
                     const unsigned int junkDataSize,
                     const vector<float> & tensorData_LayoutLeft_A,
                     const vector<float> & tensorData_LayoutLeft_B,
                     const vector<float> & tensorData_LayoutRight_A,
                     const vector<float> & tensorData_LayoutRight_B,
                     int * const dev_junkDataCounter,
                     unsigned int * const totalNumberOfRepeats,
                     float * const dev_tensorResults,
                     vector<float> * const tensorResults) {
  // if i can't saturate occupancy, do the reduction version
  // i got this number by just looking at where the plots crossed, where
  //  the reduction style actually starts beating the independent.
  if (numberOfTensors < 200) {
    const unsigned int numberOfThreadsPerBlock =
      std::min(unsigned(1024),
               unsigned(ceil(tensorSize / 32.)) * 32);
    return
      runCudaTest(CudaStyle_Reduction,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfTensors,
                  numLeftFields,
                  numRightFields,
                  numPoints,
                  tens1,
                  tens2,
                  maxNumberOfTensors,
                  tensorSize,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  tensorData_LayoutRight_A,
                  tensorData_LayoutRight_B,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_tensorResults,
                  tensorResults);
  } else {
    const unsigned int numberOfThreadsPerBlock = 1024;
    return
      runCudaTest(CudaStyle_Independent,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfTensors,
                  numLeftFields,
                  numRightFields,
                  numPoints,
                  tens1,
                  tens2,
                  maxNumberOfTensors,
                  tensorSize,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  tensorData_LayoutLeft_A,
                  tensorData_LayoutLeft_B,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_tensorResults,
                  tensorResults);
  }
}



template <class DeviceType, class KokkosJunkVector>
struct KokkosFunctor_ClearCache {

  typedef size_t     value_type;
  typedef DeviceType device_type;

  KokkosJunkVector _junkDataToClearTheCache;

  KokkosFunctor_ClearCache(KokkosJunkVector dev_junkDataToClearTheCache) :
    _junkDataToClearTheCache(dev_junkDataToClearTheCache) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int index,
                  value_type & junkDataCounter) const {
    junkDataCounter += _junkDataToClearTheCache(index);
  }

private:
  KokkosFunctor_ClearCache();

};

template <class DeviceType, class KokkosTensorData,
          class KokkosTensorResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _tensorSize;
  KokkosTensorData _data_A;
  KokkosTensorData _data_B;
  KokkosTensorResults _results;

  KokkosFunctor_Independent(const unsigned int tensorSize,
                            KokkosTensorData data_A,
                            KokkosTensorData data_B,
                            KokkosTensorResults results) :
    _tensorSize(tensorSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int tensorIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _tensorSize;
         ++entryIndex) {
      sum +=
        _data_A(tensorIndex, entryIndex) *
        _data_B(tensorIndex, entryIndex);
    }
    _results(tensorIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};


template<class DeviceType, class LeftViewType, class RightViewType, class
OutputViewType>
struct contractFieldFieldTensorFunctor {
    typedef DeviceType device_type;
    LeftViewType _leftFields;
    RightViewType _rightFields;
    OutputViewType _outputFields;
    int _numCells;
    int _numLeftFields;
    int _numRightFields;
    int _numPoints;
    int _dim1Tens;
    int _dim2Tens;

    contractFieldFieldTensorFunctor(LeftViewType leftFields, RightViewType
    rightFields, OutputViewType outputFields, int numCells, int numLeftFields, int
    numRightFields, int numPoints, int dim1Tens, int dim2Tens) :
    _leftFields(leftFields), _rightFields(rightFields),
    _outputFields(outputFields), _numCells(numCells), _numPoints(numPoints),
    _numLeftFields(numLeftFields), _numRightFields(numRightFields),
    _dim1Tens(dim1Tens), _dim2Tens(dim2Tens)
    {

    }

    // This function expects the views to be in this order:
    // left(cell, leftBasis, points, dim1Tens, dim2Tens)
    // right(cell, points, dim1Tens, dim2Tens, rightBasis)
    KOKKOS_INLINE_FUNCTION
	void operator() (const unsigned int elementIndex) const {
	    int myID = elementIndex;

	    if(myID < (_numCells * _numLeftFields * _numRightFields)) {
		// Calculating the index in the output array for this
		// thread
		int myCell = myID / (_numLeftFields * _numRightFields);
		int matrixIndex = myID % (_numLeftFields * _numRightFields);
		int lbf = matrixIndex / _numRightFields;
		int rbf = matrixIndex % _numRightFields;

		double temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    for (int iTens1 = 0; iTens1 < _dim1Tens; iTens1++) {
			for (int iTens2 = 0; iTens2 < _dim2Tens; iTens2++) {
			    temp += _leftFields(myCell, lbf, qp, iTens1, iTens2) *
				_rightFields(myCell, qp, iTens1, iTens2, rbf);
			}
		    }
		}
		_outputFields(myCell, lbf, rbf) = temp;
	    }
	}

};


template <class DeviceType, class InputViewType>
double
runKokkosTest(const unsigned int numCells,
    const unsigned int numberOfRepeats,
    const unsigned int numLeftFields,
    const unsigned int numRightFields,
    const int numPoints,
    const int tens1,
    const int tens2,
    const unsigned int memorySize,
    const vector<float> & tensorData_LayoutRight_A,
    const vector<float> & tensorData_LayoutRight_B,
    const vector<float> & correctResults,
    const string & kokkosFlavor,
    const ClearCacheStyle clearCacheStyle,
    const vector<int> & junkDataToClearTheCache,
    size_t * junkDataCounter,
    unsigned int * const totalNumberOfRepeats,
    vector<float> * results,
    KokkosStyle kokkosStyle,
    const unsigned int tile_size) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename InputViewType::HostMirror     Kokkos_input_Host;
  typedef Kokkos::View<float***, Kokkos::LayoutRight, DeviceType>              KokkosResults;
  typedef typename KokkosResults::HostMirror  KokkosResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;


  int p=numPoints, t1=tens1, t2=tens2;

  // These are indices into the arrays and should not be
  // changed unless you change the order of the indices
  // as well
  int cLOff = numLeftFields*p*t1*t2;
  int cROff = numRightFields*p*t1*t2;
  int basisOff = p*t1*t2;
  int pLOff = t1*t2;
  int pROff = t1*t2;
  int tROff = t2;
  int t2ROff = 1;
  int tOff = t2;

  InputViewType dev_kokkosData_Right("kokkos data A",
      numCells,
      p,
      t1,
      t2,
      numRightFields);

  Kokkos_input_Host kokkosData_Right =
    Kokkos::create_mirror_view(dev_kokkosData_Right);

  InputViewType dev_kokkosData_Left("kokkos data B",
      numCells,
      numLeftFields,
      p,
      t1,
      t2);

  Kokkos_input_Host kokkosData_Left =
    Kokkos::create_mirror_view(dev_kokkosData_Left);

  KokkosResults dev_kokkosResults("kokkos dot product results",
                                                      numCells,
						      numLeftFields,
						      numRightFields);
  KokkosResults_Host kokkosResults =
    Kokkos::create_mirror_view(dev_kokkosResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);

  // copy the data into the device views and ship them over
  /* for (unsigned int tensorIndex = 0;
       tensorIndex < numberOfTensors; ++tensorIndex) {
    for (unsigned int entryIndex = 0;
         entryIndex < tensorSize; ++entryIndex) {
      kokkosTensorData_A(tensorIndex, entryIndex) =
        tensorData_LayoutRight_A[tensorIndex * tensorSize +
                                     entryIndex];
      kokkosTensorData_B(tensorIndex, entryIndex) =
        tensorData_LayoutRight_B[tensorIndex * tensorSize +
                                     entryIndex];
    }
  } */
  for (int cl = 0; cl < numCells; ++cl) {
    for (int qp = 0; qp < p; ++qp) {
      for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
          for(int rbf = 0; rbf < numRightFields; ++rbf) {
            kokkosData_Right(cl, qp, iTens1, iTens2, rbf) =
              tensorData_LayoutRight_A[cl*cROff + rbf*basisOff + qp*pROff +
              iTens1*tROff + iTens2*t2ROff];
          }
          for(int lbf = 0; lbf < numLeftFields; ++lbf) {
            kokkosData_Left(cl, lbf, qp, iTens1, iTens2) =
              tensorData_LayoutRight_B[cl*cLOff + lbf*basisOff + qp*pLOff +
              iTens1*tOff + iTens2];
          }
        }
      }
    }
  }


  Kokkos::deep_copy(dev_kokkosData_Right, kokkosData_Right);
  Kokkos::deep_copy(dev_kokkosData_Left, kokkosData_Left);

  // copy the data into the device views and ship them over
  for (unsigned int junkDataIndex = 0;
       junkDataIndex < junkDataSize; ++junkDataIndex) {
    kokkosJunkDataToClearTheCache(junkDataIndex) =
      junkDataToClearTheCache[junkDataIndex];
  }
  Kokkos::deep_copy(dev_kokkosJunkDataToClearTheCache, kokkosJunkDataToClearTheCache);

  KokkosFunctor_ClearCache<DeviceType,
                           KokkosJunkVector>
    kokkosFunctor_ClearCache(dev_kokkosJunkDataToClearTheCache);

  double totalElapsedTime = 0;
  if (kokkosStyle == KokkosStyle_Independent)
  {
    // breaking formatting convention because holy freak that's long
    contractFieldFieldTensorFunctor<DeviceType,
      InputViewType,
      InputViewType,
      KokkosResults>
        tensorFunctor(dev_kokkosData_Left,
            dev_kokkosData_Right,
            dev_kokkosResults,
            numCells,
            numLeftFields,
            numRightFields,
            p,
            t1,
            t2);

    timespec tic;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats+1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }

      Kokkos::parallel_for(numCells*numRightFields*numLeftFields, tensorFunctor);

      // wait for this repeat's results to finish
      Kokkos::fence();
      if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
        const timespec toc = getTimePoint();
        const float elapsedTime = getElapsedTime(tic, toc);
        totalElapsedTime += elapsedTime;

        // attempt to scrub all levels of cache
        size_t partialJunkDataCounter = 0;
        Kokkos::parallel_reduce(junkDataSize, kokkosFunctor_ClearCache,
            partialJunkDataCounter);
        *junkDataCounter += partialJunkDataCounter;
      }
    }
    if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
      const timespec toc = getTimePoint();
      totalElapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
    }
  // endif kokkosStyle == Independent
  } else if (kokkosStyle == KokkosStyle_Tiling) {
    CFFS_Tiling_TeamFunctor_1D<InputViewType,
      InputViewType, KokkosResults>
        tensorFunctor(numCells,
            numLeftFields,
            numRightFields,
            p,
            t1,
            t2,
            dev_kokkosData_Left,
            dev_kokkosData_Right,
            dev_kokkosResults,
            tile_size);

    const unsigned int numBlocks = numCells *
      (((numLeftFields - 1)/tile_size)+1) * (((numRightFields -1)/tile_size)+1);

    const team_policy tiling_policy(numBlocks,tile_size*tile_size);


    timespec tic;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats+1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }

      Kokkos::parallel_for(tiling_policy, tensorFunctor);

      // wait for this repeat's results to finish
      Kokkos::fence();
      if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
        const timespec toc = getTimePoint();
        const float elapsedTime = getElapsedTime(tic, toc);
        totalElapsedTime += elapsedTime;

        // attempt to scrub all levels of cache
        size_t partialJunkDataCounter = 0;
        Kokkos::parallel_reduce(junkDataSize, kokkosFunctor_ClearCache,
            partialJunkDataCounter);
        *junkDataCounter += partialJunkDataCounter;
      }
    }
    if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
      const timespec toc = getTimePoint();
      totalElapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
    }

  }


void contractFieldFieldTensorSerial(vector<float> & outputFields,
                                    vector<float> &  leftFields,
                                    vector<float> & rightFields,
                                    const bool sumInto,
				    int numCells,
				    int numLeftFields,
				    int numRightFields,
				    int numPoints,
				    int dim1Tensor,
				    int dim2Tensor) {
    /* This function expects the left and right arrays to be in the order of
     * (cell, left or right, points, dim1Tens, dim2Tens). That is the way
     * the indexing is calculated.
     */

    if (sumInto) {
	for (int cl = 0; cl < numCells; cl++) {
	    // Need to index into the different arrays, so I am doing the
	    // calculation once here
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    float tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] += tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
    }
    // This is exactly the same as above but outputfields is set equal
    // to temp instead of += temp
    else {
	for (int cl = 0; cl < numCells; cl++) {
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    float tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] = tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
   }
} // end contractFieldFieldTensor



int main(int argc, char* argv[]) {

#ifdef ENABLE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

  // ===============================================================
  // ********************** < input> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const vector<unsigned int> tensorSizes =
    {{160, 320, 1600, 6400}};
  const array<float, 2> memorySizeExtrema = {{1e7, 1e8}};
  const unsigned int numberOfMemorySizes = 10;
  const unsigned int maxNumberOfCudaBlocks = unsigned(1e4);
  const ClearCacheStyle clearCacheStyle =
    ClearCacheAfterEveryRepeat;
  const unsigned int numberOfRepeats =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? 5 : 250;
  const string machineName = "shadowfax";
  const string prefix = "data/ContractFieldFieldTensor_";
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </input> ******************************
  // ===============================================================

  // derive some values from the inputs
  const unsigned int numberOfTensorSizes = tensorSizes.size();
  const string clearCacheStyleString =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? "clearCache" :
    "dontClearCache";
  const string suffix = "_" + clearCacheStyleString + "_" + machineName;

  // create the actual sizes
  vector<unsigned int> memorySizes(numberOfMemorySizes);
  for (unsigned int memorySizeIndex = 0;
       memorySizeIndex < numberOfMemorySizes; ++memorySizeIndex) {
    const float percent = memorySizeIndex / float(numberOfMemorySizes - 1);
    const float minLog = log10(memorySizeExtrema[0]);
    const float maxLog = log10(memorySizeExtrema[1]);
    const float thisLog = minLog + percent * (maxLog - minLog);
    const unsigned int maxTensorSize = tensorSizes.back();
    // memory size is linear on a log scale, but rounded to a multiple of the
    //  largest dot product size
    const unsigned int desiredMemorySizeInBytes = pow(10., thisLog);
    // now, in this amount of memory i have to fit two vectors of data
    // that are multiples of the max dot product size
    const unsigned int memorySizeInBytes =
      unsigned(desiredMemorySizeInBytes /
               float(4 * sizeof(float) * maxTensorSize)) *
      4 * sizeof(float) * maxTensorSize;
    memorySizes[memorySizeIndex] = memorySizeInBytes;
  }

  // create a c++11 random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  // these are just containers for storing the numbers we'll be plotting.
  // i feel a little dirty using a vector<vector>, but i don't want to introduce
  //  a dependence on eigen or something for a real matrix.
  vector<vector<float> >
    tensorSizeMatrix(numberOfTensorSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    numberOfTensorsMatrix(numberOfTensorSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    memorySizeMatrix(numberOfTensorSizes,
                     vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    serialTimesMatrix(numberOfTensorSizes,
                      vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    ompTimesMatrix(numberOfTensorSizes,
                   vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaIndependent_TimesMatrix(numberOfTensorSizes,
                                vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaReduction_TimesMatrix(numberOfTensorSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaSwitchingTimesMatrix(numberOfTensorSizes,
                             vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    cudaSlicingTimesMatrix(numberOfTensorSizes,
                            vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    cudaAdaptiveSlicingTimesMatrix(numberOfTensorSizes,
                            vector<float>(numberOfMemorySizes, 0));

#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfTensorSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfTensorSizes,
                                     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaTilingTimesMatrix(numberOfTensorSizes,
                                     vector<float>(numberOfMemorySizes, 0));
#endif


  // create some junk data to use in clearing the cache
  size_t junkDataCounter = 0;
  const size_t junkDataSize = 1e7;
  vector<int> junkDataToClearTheCache(junkDataSize, 0);
  for (unsigned int i = 0; i < junkDataSize/100; ++i) {
    junkDataToClearTheCache[(rand() / float(RAND_MAX))*junkDataSize] = 1;
  }
  int * dev_junkDataToClearTheCache;
  checkCudaError(cudaMalloc((void **) &dev_junkDataToClearTheCache,
                            junkDataSize * sizeof(int)));
  checkCudaError(cudaMemcpy(dev_junkDataToClearTheCache,
                            &junkDataToClearTheCache[0],
                            junkDataSize * sizeof(int),
                            cudaMemcpyHostToDevice));
  int * dev_junkDataCounter;
  checkCudaError(cudaMalloc((void **) &dev_junkDataCounter,
                            sizeof(int)));
  {
    int temp = 0;
    checkCudaError(cudaMemcpy(dev_junkDataCounter,
                              &temp,
                              sizeof(int),
                              cudaMemcpyHostToDevice));
  }

  unsigned int totalNumberOfRepeats = 0;

  // for each dot product size
  for (unsigned int tensorSizeIndex = 0;
	tensorSizeIndex < numberOfTensorSizes;
       ++tensorSizeIndex) {
    const unsigned int tensorSize = tensorSizes[tensorSizeIndex];

    const timespec thisSizesTic = getTimePoint();

    // these must be the same size or else the allocation for vectors won't
    // work
    const int numLeftFields = 10;
    const int numRightFields = 10;
    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumberOfTensors =
      memorySizes.back() / 4 / sizeof(float) / tensorSize;
    vector<float> tensorData_LayoutRight_A(maxNumberOfTensors * tensorSize);
    vector<float> tensorData_LayoutRight_B(tensorData_LayoutRight_A.size());
    vector<float> tensorData_LayoutLeft_A(tensorData_LayoutRight_A.size());
    vector<float> tensorData_LayoutLeft_B(tensorData_LayoutRight_B.size());
    for (unsigned int tensorIndex = 0;
         tensorIndex < maxNumberOfTensors; ++tensorIndex) {
      for (unsigned int entryIndex = 0;
           entryIndex < tensorSize; ++entryIndex) {

        const unsigned int layoutRightIndex =
          tensorIndex * tensorSize + entryIndex;
        tensorData_LayoutRight_A[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        tensorData_LayoutRight_B[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        const unsigned int layoutLeftIndex =
          entryIndex * maxNumberOfTensors + tensorIndex;
        tensorData_LayoutLeft_A[layoutLeftIndex] =
          tensorData_LayoutRight_A[layoutRightIndex];
        tensorData_LayoutLeft_B[layoutLeftIndex] =
          tensorData_LayoutRight_B[layoutRightIndex];
      }
    }
    vector<float>
    tensorResults(maxNumberOfTensors*numLeftFields*numRightFields,
                                    std::numeric_limits<float>::quiet_NaN());


    // now, because we'll be working with cuda stuff, also allocate the inputs
    //  and output on the gpu and copy them over
    /*
    float * dev_tensorData_LayoutRight_A;
    checkCudaError(cudaMalloc((void **) &dev_tensorData_LayoutRight_A,
                              maxNumberOfTensors * tensorSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_tensorData_LayoutRight_A,
                              &tensorData_LayoutRight_A[0],
                              maxNumberOfTensors * tensorSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_tensorData_LayoutRight_B;
    checkCudaError(cudaMalloc((void **) &dev_tensorData_LayoutRight_B,
                              maxNumberOfTensors * tensorSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_tensorData_LayoutRight_B,
                              &tensorData_LayoutRight_B[0],
                              maxNumberOfTensors * tensorSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    */
    float * dev_tensorResults;
    checkCudaError(cudaMalloc((void **) &dev_tensorResults,
                              maxNumberOfTensors *numLeftFields*numRightFields*sizeof(float)));
    checkCudaError(cudaMemcpy(dev_tensorResults, &tensorResults[0],
                              maxNumberOfTensors*numLeftFields*numRightFields*sizeof(float),
                              cudaMemcpyHostToDevice));
    /*
    // make and populate the LayoutLeft versions
    float * dev_tensorData_LayoutLeft_A;
    checkCudaError(cudaMalloc((void **) &dev_tensorData_LayoutLeft_A,
                              maxNumberOfTensors * tensorSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_tensorData_LayoutLeft_A,
                              &tensorData_LayoutLeft_A[0],
                              maxNumberOfTensors * tensorSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_tensorData_LayoutLeft_B;
    checkCudaError(cudaMalloc((void **) &dev_tensorData_LayoutLeft_B,
                              maxNumberOfTensors * tensorSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_tensorData_LayoutLeft_B,
                              &tensorData_LayoutLeft_B[0],
                              maxNumberOfTensors * tensorSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    */
    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex];
      const unsigned int numberOfTensors =
        memorySize / 4 / sizeof(float) / tensorSize / numLeftFields;

      const int tens1 = 4;
      const int tens2 = 4;
      const int numPoints = tensorSize/(tens1*tens2);

      // ===============================================================
      // ********************** < do serial> ***************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        timespec tic;
        for (unsigned int repeatIndex = 0;
             repeatIndex < numberOfRepeats+1; ++repeatIndex) {
          ++totalNumberOfRepeats;
          if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
               repeatIndex == 1) ||
              clearCacheStyle == ClearCacheAfterEveryRepeat) {
            tic = getTimePoint();
          }
          // do the actual calculation
          contractFieldFieldTensorSerial(tensorResults,
	  tensorData_LayoutRight_B, tensorData_LayoutRight_A, false,
	  numberOfTensors, numLeftFields, numRightFields, numPoints, tens1, tens2);
          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            serialTimesMatrix[tensorSizeIndex][memorySizeIndex] += elapsedTime;

            junkDataCounter +=
              std::accumulate(junkDataToClearTheCache.begin(),
                              junkDataToClearTheCache.end(), size_t(0));
          }
        }
        if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
          const timespec toc = getTimePoint();
          const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
          serialTimesMatrix[tensorSizeIndex][memorySizeIndex] = elapsedTime;
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do serial> ***************************
      // ===============================================================
      const vector<float> correctResults = tensorResults;
      // scrub the results
      std::fill(tensorResults.begin(),
                tensorResults.end(),
                std::numeric_limits<float>::quiet_NaN());

      // ===============================================================
      // ********************** < do omp> ******************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        timespec tic;
        for (unsigned int repeatIndex = 0;
             repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
          ++totalNumberOfRepeats;
          if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
               repeatIndex == 1) ||
              clearCacheStyle == ClearCacheAfterEveryRepeat) {
            tic = getTimePoint();
          }
          // do the actual calculation
#pragma omp parallel for default(none)                                  \
  shared(tensorData_LayoutRight_A, tensorData_LayoutRight_B,    \
         tensorResults)

	  for (unsigned int elementId = 0;
		elementId <  numberOfTensors * numLeftFields * numRightFields;
		elementId++) {
		int myCell = elementId / (numLeftFields * numRightFields);
		int matrixIndex = elementId % (numLeftFields*numRightFields);
		int lbf = matrixIndex / numRightFields;
		int rbf = matrixIndex % numRightFields;

		float temp = 0;
		for (int qp = 0; qp < numPoints; qp++) {
		    for (int iTens1 = 0; iTens1 < tens1; iTens1++) {
			for (int iTens2 = 0; iTens2 < tens2; iTens2++) {
			    temp +=
			    tensorData_LayoutRight_B.at(myCell*numLeftFields*numPoints*tens1*tens2
			    + lbf*numPoints*tens1*tens2 + qp*tens1*tens2 +
			    iTens1*tens2 + iTens2) *
				tensorData_LayoutRight_A.at(myCell*numRightFields*numPoints*tens1*tens2
				+ rbf*numPoints*tens1*tens2 + qp*tens1*tens2 +
				iTens1*tens2 + iTens2);
			}
		    }
		}
		tensorResults.at(myCell*numLeftFields*numRightFields +
		lbf*numRightFields + rbf) = temp;
	    }


	  /*
	  for (unsigned int tensorIndex = 0;
               tensorIndex < numberOfTensors;
               ++tensorIndex) {
            const unsigned int shortcutIndex = tensorIndex * tensorSize;
            float sum = 0;
            for (unsigned int entryIndex = 0;
                 entryIndex < tensorSize; ++entryIndex) {
              sum +=
                tensorData_LayoutRight_A[shortcutIndex + entryIndex] *
                tensorData_LayoutRight_B[shortcutIndex + entryIndex];
            }
            tensorResults[tensorIndex] = sum;
          }*/

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompTimesMatrix[tensorSizeIndex][memorySizeIndex] += elapsedTime;

            // attempt to scrub all levels of cache
#pragma omp parallel default(none)                      \
  shared(junkDataCounter, junkDataToClearTheCache)
            {
              const size_t thisThreadsJunkDataCounter =
                std::accumulate(junkDataToClearTheCache.begin(),
                                junkDataToClearTheCache.end(), size_t(0));
              // only one thread adds the junk counter so that the total
              //  at the end is not a function of the number of threads.
#pragma omp single
              junkDataCounter += thisThreadsJunkDataCounter;
            }
          }
        }
        if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
          const timespec toc = getTimePoint();
          const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
          ompTimesMatrix[tensorSizeIndex][memorySizeIndex] = elapsedTime;
          // check the results
          checkAnswer(correctResults, tensorResults,
                      tensorSize, memorySize,
                      string("omp"));
        }

      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp> ******************************
      // ===============================================================

      // scrub the results
      std::fill(tensorResults.begin(),
                tensorResults.end(),
                std::numeric_limits<float>::quiet_NaN());

      // ===============================================================
      // ***************** < do cuda independent> **********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      {
        const unsigned int numberOfThreadsPerBlock = 256;
        cudaIndependent_TimesMatrix[tensorSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Independent,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfTensors,
                      numLeftFields,
                      numRightFields,
                      numPoints,
                      tens1,
                      tens2,
                      maxNumberOfTensors,
                      tensorSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      tensorData_LayoutRight_A,
                      tensorData_LayoutRight_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_tensorResults,
                      &tensorResults);
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda independent> **********************
      // ===============================================================

      // scrub the results
      std::fill(tensorResults.begin(),
                tensorResults.end(),
                std::numeric_limits<float>::quiet_NaN());

      // ===============================================================
      // ***************** < do cuda slicing> **********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
      const unsigned int numberOfThreadsPerBlock = numRightFields;
      cudaSlicingTimesMatrix[tensorSizeIndex][memorySizeIndex] =
        runCudaTeamTest(CudaStyle_Slicing,
            numberOfThreadsPerBlock,
            numberOfRepeats,
            maxNumberOfCudaBlocks,
            numberOfTensors,
            numLeftFields,
            numRightFields,
            numPoints,
            tens1,
            tens2,
            maxNumberOfTensors,
            tensorSize,
            memorySize,
            correctResults,
            clearCacheStyle,
            dev_junkDataToClearTheCache,
            junkDataSize,
            tensorData_LayoutRight_A,
            tensorData_LayoutRight_B,
            dev_junkDataCounter,
            &totalNumberOfRepeats,
            dev_tensorResults,
            &tensorResults);
      }
	/*
      {
        //HARDCODED TO NUMRIGHTFIELDS*2 FOR NOW
        //In theory this should be dynamically adjusted to be around
        //256, since we seem to get best results when the
        //numberOfThreadsPerBlock is around 256. I want to sanity check
        //that this will actually create good performance by trying
        //something in that ballpark and since numrightfields = 125 for
        //the current example that's good.
      const unsigned int numberOfThreadsPerBlock = numLeftFields*2;

      cudaAdaptiveSlicingTimesMatrix[tensorSizeIndex][memorySizeIndex] =
        runCudaTeamTest(CudaStyle_AdaptiveSlicing,
            numberOfThreadsPerBlock,
            numberOfRepeats,
            maxNumberOfCudaBlocks,
            numberOfTensors,
            numLeftFields,
            numRightFields,
            numPoints,
            tens1,
            tens2,
            maxNumberOfTensors,
            tensorSize,
            memorySize,
            correctResults,
            clearCacheStyle,
            dev_junkDataToClearTheCache,
            junkDataSize,
            tensorData_LayoutRight_A,
            tensorData_LayoutRight_B,
            dev_junkDataCounter,
            &totalNumberOfRepeats,
            dev_tensorResults,
            &tensorResults);

      }
	*/
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda slicing> **********************
      // ===============================================================
      #if 0
      // ===============================================================
      // ***************** < do cuda reductions> ***********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock =
          std::min(unsigned(1024),
                   unsigned(ceil(tensorSize / 32.)) * 32);

        cudaReduction_TimesMatrix[tensorSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Reduction,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfTensors,
                      maxNumberOfTensors,
                      tensorSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_tensorData_LayoutRight_A,
                      dev_tensorData_LayoutRight_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_tensorResults,
                      &tensorResults);

      }
      cudaSwitchingTimesMatrix[tensorSizeIndex][memorySizeIndex] =
        runSwitchingCudaTest(numberOfRepeats,
                             maxNumberOfCudaBlocks,
                             numberOfTensors,
                             numLeftFields,
                             numRightFields,
                             numPoints,
                             tens1,
                             tens2,
                             maxNumberOfTensors,
                             tensorSize,
                             memorySize,
                             correctResults,
                             clearCacheStyle,
                             dev_junkDataToClearTheCache,
                             junkDataSize,
                             dev_tensorData_LayoutLeft_A,
                             dev_tensorData_LayoutLeft_B,
                             dev_tensorData_LayoutRight_A,
                             dev_tensorData_LayoutRight_B,
                             dev_junkDataCounter,
                             &totalNumberOfRepeats,
                             dev_tensorResults,
                             &tensorResults);
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda reductions> ***********************
      // ===============================================================
#endif

#ifdef ENABLE_KOKKOS
      // ===============================================================
      // ***************** < do kokkos> ********************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float*****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosData;

        kokkosOmpTimesMatrix[tensorSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosData>(numberOfTensors,
                                              numberOfRepeats,
                                              numLeftFields,
					      numRightFields,
					      numPoints,
					      tens1,
					      tens2,
                                              memorySize,
				              tensorData_LayoutRight_A,
                                              tensorData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &tensorResults,
                                              KokkosStyle_Independent,
                                              0);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float*****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaIndependentTimesMatrix[tensorSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosData>(numberOfTensors,
                                              numberOfRepeats,
                                              numLeftFields,
					      numRightFields,
					      numPoints,
					      tens1,
					      tens2,
                                              memorySize,
                                              tensorData_LayoutRight_A,
                                              tensorData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &tensorResults,
                                              KokkosStyle_Independent,
                                              0);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float*****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTilingTimesMatrix[tensorSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosData>(numberOfTensors,
                                              numberOfRepeats,
                                              numLeftFields,
					      numRightFields,
					      numPoints,
					      tens1,
					      tens2,
                                              memorySize,
                                              tensorData_LayoutRight_A,
                                              tensorData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &tensorResults,
                                              KokkosStyle_Tiling,
                                              16);
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
#endif // ENABLE_KOKKOS

      tensorSizeMatrix[tensorSizeIndex][memorySizeIndex] =
        tensorSize;
      numberOfTensorsMatrix[tensorSizeIndex][memorySizeIndex] =
        numberOfTensors;
      memorySizeMatrix[tensorSizeIndex][memorySizeIndex] =
        memorySize;

    }

    const timespec thisSizesToc = getTimePoint();
    const float thisSizesElapsedTime =
      getElapsedTime(thisSizesTic, thisSizesToc);
    printf("completed %4u repeats of dot products of size %4u "
           "in %7.2f seconds\n", numberOfRepeats,
           tensorSize, thisSizesElapsedTime);
    /*
    checkCudaError(cudaFree(dev_tensorData_LayoutLeft_A));
    checkCudaError(cudaFree(dev_tensorData_LayoutLeft_B));
    checkCudaError(cudaFree(dev_tensorData_LayoutRight_A));
    checkCudaError(cudaFree(dev_tensorData_LayoutRight_B));
    checkCudaError(cudaFree(dev_tensorResults));
    */
  }
  printf("finished, starting to write\n");
  writeTimesMatrixToFile(tensorSizeMatrix,
                         prefix + string("tensorSize") + suffix);
  writeTimesMatrixToFile(numberOfTensorsMatrix,
                         prefix + string("numberOfTensors") + suffix);
  writeTimesMatrixToFile(memorySizeMatrix,
                         prefix + string("memorySize") + suffix);
  writeTimesMatrixToFile(serialTimesMatrix,
                         prefix + string("serialTimes") + suffix);
  writeTimesMatrixToFile(ompTimesMatrix,
                         prefix + string("ompTimes") + suffix);
  writeTimesMatrixToFile(cudaIndependent_TimesMatrix,
                         prefix + string("cudaIndependentTimes") + suffix);
  writeTimesMatrixToFile(cudaReduction_TimesMatrix,
                         prefix + string("cudaReductionTimes") + suffix);
  writeTimesMatrixToFile(cudaSwitchingTimesMatrix,
                         prefix + string("cudaSwitchingTimes") + suffix);
  writeTimesMatrixToFile(cudaSlicingTimesMatrix,
                         prefix + string("cudaSlicingTimes") + suffix);
  writeTimesMatrixToFile(cudaAdaptiveSlicingTimesMatrix,
                         prefix + string("cudaAdaptiveSlicingTimes") + suffix);
#ifdef ENABLE_KOKKOS
  writeTimesMatrixToFile(kokkosOmpTimesMatrix,
                         prefix + string("kokkosOmpTimes") + suffix);
  writeTimesMatrixToFile(kokkosCudaIndependentTimesMatrix,
                         prefix + string("kokkosCudaIndependentTimes") + suffix);

  writeTimesMatrixToFile(kokkosCudaTilingTimesMatrix,
                         prefix + string("kokkosCudaTilingTimes") + suffix);
#endif

  printf("done writing\n");

  const size_t junkDataSum =
    std::accumulate(junkDataToClearTheCache.begin(),
                    junkDataToClearTheCache.end(), size_t(0));
  {
    int temp = 0;
    checkCudaError(cudaMemcpy(&temp,
                              dev_junkDataCounter,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    junkDataCounter += temp;
  }
  if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
    const size_t expectedDataCounter = 0;
    if (junkDataCounter != expectedDataCounter) {
      fprintf(stderr, "for DontClearCacheAfterEveryRepeat, invalid "
              "junkDataCounter = %zu (%e), it should be %zu (%e)\n",
              junkDataCounter, float(junkDataCounter),
              expectedDataCounter, float(expectedDataCounter));
      exit(1);
    }
  }

#ifdef ENABLE_KOKKOS
  Kokkos::finalize();
#endif
    exit(1);


  return 0;
}
