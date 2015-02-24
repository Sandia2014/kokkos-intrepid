// -*- C++ -*-
// ArrayOfContractions.cc
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

//#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
typedef Kokkos::DefaultExecutionSpace Device;
typedef Kokkos::HostSpace::execution_space Host;
typedef Kokkos::TeamPolicy<Device> team_policy;
typedef team_policy::member_type team_member;
//#endif // ENABLE_KOKKOS




enum CudaStyle {CudaStyle_Independent,
                CudaStyle_Reduction,
                CudaStyle_Slicing,
                CudaStyle_Tiling};

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
  case CudaStyle_Tiling:
    return string("CudaStyle_Tiling");
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
doCudaContractions_Independent_kernel(const unsigned int numberOfContractions,
                                     const unsigned int maxNumberOfContractions,
                                     const unsigned int contractionSize,
                                     const unsigned int numBasis,
                                     const float * const __restrict__ dev_contractionData_LayoutLeft_Right,
                                     const float * const __restrict__ dev_contractionData_LayoutLeft_Left,
                                     float * dev_contractionResults) {

  unsigned int contractionIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (contractionIndex < numberOfContractions) {

    int myID = contractionIndex;
    int myMatrix = myID / (numBasis * numBasis);
    int matrixIndex = myID % (numBasis * numBasis);

    int matrixRow = matrixIndex / numBasis;
    int matrixCol = matrixIndex % numBasis;

    float temp = 0;
    for (int qp = 0; qp < contractionSize; qp++) {
      temp += dev_contractionData_LayoutLeft_Left[myMatrix * numBasis * contractionSize + qp * numBasis + matrixRow]
      * dev_contractionData_LayoutLeft_Right[myMatrix * numBasis * contractionSize + qp*numBasis + matrixCol];
    }

    dev_contractionResults[myMatrix * numBasis * numBasis + matrixRow * numBasis + matrixCol] = temp;
    contractionIndex += blockDim.x * gridDim.x;
  }
}

__global__
void
doCudaContractions_Slicing_kernel(const unsigned int numCells,
                                     const unsigned int contractionSize,
                                     const unsigned int numBasis,
                                     const float * const __restrict__ dev_contractionData_Right,
                                     const float * const __restrict__ dev_contractionData_Left,
                                     float * dev_contractionResults) {

  extern __shared__ float sliceStorage[];

  unsigned int globalRowIndex = blockIdx.x;
  unsigned int col = threadIdx.x;

  while (globalRowIndex < numCells * numBasis){

    int myMatrix = globalRowIndex / (numBasis * contractionSize);
    int localRowIndex = globalRowIndex % (numBasis * contractionSize);

    for(int i = threadIdx.x; i < contractionSize; i += blockDim.x) {
      sliceStorage[i] = dev_contractionData_Left[myMatrix*numBasis*contractionSize
                              + localRowIndex * contractionSize + i];
    }

    syncthreads();

    float temp = 0;

    for (int qp = 0; qp < contractionSize; qp++) {
      temp += sliceStorage[qp]
      * dev_contractionData_Right[myMatrix * numBasis * contractionSize + qp * numBasis + col];
    }

    dev_contractionResults[myMatrix * numBasis * numBasis + localRowIndex * numBasis + col] = temp;
    globalRowIndex += gridDim.x;
  }
}

__global__
void
doCudaContractions_Tiling_kernel(const unsigned int numCells,
                                 const unsigned int contractionSize,
                                 const unsigned int tileSize,
                                 const unsigned int numBasis,
                                 const float * const __restrict__ dev_contractionData_Right,
                                 const float * const __restrict__ dev_contractionData_Left,
                                 float * dev_contractionResults) {

  extern __shared__ float tileStorage[];

  const unsigned int numbersPerTile = tileSize * tileSize;
  //NOTE: This relies on contractionSize being a multiple of tileSize (16)
  const unsigned int numberOfHorizontalTiles = contractionSize / tileSize;
  //NOTE: This relies on numBasis being a multiple of tileSize(16)
  const unsigned int numberOfVerticalTiles = numBasis / tileSize;

  const unsigned int numberOfTiles = numCells * numberOfVerticalTiles * numberOfVerticalTiles;

  const unsigned int subRow = threadIdx.x / tileSize;
  const unsigned int subCol = threadIdx.x  - subRow * tileSize;

  unsigned int resultTileIndex = blockIdx.x;

  while (resultTileIndex < numberOfTiles) {

    unsigned int resultSubmatrixIndex = resultTileIndex % (numberOfVerticalTiles * numberOfVerticalTiles);
    unsigned int resultMatrix = resultTileIndex / (numberOfVerticalTiles * numberOfVerticalTiles);

    // for tileNumber in 0...numberOfTilesPerSide
    for (unsigned int tileNumber = 0;
       tileNumber < numberOfHorizontalTiles; ++tileNumber) {
         // calculate result tile indices

         const unsigned int resultTileRow = resultSubmatrixIndex / numberOfHorizontalTiles;
         const unsigned int resultTileCol = resultSubmatrixIndex  -
          resultTileRow * numberOfHorizontalTiles;

          // calculate this threads actual output index
          const unsigned int row = resultTileRow * tileSize + subRow;
          const unsigned int col = resultTileCol * tileSize + subCol;

          // these are base indices into the shared memory
          const unsigned int leftBaseIndex = subRow * tileSize;
          const unsigned int rightBaseIndex = numbersPerTile + subCol;

          const unsigned int resultIndex = row * numBasis + col;

          // load the left and right tiles into shared memory
          syncthreads();
          tileStorage[threadIdx.x]              = dev_contractionData_Left[resultMatrix * numBasis * contractionSize
                                                  + row * contractionSize + tileNumber * tileSize + subCol];
          tileStorage[threadIdx.x + blockDim.x] = dev_contractionData_Right[resultMatrix * numBasis * contractionSize
                                                  + (tileNumber * tileSize + subRow) * numBasis + col];
          // make sure everyone's finished loading their pieces of the tiles
          syncthreads();

          double sum = 0;
          for (unsigned int dummy = 0; dummy < tileSize; ++dummy) {
            sum +=
            tileStorage[leftBaseIndex + dummy] *
            tileStorage[rightBaseIndex + dummy * tileSize];
          }
          dev_contractionResults[resultIndex] += sum;
    }
    resultTileIndex += gridDim.x;
  }

}
__global__
void
doCudaContractions_Reduction_kernel(const unsigned int numberOfContractions,
                                   const unsigned int contractionSize,
                                   const float * const __restrict__ dev_contractionData_LayoutRight_Right,
                                   const float * const __restrict__ dev_contractionData_LayoutRight_Left,
                                   float * dev_contractionResults) {

  extern __shared__ float sharedMemory[];

  unsigned int contractionIndex = blockIdx.x;
  while (contractionIndex < numberOfContractions) {

    // goal: compute the contribution to the dot product from this thread
    const unsigned int shortcutIndex = contractionIndex * contractionSize;
    float partialSum = 0;
    unsigned int entryIndex = threadIdx.x;
    while (entryIndex < contractionSize) {
      const unsigned int index = shortcutIndex + entryIndex;
      partialSum +=
        dev_contractionData_LayoutRight_Right[index] *
        dev_contractionData_LayoutRight_Left[index];
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
      atomicAdd(&dev_contractionResults[contractionIndex], partialSum);
    }

    // move on to the next dot product
    contractionIndex += gridDim.x;
  }
}

void
writeTimesMatrixToFile(const vector<vector<float> > & times,
                       const string filename) {

  const unsigned int numberOfContractionSizes = times.size();
  // yeah, yeah, kinda unsafe
  const unsigned int numberOfMemorySizes = times[0].size();
  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%s.csv", filename.c_str());
  FILE* file = fopen(sprintfBuffer, "w");
  for (unsigned int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      if (memorySizeIndex > 0) {
        fprintf(file, ", ");
      }
      fprintf(file, "%10.4e", times[contractionSizeIndex][memorySizeIndex]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

void
checkAnswer(const vector<float> & correctResults,
            const vector<float> & contractionResults,
            const unsigned int contractionSize,
            const unsigned int memorySize,
            const string flavorName) {
  for (unsigned int contractionIndex = 0;
       contractionIndex < correctResults.size();
       ++contractionIndex) {
    if (std::abs(correctResults[contractionIndex] -
                 contractionResults[contractionIndex]) /
        std::abs(correctResults[contractionIndex]) > 1e-4) {
      fprintf(stderr, "invalid answer for dot product index %u for "
              "flavor %s, "
              "should be %e but we have %e, "
              "contractionSize = %u, memorySize = %8.2e\n",
              contractionIndex, flavorName.c_str(),
              correctResults[contractionIndex],
              contractionResults[contractionIndex],
              contractionSize, float(memorySize));
      exit(1);
    }
  }
}

double
runCudaTest(const CudaStyle cudaStyle,
            const unsigned int numberOfThreadsPerBlock,
            const unsigned int numberOfRepeats,
            const unsigned int maxNumberOfCudaBlocks,
            const unsigned int numberOfContractions,
            const unsigned int maxNumberOfContractions,
            const unsigned int contractionSize,
            const unsigned int numBasis,
            const unsigned int memorySize,
            const vector<float> & correctResults,
            const ClearCacheStyle clearCacheStyle,
            const int * const dev_junkDataToClearTheCache,
            const unsigned int junkDataSize,
            const vector<float> & contractionData_Right,
            const vector<float> & contractionData_Left,
            int * const dev_junkDataCounter,
            unsigned int * const totalNumberOfRepeats,
            float * const dev_contractionResults,
            vector<float> * const contractionResults) {
  const unsigned int numberOfBlocks =
    min(maxNumberOfCudaBlocks,
        (unsigned int)ceil(numberOfContractions*numBasis*numBasis/float(numberOfThreadsPerBlock)));

    // Format the data the way we want and then copy it to the GPU
    vector<float> contractionData_GPURight(contractionData_Right.size());
    vector<float> contractionData_GPULeft(contractionData_Right.size());

    for (int cl = 0; cl < numberOfContractions; ++cl) {
      for (int qp = 0; qp < contractionSize; ++qp) {
        for(int rbf = 0; rbf < numBasis; ++rbf) {
          contractionData_GPURight.at(cl*numBasis*contractionSize + qp*numBasis + rbf) =
	  contractionData_Right.at(cl*numBasis*contractionSize + rbf*contractionSize + qp);
	}

	for(int lbf = 0; lbf < numBasis; ++lbf) {
          contractionData_GPULeft.at(cl*numBasis*contractionSize + qp*numBasis + lbf) =
	  contractionData_Left.at(cl*numBasis*contractionSize + lbf*contractionSize + qp);
	}
      }
    }

    // Then copy it over
    float * dev_contractionData_Right;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_Right,
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_Right,
                              &contractionData_GPURight[0],
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis,
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_Left;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_Left,
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_Left,
                              &contractionData_GPULeft[0],
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis,
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
      doCudaContractions_Independent_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(numberOfContractions*numBasis*numBasis,
                                   maxNumberOfContractions,
                                   contractionSize,
                                   numBasis,
                                   dev_contractionData_Right,
                                   dev_contractionData_Left,
                                   dev_contractionResults);
    } else if (cudaStyle == CudaStyle_Reduction) {
      doCudaContractions_Reduction_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        numberOfThreadsPerBlock * sizeof(float)>>>(numberOfContractions,
                                                   contractionSize,
                                                   dev_contractionData_Right,
                                                   dev_contractionData_Left,
                                                   dev_contractionResults);
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
  checkCudaError(cudaMemcpy(&contractionResults->at(0), dev_contractionResults,
                            numberOfContractions *numBasis*numBasis* sizeof(float),
                            cudaMemcpyDeviceToHost));
  // check the results
  checkAnswer(correctResults, *contractionResults,
              numberOfContractions*numBasis*numBasis, memorySize,
              convertCudaStyleToString(cudaStyle));

  // scrub the results
  std::fill(contractionResults->begin(),
            contractionResults->end(),
            std::numeric_limits<float>::quiet_NaN());
  checkCudaError(cudaMemcpy(dev_contractionResults, &contractionResults->at(0),
                            numberOfContractions * numBasis*numBasis*sizeof(float),
                            cudaMemcpyHostToDevice));


  //Free data
  checkCudaError(cudaFree(dev_contractionData_Right));
  checkCudaError(cudaFree(dev_contractionData_Left));

  return totalElapsedTime;
}

double
runCudaTeamTest(const CudaStyle cudaStyle,
            const unsigned int numberOfThreadsPerBlock,
            const unsigned int numberOfRepeats,
            const unsigned int maxNumberOfCudaBlocks,
            const unsigned int numCells,
            const unsigned int maxNumberOfContractions,
            const unsigned int contractionSize,
            const unsigned int numBasis,
            const unsigned int memorySize,
            const vector<float> & correctResults,
            const ClearCacheStyle clearCacheStyle,
            const int * const dev_junkDataToClearTheCache,
            const unsigned int junkDataSize,
            const vector<float> & contractionData_Right,
            const vector<float> & contractionData_Left,
            int * const dev_junkDataCounter,
            unsigned int * const totalNumberOfRepeats,
            float * const dev_contractionResults,
            vector<float> * const contractionResults,
            const unsigned int tileSize) {
  const unsigned int numberOfBlocks =
    min(maxNumberOfCudaBlocks,
        (unsigned int)ceil(numCells*numBasis*numBasis/float(numberOfThreadsPerBlock)));

    // Format the data the way we want and then copy it to the GPU
    vector<float> contractionData_GPURight(contractionData_Right.size());
    vector<float> contractionData_GPULeft(contractionData_Right.size());

    for (int cl = 0; cl < numBasis; ++cl) {
      for (int qp = 0; qp < contractionSize; ++qp) {
        for(int rbf = 0; rbf < numBasis; ++rbf) {
          contractionData_GPURight.at(cl*numBasis*contractionSize + qp*numBasis + rbf) =
    contractionData_Right.at(cl*numBasis*contractionSize + rbf*contractionSize + qp);
  }

  for(int lbf = 0; lbf < numBasis; ++lbf) {
          contractionData_GPULeft.at(cl*numBasis*contractionSize + lbf*contractionSize + qp) =
    contractionData_Left.at(cl*numBasis*contractionSize + lbf*contractionSize + qp);
  }
      }
    }

    // Then copy it over
    float * dev_contractionData_Right;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_Right,
                              maxNumberOfContractions * contractionSize *
            sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_Right,
                              &contractionData_GPURight[0],
                              maxNumberOfContractions * contractionSize *
            sizeof(float) * numBasis,
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_Left;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_Left,
                              maxNumberOfContractions * contractionSize *
            sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_Left,
                              &contractionData_GPULeft[0],
                              maxNumberOfContractions * contractionSize *
            sizeof(float) * numBasis,
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
      doCudaContractions_Slicing_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        contractionSize * sizeof(float)>>>(numCells,
                                   contractionSize,
                                   numBasis,
                                   dev_contractionData_Right,
                                   dev_contractionData_Left,
                                   dev_contractionResults);
    } else if (cudaStyle == CudaStyle_Tiling) {
      doCudaContractions_Tiling_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        2 * tileSize * tileSize * sizeof(float)>>>(numCells,
                                                   contractionSize,
                                                   tileSize,
                                                   numBasis,
                                                   dev_contractionData_Right,
                                                   dev_contractionData_Left,
                                                   dev_contractionResults);
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
  checkCudaError(cudaMemcpy(&contractionResults->at(0), dev_contractionResults,
                            numCells *numBasis*numBasis* sizeof(float),
                            cudaMemcpyDeviceToHost));
  // check the results
  checkAnswer(correctResults, *contractionResults,
              numBasis, memorySize,
              convertCudaStyleToString(cudaStyle));

  // scrub the results
  std::fill(contractionResults->begin(),
            contractionResults->end(),
            std::numeric_limits<float>::quiet_NaN());
  checkCudaError(cudaMemcpy(dev_contractionResults, &contractionResults->at(0),
                            numCells * numBasis*numBasis*sizeof(float),
                            cudaMemcpyHostToDevice));


  //Free data
  checkCudaError(cudaFree(dev_contractionData_Right));
  checkCudaError(cudaFree(dev_contractionData_Left));

  return totalElapsedTime;
}
double
runSwitchingCudaTest(const unsigned int numberOfRepeats,
                     const unsigned int maxNumberOfCudaBlocks,
                     const unsigned int numberOfContractions,
                     const unsigned int maxNumberOfContractions,
                     const unsigned int contractionSize,
		     const unsigned int numBasis,
                     const unsigned int memorySize,
                     const vector<float> & correctResults,
                     const ClearCacheStyle clearCacheStyle,
                     const int * const dev_junkDataToClearTheCache,
                     const unsigned int junkDataSize,
                     const vector<float> & contractionData_LayoutLeft_Right,
                     const vector<float> & contractionData_LayoutLeft_Left,
                     const vector<float> & contractionData_LayoutRight_Right,
                     const vector<float> & contractionData_LayoutRight_Left,
                     int * const dev_junkDataCounter,
                     unsigned int * const totalNumberOfRepeats,
                     float * const dev_contractionResults,
                     vector<float> * const contractionResults) {
  // if i can't saturate occupancy, do the reduction version
  // i got this number by just looking at where the plots crossed, where
  //  the reduction style actually starts beating the independent.
  if (numberOfContractions < 200) {
    const unsigned int numberOfThreadsPerBlock =
      std::min(unsigned(1024),
               unsigned(ceil(contractionSize / 32.)) * 32);
    return
      runCudaTest(CudaStyle_Reduction,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfContractions,
                  maxNumberOfContractions,
                  contractionSize,
		  numBasis,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  contractionData_LayoutRight_Right,
                  contractionData_LayoutRight_Left,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_contractionResults,
                  contractionResults);
  } else {
    const unsigned int numberOfThreadsPerBlock = 1024;
    return
      runCudaTest(CudaStyle_Independent,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfContractions,
                  maxNumberOfContractions,
                  contractionSize,
		  numBasis,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  contractionData_LayoutLeft_Right,
                  contractionData_LayoutLeft_Left,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_contractionResults,
                  contractionResults);
  }
}





//#ifdef ENABLE_KOKKOS



template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarKokkosCudaFunctor {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numCells;
	int _numPoints;
	int _numLeftFields;
	int _numRightFields;

	contractFieldFieldScalarKokkosCudaFunctor(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numCells,
			int numLeftFields,
			int numRightFields,
			int numPoints) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numCells(numCells),
		_numPoints(numPoints),
		_numLeftFields(numLeftFields),
		_numRightFields(numRightFields)
	{
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION
	    void operator()(const unsigned int elementIndex) const {

		int myID = elementIndex;
		int myMatrix = myID / (_numLeftFields * _numRightFields);
		int matrixIndex = myID % (_numLeftFields * _numRightFields);

		int matrixRow = matrixIndex / _numRightFields;
		int matrixCol = matrixIndex % _numRightFields;

		float temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    temp += _leftFields(myMatrix, qp, matrixRow) * _rightFields(myMatrix, qp, matrixCol);
		}
		_outputFields(myMatrix, matrixRow, matrixCol) = temp;
	    }
};




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

template <class DeviceType, class KokkosContractionData,
          class KokkosContractionResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _contractionSize;
  KokkosContractionData _data_A;
  KokkosContractionData _data_B;
  KokkosContractionResults _results;

  KokkosFunctor_Independent(const unsigned int contractionSize,
                            KokkosContractionData data_A,
                            KokkosContractionData data_B,
                            KokkosContractionResults results) :
    _contractionSize(contractionSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int contractionIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _contractionSize;
         ++entryIndex) {
      sum +=
        _data_A(contractionIndex, entryIndex) *
        _data_B(contractionIndex, entryIndex);
    }
    _results(contractionIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};

template <class DeviceType, class KokkosContractionData>
double
runKokkosTest(const unsigned int numberOfContractions,
              const unsigned int numberOfRepeats,
              const unsigned int contractionSize,
	      const unsigned int numLeftFields,
	      const unsigned int numRightFields,
              const unsigned int memorySize,
              const vector<float> & contractionData_LayoutRight_Right,
              const vector<float> & contractionData_LayoutRight_Left,
              const vector<float> & correctResults,
              const string & kokkosFlavor,
              const ClearCacheStyle clearCacheStyle,
              const vector<int> & junkDataToClearTheCache,
              size_t * junkDataCounter,
              unsigned int * const totalNumberOfRepeats,
              vector<float> * contractionResults) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename KokkosContractionData::HostMirror     KokkosContractionData_Host;
  typedef Kokkos::View<float***, Kokkos::LayoutRight,
	DeviceType>              KokkosContractionResults;
  typedef typename KokkosContractionResults::HostMirror  KokkosContractionResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  const unsigned int numPoints = contractionSize;



    KokkosContractionData dev_kokkosContractionData_Right("kokkos data A",
                                                  numberOfContractions,
						  numPoints,
                                                  numRightFields);
  KokkosContractionData_Host kokkosContractionData_Right =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Right);

  KokkosContractionData dev_kokkosContractionData_Left("kokkos data B",
                                                  numberOfContractions,
						  numPoints,
                                                  numLeftFields);
  KokkosContractionData_Host kokkosContractionData_Left =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Left);

  KokkosContractionResults dev_kokkosContractionResults("kokkos dot product results",
                                                      numberOfContractions,
						      numLeftFields,
						      numRightFields);
  KokkosContractionResults_Host kokkosContractionResults =
    Kokkos::create_mirror_view(dev_kokkosContractionResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);



	for (int cl = 0; cl < numberOfContractions; ++cl) {
	    for (int qp = 0; qp < numPoints; ++qp) {
		for(int rbf = 0; rbf < numRightFields; ++rbf) {
		    kokkosContractionData_Right(cl, qp, rbf) =
		    contractionData_LayoutRight_Right.at(cl*numRightFields*numPoints
		    + rbf*numPoints + qp);
		}
		for(int lbf = 0; lbf < numLeftFields; ++lbf) {
		    kokkosContractionData_Left(cl, qp, lbf) =
		    contractionData_LayoutRight_Left.at(cl*numLeftFields*numPoints +
		    lbf*numPoints + qp);
		}
	    }
	}


  /*

  // copy the data into the device views and ship them over
  for (unsigned int contractionIndex = 0;
       contractionIndex < numberOfContractions; ++contractionIndex) {
    for (unsigned int entryIndex = 0;
         entryIndex < contractionSize; ++entryIndex) {
      kokkosContractionData_Right(contractionIndex, entryIndex) =
        contractionData_LayoutRight_Right[contractionIndex * contractionSize +
                                     entryIndex];
      kokkosContractionData_Left(contractionIndex, entryIndex) =
        contractionData_LayoutRight_Left[contractionIndex * contractionSize +
                                     entryIndex];
    }
  }*/
  Kokkos::deep_copy(dev_kokkosContractionData_Right, kokkosContractionData_Right);
  Kokkos::deep_copy(dev_kokkosContractionData_Left, kokkosContractionData_Left);

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


  // breaking formatting convention because holy freak that's long
  contractFieldFieldScalarKokkosCudaFunctor<DeviceType,
                            KokkosContractionData,
			    KokkosContractionData,
                            KokkosContractionResults>
    contractionFunctor(dev_kokkosContractionData_Left,
		       dev_kokkosContractionData_Right,
		       dev_kokkosContractionResults,
		       numberOfContractions,
		       numLeftFields,
		       numRightFields,
		       numPoints);

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

    // actually do the calculation
    Kokkos::parallel_for(numberOfContractions*numLeftFields*numRightFields,
    contractionFunctor);

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
  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosContractionResults, dev_kokkosContractionResults);
  for (unsigned int contractionIndex = 0;
	  contractionIndex < numberOfContractions; ++contractionIndex) {
      for (int lbf = 0; lbf < numLeftFields; lbf++) {
	  for (int rbf = 0; rbf < numRightFields; rbf++) {
	      contractionResults->at(contractionIndex*numLeftFields*numRightFields+lbf*numRightFields+rbf) =
		  kokkosContractionResults(contractionIndex, lbf, rbf);
	  }
      }
  }

  // check the results
  checkAnswer(correctResults, *contractionResults,
              numberOfContractions*numLeftFields*numRightFields, memorySize,
              kokkosFlavor);
  // scrub the results
  std::fill(contractionResults->begin(),
            contractionResults->end(),
            std::numeric_limits<float>::quiet_NaN());

  return totalElapsedTime;
}


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Reduction_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Reduction_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
      unsigned int _numRightFields, unsigned int _numPoints,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    int myID =  thread.league_rank();
    int myMatrix = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);

    int matrixRow = matrixIndex / numRightFields;
    int matrixCol = matrixIndex % numRightFields;

    float sum = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, numPoints),
        [&] (const unsigned int& i, float& sum) {
          sum += leftView(myMatrix, matrixRow, i)
                 * rightView(myMatrix, i, matrixCol);
        },
        sum);
    outputView(myMatrix, matrixRow, matrixCol) = sum;
  }
};



template <class DeviceType, class KokkosContractionData>
double
runKokkosTeamReductionTest(const unsigned int numberOfContractions,
              const unsigned int numberOfRepeats,
              const unsigned int contractionSize,
	      const unsigned int numLeftFields,
	      const unsigned int numRightFields,
              const unsigned int memorySize,
              const vector<float> & contractionData_LayoutRight_Right,
              const vector<float> & contractionData_LayoutRight_Left,
              const vector<float> & correctResults,
              const string & kokkosFlavor,
              const ClearCacheStyle clearCacheStyle,
              const vector<int> & junkDataToClearTheCache,
              size_t * junkDataCounter,
              unsigned int * const totalNumberOfRepeats,
              vector<float> * contractionResults) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename KokkosContractionData::HostMirror     KokkosContractionData_Host;
  typedef Kokkos::View<float***, Kokkos::LayoutRight,
	DeviceType>              KokkosContractionResults;
  typedef typename KokkosContractionResults::HostMirror  KokkosContractionResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  const unsigned int numPoints = contractionSize;



    KokkosContractionData dev_kokkosContractionData_Right("kokkos data A",
                                                  numberOfContractions,
						  numPoints,
                                                  numRightFields);
  KokkosContractionData_Host kokkosContractionData_Right =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Right);

  KokkosContractionData dev_kokkosContractionData_Left("kokkos data B",
                                                  numberOfContractions,
                                                  numLeftFields,
						  numPoints);
  KokkosContractionData_Host kokkosContractionData_Left =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Left);

  KokkosContractionResults dev_kokkosContractionResults("kokkos dot product results",
                                                      numberOfContractions,
						      numLeftFields,
						      numRightFields);
  KokkosContractionResults_Host kokkosContractionResults =
    Kokkos::create_mirror_view(dev_kokkosContractionResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);



	for (int cl = 0; cl < numberOfContractions; ++cl) {
	    for (int qp = 0; qp < numPoints; ++qp) {
		for(int rbf = 0; rbf < numRightFields; ++rbf) {
		    kokkosContractionData_Right(cl, qp, rbf) =
		    contractionData_LayoutRight_Right.at(cl*numRightFields*numPoints
		    + rbf*numPoints + qp);
		}
		for(int lbf = 0; lbf < numLeftFields; ++lbf) {
		    kokkosContractionData_Left(cl, lbf, qp) =
		    contractionData_LayoutRight_Left.at(cl*numLeftFields*numPoints +
		    lbf*numPoints + qp);
		}
	    }
	}


  /*

  // copy the data into the device views and ship them over
  for (unsigned int contractionIndex = 0;
       contractionIndex < numberOfContractions; ++contractionIndex) {
    for (unsigned int entryIndex = 0;
         entryIndex < contractionSize; ++entryIndex) {
      kokkosContractionData_Right(contractionIndex, entryIndex) =
        contractionData_LayoutRight_Right[contractionIndex * contractionSize +
                                     entryIndex];
      kokkosContractionData_Left(contractionIndex, entryIndex) =
        contractionData_LayoutRight_Left[contractionIndex * contractionSize +
                                     entryIndex];
    }
  }*/
  Kokkos::deep_copy(dev_kokkosContractionData_Right, kokkosContractionData_Right);
  Kokkos::deep_copy(dev_kokkosContractionData_Left, kokkosContractionData_Left);

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


  // breaking formatting convention because holy freak that's long
  CFFS_Reduction_TeamFunctor<KokkosContractionData,
			    KokkosContractionData,
                            KokkosContractionResults>
    contractionFunctor(numberOfContractions,
		       numLeftFields,
		       numRightFields,
		       numPoints,
		       dev_kokkosContractionData_Left,
		       dev_kokkosContractionData_Right,
		       dev_kokkosContractionResults);

    const team_policy reduction_policy(
	numberOfContractions*numLeftFields*numRightFields , numPoints );


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

    // actually do the calculation
    Kokkos::parallel_for(reduction_policy, contractionFunctor);

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
  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosContractionResults, dev_kokkosContractionResults);
  for (unsigned int contractionIndex = 0;
	  contractionIndex < numberOfContractions; ++contractionIndex) {
      for (int lbf = 0; lbf < numLeftFields; lbf++) {
	  for (int rbf = 0; rbf < numRightFields; rbf++) {
	      contractionResults->at(contractionIndex*numLeftFields*numRightFields+lbf*numRightFields+rbf) =
		  kokkosContractionResults(contractionIndex, lbf, rbf);
	  }
      }
  }

  // check the results
  checkAnswer(correctResults, *contractionResults,
              numPoints, memorySize,
              kokkosFlavor);
  // scrub the results
  std::fill(contractionResults->begin(),
            contractionResults->end(),
            std::numeric_limits<float>::quiet_NaN());

  return totalElapsedTime;
}

template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Slicing_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Slicing_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
      unsigned int _numRightFields, unsigned int _numPoints,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    int l = thread.league_rank() % numLeftFields;
    int r = thread.team_rank();
    int c = thread.league_rank() / numLeftFields;

    Kokkos::View<float*, Kokkos::MemoryUnmanaged> shared_slice(thread.team_shmem(), numPoints);
    for (int p = thread.team_rank(); p < numPoints; p += thread.team_size()) {
      shared_slice(p) = leftView(c, l, p);
    }
    thread.team_barrier();

    float sum = 0;
    for (int p = 0; p < numPoints; ++p) {
      sum += shared_slice(p) * rightView(c, p, r);
    }
    outputView(c, l, r) = sum;

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints;
  }
};

template <class DeviceType, class KokkosContractionData>
double
runKokkosSlicingTest(const unsigned int numberOfContractions,
    const unsigned int numberOfRepeats,
    const unsigned int contractionSize,
    const unsigned int numLeftFields,
    const unsigned int numRightFields,
    const unsigned int memorySize,
    const vector<float> & contractionData_LayoutRight_Right,
    const vector<float> & contractionData_LayoutRight_Left,
    const vector<float> & correctResults,
    const string & kokkosFlavor,
    const ClearCacheStyle clearCacheStyle,
    const vector<int> & junkDataToClearTheCache,
    size_t * junkDataCounter,
    unsigned int * const totalNumberOfRepeats,
    vector<float> * contractionResults) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename KokkosContractionData::HostMirror     KokkosContractionData_Host;
  typedef Kokkos::View<float***, Kokkos::LayoutRight,
          DeviceType>              KokkosContractionResults;
  typedef typename KokkosContractionResults::HostMirror  KokkosContractionResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  const unsigned int numPoints = contractionSize;



  KokkosContractionData dev_kokkosContractionData_Right("kokkos data A",
      numberOfContractions,
      numPoints,
      numRightFields);
  KokkosContractionData_Host kokkosContractionData_Right =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Right);

  KokkosContractionData dev_kokkosContractionData_Left("kokkos data B",
      numberOfContractions,
      numLeftFields,
      numPoints);
  KokkosContractionData_Host kokkosContractionData_Left =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Left);

  KokkosContractionResults dev_kokkosContractionResults("kokkos dot product results",
      numberOfContractions,
      numLeftFields,
      numRightFields);
  KokkosContractionResults_Host kokkosContractionResults =
    Kokkos::create_mirror_view(dev_kokkosContractionResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
      junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);



  for (int cl = 0; cl < numberOfContractions; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for(int rbf = 0; rbf < numRightFields; ++rbf) {
        kokkosContractionData_Right(cl, qp, rbf) =
          contractionData_LayoutRight_Right.at(cl*numRightFields*numPoints
              + rbf*numPoints + qp);
      }
      for(int lbf = 0; lbf < numLeftFields; ++lbf) {
        kokkosContractionData_Left(cl, lbf, qp) =
          contractionData_LayoutRight_Left.at(cl*numLeftFields*numPoints +
              lbf*numPoints + qp);
      }
    }
  }


  /*

  // copy the data into the device views and ship them over
  for (unsigned int contractionIndex = 0;
  contractionIndex < numberOfContractions; ++contractionIndex) {
  for (unsigned int entryIndex = 0;
  entryIndex < contractionSize; ++entryIndex) {
  kokkosContractionData_Right(contractionIndex, entryIndex) =
  contractionData_LayoutRight_Right[contractionIndex * contractionSize +
  entryIndex];
  kokkosContractionData_Left(contractionIndex, entryIndex) =
  contractionData_LayoutRight_Left[contractionIndex * contractionSize +
  entryIndex];
  }
  }*/
  Kokkos::deep_copy(dev_kokkosContractionData_Right, kokkosContractionData_Right);
  Kokkos::deep_copy(dev_kokkosContractionData_Left, kokkosContractionData_Left);

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


  // breaking formatting convention because holy freak that's long
  CFFS_Slicing_TeamFunctor<KokkosContractionData,
    KokkosContractionData,
    KokkosContractionResults>
      contractionFunctor(numberOfContractions,
          numLeftFields,
          numRightFields,
          numPoints,
          dev_kokkosContractionData_Left,
          dev_kokkosContractionData_Right,
          dev_kokkosContractionResults);

  const team_policy slicing_policy(
      numberOfContractions*numLeftFields , numRightFields );


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

    // actually do the calculation
    Kokkos::parallel_for(slicing_policy, contractionFunctor);

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
  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosContractionResults, dev_kokkosContractionResults);
  for (unsigned int contractionIndex = 0;
      contractionIndex < numberOfContractions; ++contractionIndex) {
    for (int lbf = 0; lbf < numLeftFields; lbf++) {
      for (int rbf = 0; rbf < numRightFields; rbf++) {
        contractionResults->at(contractionIndex*numLeftFields*numRightFields+lbf*numRightFields+rbf) =
          kokkosContractionResults(contractionIndex, lbf, rbf);
      }
    }
  }

  // check the results
  checkAnswer(correctResults, *contractionResults,
      numberOfContractions*numLeftFields*numRightFields, memorySize,
      kokkosFlavor);
  // scrub the results
  std::fill(contractionResults->begin(),
      contractionResults->end(),
      std::numeric_limits<float>::quiet_NaN());

  return totalElapsedTime;
}


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Tiling_TeamFunctor {
  const unsigned int numCells;
  const unsigned int numLeftFields;
  const unsigned int numRightFields;
  const unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;
  const unsigned int tile_size;


  CFFS_Tiling_TeamFunctor(const unsigned int _numCells,
      const unsigned int _numLeftFields,
      const unsigned int _numRightFields,
      const unsigned int _numPoints,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView,
      const unsigned int _tile_size) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView),
    tile_size(_tile_size)
    {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    // Num teams is (numLeftField * numRightField)/tile_size^2 * numCells
    int numTiles = thread.league_size() / numCells;
    int c = thread.league_rank() / numTiles;
    int tilePosition = thread.league_rank() % numTiles;
    int lTile = tilePosition / ((numRightFields-1) / tile_size + 1);
    int rTile = tilePosition % ((numRightFields-1) / tile_size + 1);

    int tileCol = thread.team_rank() % tile_size;
    int tileRow = thread.team_rank() / tile_size;

    int l = lTile*tile_size + tileRow;
    int r = rTile*tile_size + tileCol;

    Kokkos::View<float**, Kokkos::MemoryUnmanaged> left_tile(thread.team_shmem(), tile_size, tile_size);
    Kokkos::View<float**, Kokkos::MemoryUnmanaged> right_tile(thread.team_shmem(), tile_size, tile_size);

    float totalSum = 0;
    for (int tileIndex = 0; tileIndex < ((numPoints-1)/ tile_size) + 1; ++tileIndex) {
	if (tileIndex*tile_size + tileCol < numPoints && l < numLeftFields) {
	    left_tile(tileRow, tileCol) = leftView(c, l, tileIndex*tile_size + tileCol);
	}
	else {
	    left_tile(tileRow, tileCol) = 0.0;
	}
	if (tileIndex*tile_size + tileRow < numPoints && r < numRightFields) {
	    right_tile(tileRow, tileCol) = rightView(c, tileIndex*tile_size + tileRow, r);
	}
	else {
	    right_tile(tileRow, tileCol) = 0.0;
	}
	thread.team_barrier();

	float sum = 0;
	for (int i = 0; i < tile_size; ++i) {
	    sum += left_tile(tileRow, i) * right_tile(i, tileCol);
	}
	totalSum += sum;

	thread.team_barrier();
    }

    if (l < numLeftFields && r < numRightFields) {
	outputView(c, l, r) = totalSum;
    }
  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * team_size * 2;
  }

};

template <class DeviceType, class KokkosContractionData>
double
runKokkosTilingTest(const unsigned int numberOfContractions,
    const unsigned int numberOfRepeats,
    const unsigned int contractionSize,
    const unsigned int numLeftFields,
    const unsigned int numRightFields,
    const unsigned int memorySize,
    const vector<float> & contractionData_LayoutRight_Right,
    const vector<float> & contractionData_LayoutRight_Left,
    const vector<float> & correctResults,
    const string & kokkosFlavor,
    const ClearCacheStyle clearCacheStyle,
    const vector<int> & junkDataToClearTheCache,
    size_t * junkDataCounter,
    unsigned int * const totalNumberOfRepeats,
    vector<float> * contractionResults,
    const unsigned int tile_size) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename KokkosContractionData::HostMirror     KokkosContractionData_Host;
  typedef Kokkos::View<float***, Kokkos::LayoutRight,
          DeviceType>              KokkosContractionResults;
  typedef typename KokkosContractionResults::HostMirror  KokkosContractionResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  const unsigned int numPoints = contractionSize;



  KokkosContractionData dev_kokkosContractionData_Right("kokkos data A",
      numberOfContractions,
      numPoints,
      numRightFields);
  KokkosContractionData_Host kokkosContractionData_Right =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Right);

  KokkosContractionData dev_kokkosContractionData_Left("kokkos data B",
      numberOfContractions,
      numLeftFields,
      numPoints);
  KokkosContractionData_Host kokkosContractionData_Left =
    Kokkos::create_mirror_view(dev_kokkosContractionData_Left);

  KokkosContractionResults dev_kokkosContractionResults("kokkos dot product results",
      numberOfContractions,
      numLeftFields,
      numRightFields);
  KokkosContractionResults_Host kokkosContractionResults =
    Kokkos::create_mirror_view(dev_kokkosContractionResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
      junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);



  for (int cl = 0; cl < numberOfContractions; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for(int rbf = 0; rbf < numRightFields; ++rbf) {
        kokkosContractionData_Right(cl, qp, rbf) =
          contractionData_LayoutRight_Right.at(cl*numRightFields*numPoints
              + rbf*numPoints + qp);
      }
      for(int lbf = 0; lbf < numLeftFields; ++lbf) {
        kokkosContractionData_Left(cl, lbf, qp) =
          contractionData_LayoutRight_Left.at(cl*numLeftFields*numPoints +
              lbf*numPoints + qp);
      }
    }
  }


  /*

  // copy the data into the device views and ship them over
  for (unsigned int contractionIndex = 0;
  contractionIndex < numberOfContractions; ++contractionIndex) {
  for (unsigned int entryIndex = 0;
  entryIndex < contractionSize; ++entryIndex) {
  kokkosContractionData_Right(contractionIndex, entryIndex) =
  contractionData_LayoutRight_Right[contractionIndex * contractionSize +
  entryIndex];
  kokkosContractionData_Left(contractionIndex, entryIndex) =
  contractionData_LayoutRight_Left[contractionIndex * contractionSize +
  entryIndex];
  }
  }*/
  Kokkos::deep_copy(dev_kokkosContractionData_Right, kokkosContractionData_Right);
  Kokkos::deep_copy(dev_kokkosContractionData_Left, kokkosContractionData_Left);

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


  // breaking formatting convention because holy freak that's long
  CFFS_Tiling_TeamFunctor<KokkosContractionData,
    KokkosContractionData,
    KokkosContractionResults>
      contractionFunctor(numberOfContractions,
          numLeftFields,
          numRightFields,
          numPoints,
          dev_kokkosContractionData_Left,
          dev_kokkosContractionData_Right,
          dev_kokkosContractionResults,
          tile_size);

  const team_policy tiling_policy(
      numberOfContractions
      * ((numLeftFields-1/tile_size) +1)
      * ((numRightFields-1/tile_size) +1) ,
      tile_size*tile_size );


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

    // actually do the calculation
    Kokkos::parallel_for(tiling_policy, contractionFunctor);

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
  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosContractionResults, dev_kokkosContractionResults);
  for (unsigned int contractionIndex = 0;
      contractionIndex < numberOfContractions; ++contractionIndex) {
    for (int lbf = 0; lbf < numLeftFields; lbf++) {
      for (int rbf = 0; rbf < numRightFields; rbf++) {
        contractionResults->at(contractionIndex*numLeftFields*numRightFields+lbf*numRightFields+rbf) =
          kokkosContractionResults(contractionIndex, lbf, rbf);
      }
    }
  }

  // check the results
  checkAnswer(correctResults, *contractionResults,
      numberOfContractions*numLeftFields*numRightFields, memorySize,
      kokkosFlavor);
  // scrub the results
  std::fill(contractionResults->begin(),
      contractionResults->end(),
      std::numeric_limits<float>::quiet_NaN());

  return totalElapsedTime;
}



//#endif // ENABLE_KOKKOS





void contractFieldFieldScalarSerial(vector<float> & outputFields, // c, l, r
		const vector<float> &            leftFields,  // c, l ,p
		const vector<float> &           rightFields,  // c, r, p
    int                  numCells,
    int                  numLeftFields,
    int                  numRightFields,
    int                  numPoints) {

    float  tmpVal;
    for (int cl = 0; cl < numCells; cl++) {
	for (int lbf = 0; lbf < numLeftFields; lbf++) {
	    for (int rbf = 0; rbf < numRightFields; rbf++) {
		tmpVal = 0;
		for (int qp = 0; qp < numPoints; qp++) {
		    tmpVal += leftFields.at(cl*numLeftFields*numPoints +
		    lbf*numPoints + qp)
			* rightFields.at(cl*numPoints*numRightFields +
			rbf*numPoints + qp);
		} // P-loop
		outputFields.at(cl*numLeftFields*numRightFields +
		lbf*numRightFields + rbf) = tmpVal;
	    } // R-loop
	} // L-loop
    } // C-loop
}

int main(int argc, char* argv[]) {

//#ifdef ENABLE_KOKKOS
  Kokkos::initialize(argc, argv);
//#endif

  // ===============================================================
  // ********************** < input> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const vector<unsigned int> contractionSizes =
    {{/*8,*/ 16, 32, 64, 128, 512, 1024/*, 2048*/}};
  const array<float, 2> memorySizeExtrema = {{1e6, 1e9}};
  const unsigned int numberOfMemorySizes = 10;
  const unsigned int maxNumberOfCudaBlocks = unsigned(1e4);
  const unsigned int tile_size = 16;
  const ClearCacheStyle clearCacheStyle =
    ClearCacheAfterEveryRepeat;
  const unsigned int numberOfRepeats =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? 2 : 250;
  const string machineName = "shadowfax";
  const string prefix = "data/ArrayOfContractions_";
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </input> ******************************
  // ===============================================================

  // derive some values from the inputs
  const unsigned int numberOfContractionSizes = contractionSizes.size();
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
    const unsigned int maxContractionSize = contractionSizes.back();
    // memory size is linear on a log scale, but rounded to a multiple of the
    //  largest dot product size
    const unsigned int desiredMemorySizeInBytes = pow(10., thisLog);
    // now, in this amount of memory i have to fit two vectors of data
    // that are multiples of the max dot product size
    const unsigned int memorySizeInBytes =
      unsigned(desiredMemorySizeInBytes /
               float(4 * sizeof(float) * maxContractionSize)) *
      4 * sizeof(float) * maxContractionSize;
    memorySizes[memorySizeIndex] = memorySizeInBytes;
  }

  // create a c++11 random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  // these are just containers for storing the numbers we'll be plotting.
  // i feel a little dirty using a vector<vector>, but i don't want to introduce
  //  a dependence on eigen or something for a real matrix.
  vector<vector<float> >
    contractionSizeMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    numberOfContractionsMatrix(numberOfContractionSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    memorySizeMatrix(numberOfContractionSizes,
                     vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    serialTimesMatrix(numberOfContractionSizes,
                      vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    ompTimesMatrix(numberOfContractionSizes,
                   vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaIndependent_TimesMatrix(numberOfContractionSizes,
                                vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaReduction_TimesMatrix(numberOfContractionSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaSwitchingTimesMatrix(numberOfContractionSizes,
                             vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    cudaSlicingTimesMatrix(numberOfContractionSizes,
                           vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    cudaTilingTimesMatrix(numberOfContractionSizes,
                            vector<float>(numberOfMemorySizes, 0));
//#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosTeamReductionTimesMatrix(numberOfContractionSizes,
				     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosSlicingTimesMatrix(numberOfContractionSizes,
				     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosTilingTimesMatrix(numberOfContractionSizes,
				     vector<float>(numberOfMemorySizes, 0));
//#endif

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

  // for each dot product s;
    for (int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {
    const unsigned int contractionSize = contractionSizes[contractionSizeIndex];

    const int numPoints = contractionSize;
    const int numBasis = 16;

    const timespec thisSizesTic = getTimePoint();

    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumberOfContractions =
      memorySizes.back() / 4 / sizeof(float) / (contractionSize * numBasis);
    vector<float> contractionData_LayoutRight_Right(maxNumberOfContractions *
    contractionSize * numBasis);
    vector<float> contractionData_LayoutRight_Left(contractionData_LayoutRight_Right.size());
    vector<float> contractionData_LayoutLeft_Right(contractionData_LayoutRight_Right.size());
    vector<float> contractionData_LayoutLeft_Left(contractionData_LayoutRight_Left.size());
    for (unsigned int contractionIndex = 0;
         contractionIndex < maxNumberOfContractions; ++contractionIndex) {

      for (unsigned int entryIndex = 0;
           entryIndex < contractionSize * numBasis; ++entryIndex) {

	const unsigned int layoutRightIndex =
          contractionIndex * contractionSize*numBasis + entryIndex;

	contractionData_LayoutRight_Right[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);

	contractionData_LayoutRight_Left[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);

	const unsigned int layoutLeftIndex =
          entryIndex * maxNumberOfContractions + contractionIndex;

	contractionData_LayoutLeft_Right[layoutLeftIndex] =
          contractionData_LayoutRight_Right[layoutRightIndex];

	contractionData_LayoutLeft_Left[layoutLeftIndex] =
          contractionData_LayoutRight_Left[layoutRightIndex];
      }
    }

    vector<float>
    contractionResults(maxNumberOfContractions*numBasis*numBasis,
                                    std::numeric_limits<float>::quiet_NaN());
    /*
    // now, because we'll be working with cuda stuff, also allocate the inputs
    //  and output on the gpu and copy them over
    float * dev_contractionData_LayoutRight_Right;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutRight_Right,
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutRight_Right,
                              &contractionData_LayoutRight_Right[0],
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis,
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_LayoutRight_Left;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutRight_Left,
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutRight_Left,
                              &contractionData_LayoutRight_Left[0],
                              maxNumberOfContractions * contractionSize *
			      sizeof(float) * numBasis,
                              cudaMemcpyHostToDevice));
			      */
    float * dev_contractionResults;
    checkCudaError(cudaMalloc((void **) &dev_contractionResults,
                              maxNumberOfContractions * numBasis * numBasis *
			      sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionResults, &contractionResults[0],
                              maxNumberOfContractions * sizeof(float) * numBasis
			      * numBasis,
                              cudaMemcpyHostToDevice));
			      /*
    // make and populate the LayoutLeft versions
    float * dev_contractionData_LayoutLeft_Right;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutLeft_Right,
                              maxNumberOfContractions * contractionSize *
			      numBasis * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutLeft_Right,
                              &contractionData_LayoutLeft_Right[0],
                              maxNumberOfContractions * contractionSize *
			      numBasis * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_LayoutLeft_Left;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutLeft_Left,
                              maxNumberOfContractions * contractionSize *
			      numBasis * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutLeft_Left,
                              &contractionData_LayoutLeft_Left[0],
                              maxNumberOfContractions * contractionSize *
			      numBasis * sizeof(float),
                              cudaMemcpyHostToDevice));
    */
    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex];
      const unsigned int numberOfContractions =
        memorySize / 4 / sizeof(float) / (contractionSize * numBasis);
	/*
      if (memorySize != 4 * sizeof(float) * numberOfContractions * contractionSize) {
        fprintf(stderr, "invalid memory size of %u for dot product size of "
                "%u because it doesn't divide evenly, remainder is %zu\n",
                memorySize, contractionSize,
                memorySize % (4 * sizeof(float) * contractionSize));
        exit(1);
      }
	*/

      // ===============================================================
      // ********************** < do serial> ***************************
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
         contractFieldFieldScalarSerial(contractionResults,
	 contractionData_LayoutRight_Left, contractionData_LayoutRight_Right,
	 numberOfContractions, numBasis, numBasis, numPoints);

	 if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            serialTimesMatrix[contractionSizeIndex][memorySizeIndex] += elapsedTime;

            junkDataCounter +=
              std::accumulate(junkDataToClearTheCache.begin(),
                              junkDataToClearTheCache.end(), size_t(0));
          }
        }
        if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
          const timespec toc = getTimePoint();
          const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
          serialTimesMatrix[contractionSizeIndex][memorySizeIndex] = elapsedTime;
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do serial> ***************************
      // ===============================================================

      const vector<float> correctResults = contractionResults;
      // scrub the results
      std::fill(contractionResults.begin(),
                contractionResults.end(),
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
  shared(contractionData_LayoutRight_Right, contractionData_LayoutRight_Left,    \
         contractionResults)
	  for (unsigned int elementId = 0; elementId <
		  numberOfContractions*numBasis*numBasis; elementId++) {
	      int myMatrix = elementId / (numBasis*numBasis);
	      int matrixIndex = elementId % (numBasis*numBasis);
	      int matrixRow = matrixIndex / numBasis;
	      int matrixCol = matrixIndex % numBasis;

	      float temp = 0;
	      int cellMult = numBasis*numPoints;
	      int finalCell = numBasis*numBasis;
	      for (int qp = 0; qp < numPoints; qp++) {
		  temp += contractionData_LayoutRight_Left.at(myMatrix*cellMult +
			  matrixRow*numPoints + qp) *
		      contractionData_LayoutRight_Right.at(myMatrix*cellMult +
			      matrixCol*numPoints + qp);
	      }
	      contractionResults.at(myMatrix*finalCell + matrixRow*numBasis +
		      matrixCol);
	  }


	  /*
	  for (unsigned int contractionIndex = 0;
               contractionIndex < numberOfContractions;
               ++contractionIndex) {
            const unsigned int shortcutIndex = contractionIndex * contractionSize;
            float sum = 0;
            for (unsigned int entryIndex = 0;
                 entryIndex < contractionSize; ++entryIndex) {
              sum +=
                contractionData_LayoutRight_Right[shortcutIndex + entryIndex] *
                contractionData_LayoutRight_Left[shortcutIndex + entryIndex];
            }
            contractionResults[contractionIndex] = sum;
          }
	  */

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompTimesMatrix[contractionSizeIndex][memorySizeIndex] += elapsedTime;

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
          ompTimesMatrix[contractionSizeIndex][memorySizeIndex] = elapsedTime;
          // check the results
          checkAnswer(correctResults, contractionResults,
                      contractionSize, memorySize,
                      string("omp"));
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp> ******************************
      // ===============================================================

      // scrub the results
      std::fill(contractionResults.begin(),
                contractionResults.end(),
                std::numeric_limits<float>::quiet_NaN());
      /* checkCudaError(cudaMemcpy(dev_contractionResults, &contractionResults[0],
                                maxNumberOfContractions * sizeof(float),
                                cudaMemcpyHostToDevice));
      */

      // ===============================================================
      // ***************** < do cuda independent> **********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock = 256;

        cudaIndependent_TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Independent,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfContractions,
                      maxNumberOfContractions,
                      contractionSize,
                      numBasis,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      contractionData_LayoutRight_Right,
                      contractionData_LayoutRight_Left,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_contractionResults,
                      &contractionResults);

      }

      {
        const unsigned int numberOfThreadsPerBlock = numBasis;

        cudaSlicingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTeamTest(CudaStyle_Slicing,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfContractions,
                      maxNumberOfContractions,
                      contractionSize,
                      numBasis,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      contractionData_LayoutRight_Right,
                      contractionData_LayoutRight_Left,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_contractionResults,
                      &contractionResults,
                      0);

      }

      {
        const unsigned int numberOfThreadsPerBlock = 256;

        cudaTilingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTeamTest(CudaStyle_Tiling,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfContractions,
                      maxNumberOfContractions,
                      contractionSize,
                      numBasis,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      contractionData_LayoutRight_Right,
                      contractionData_LayoutRight_Left,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_contractionResults,
                      &contractionResults,
                      tile_size);

      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda independent> **********************
      // ===============================================================

      /*
      // ===============================================================
      // ***************** < do cuda reductions> ***********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock =
          std::min(unsigned(1024),
                   unsigned(ceil(contractionSize / 32.)) * 32);

        cudaReduction_TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Reduction,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfContractions,
                      maxNumberOfContractions,
                      contractionSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_contractionData_LayoutRight_Right,
                      dev_contractionData_LayoutRight_Left,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_contractionResults,
                      &contractionResults);

      }
      cudaSwitchingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
        runSwitchingCudaTest(numberOfRepeats,
                             maxNumberOfCudaBlocks,
                             numberOfContractions,
                             maxNumberOfContractions,
                             contractionSize,
                             memorySize,
                             correctResults,
                             clearCacheStyle,
                             dev_junkDataToClearTheCache,
                             junkDataSize,
                             dev_contractionData_LayoutLeft_Right,
                             dev_contractionData_LayoutLeft_Left,
                             dev_contractionData_LayoutRight_Right,
                             dev_contractionData_LayoutRight_Left,
                             dev_junkDataCounter,
                             &totalNumberOfRepeats,
                             dev_contractionResults,
                             &contractionResults);
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda reductions> ***********************
      // ===============================================================
    */
//#ifdef ENABLE_KOKKOS
      // ===============================================================
      // ***************** < do kokkos> ********************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosContractionData;
        kokkosOmpTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosContractionData>(numberOfContractions,
                                              numberOfRepeats,
                                              contractionSize,
					      numBasis,
					      numBasis,
                                              memorySize,
                                              contractionData_LayoutRight_Right,
                                              contractionData_LayoutRight_Left,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosContractionData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaIndependentTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosContractionData>(numberOfContractions,
                                              numberOfRepeats,
                                              contractionSize,
					      numBasis,
					      numBasis,
                                              memorySize,
                                              contractionData_LayoutRight_Right,
                                              contractionData_LayoutRight_Left,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosContractionData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosTeamReductionTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTeamReductionTest<DeviceType,
                        KokkosContractionData>(numberOfContractions,
                                              numberOfRepeats,
                                              contractionSize,
					      numBasis,
					      numBasis,
                                              memorySize,
                                              contractionData_LayoutRight_Right,
                                              contractionData_LayoutRight_Left,
                                              correctResults,
                                              string("Kokkos Team Reduction"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults);

      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosContractionData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosSlicingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosSlicingTest<DeviceType,
                        KokkosContractionData>(numberOfContractions,
                                              numberOfRepeats,
                                              contractionSize,
					      numBasis,
					      numBasis,
                                              memorySize,
                                              contractionData_LayoutRight_Right,
                                              contractionData_LayoutRight_Left,
                                              correctResults,
                                              string("Kokkos Slicing"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosContractionData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosTilingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTilingTest<DeviceType,
                        KokkosContractionData>(numberOfContractions,
                                              numberOfRepeats,
                                              contractionSize,
					      numBasis,
					      numBasis,
                                              memorySize,
                                              contractionData_LayoutRight_Right,
                                              contractionData_LayoutRight_Left,
                                              correctResults,
                                              string("Kokkos Tiling"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults,
                                              tile_size);
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
//#endif // ENABLE_KOKKOS

      contractionSizeMatrix[contractionSizeIndex][memorySizeIndex] =
        contractionSize;
      numberOfContractionsMatrix[contractionSizeIndex][memorySizeIndex] =
        numberOfContractions;
      memorySizeMatrix[contractionSizeIndex][memorySizeIndex] =
        memorySize;
    }

    const timespec thisSizesToc = getTimePoint();
    const float thisSizesElapsedTime =
      getElapsedTime(thisSizesTic, thisSizesToc);
    printf("completed %4u repeats of dot products of size %4u "
           "in %7.2f seconds\n", numberOfRepeats,
           contractionSize, thisSizesElapsedTime);

    /*
    checkCudaError(cudaFree(dev_contractionData_LayoutLeft_Right));
    checkCudaError(cudaFree(dev_contractionData_LayoutLeft_Left));
    checkCudaError(cudaFree(dev_contractionData_LayoutRight_Right));
    checkCudaError(cudaFree(dev_contractionData_LayoutRight_Left));
    */
    checkCudaError(cudaFree(dev_contractionResults));


  }

  writeTimesMatrixToFile(contractionSizeMatrix,
                         prefix + string("contractionSize") + suffix);
  writeTimesMatrixToFile(numberOfContractionsMatrix,
                         prefix + string("numberOfContractions") + suffix);
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

  writeTimesMatrixToFile(cudaTilingTimesMatrix,
                         prefix + string("cudaTiledTimes") + suffix);
//#ifdef ENABLE_KOKKOS
  writeTimesMatrixToFile(kokkosOmpTimesMatrix,
                         prefix + string("kokkosOmpTimes") + suffix);
  writeTimesMatrixToFile(kokkosCudaIndependentTimesMatrix,
                         prefix + string("kokkosCudaIndependentTimes") + suffix);
  writeTimesMatrixToFile(kokkosTeamReductionTimesMatrix,
                         prefix + string("kokkosTeamReductionTimes") + suffix);
  writeTimesMatrixToFile(kokkosSlicingTimesMatrix,
                         prefix + string("kokkosSlicingTimes") + suffix);
  writeTimesMatrixToFile(kokkosTilingTimesMatrix,
                         prefix + string("kokkosTilingTimes") + suffix);
//#endif

//#ifdef ENABLE_KOKKOS
  //const unsigned int numberOfMethods = 7;
//#else
  //const unsigned int numberOfMethods = 5;
//#endif
/*
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
  } else {

    const size_t expectedDataCounter =
      junkDataSum * size_t(numberOfMethods) * (numberOfRepeats + 1) * numberOfMemorySizes *
      numberOfContractionSizes;
    if (junkDataCounter != expectedDataCounter) {
      fprintf(stderr, "for ClearCacheAfterEveryRepeat, invalid "
              "junkDataCounter = %zu (%e), it should be %zu (%e)\n",
              junkDataCounter, float(junkDataCounter),
              expectedDataCounter, float(expectedDataCounter));
      exit(1);
    }

  }


  const unsigned int expectedTotalNumberOfRepeats = numberOfMethods *
    (numberOfRepeats + 1) * numberOfMemorySizes * numberOfContractionSizes;
  if (totalNumberOfRepeats != expectedTotalNumberOfRepeats) {
    fprintf(stderr, "invalid totalNumberOfRepeats = %u (%e), it should be "
            "%u (%e)\n",
            totalNumberOfRepeats, float(totalNumberOfRepeats),
            expectedTotalNumberOfRepeats, float(expectedTotalNumberOfRepeats));
    exit(1);
  }
*/
//#ifdef ENABLE_KOKKOS
  Kokkos::finalize();
//#endif

  return 0;
}
