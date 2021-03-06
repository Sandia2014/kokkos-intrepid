// -*- C++ -*-
// ContractDataFieldVector.cu
// a huge comparison of different ways of doing ContractDataFieldVector
// Tyler Marklyn (outline stolen from Jeff Amelang), 2015

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
#include "../Utilities.hpp"

#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#include "ContractDataFieldVectorFunctors.hpp"
#endif // ENABLE_KOKKOS

enum ClearCacheStyle {ClearCacheAfterEveryRepeat,
                      DontClearCacheAfterEveryRepeat};

// NOTE: Everything in RAW_CUDA guards is still from Array of Dot Products
#ifdef RAW_CUDA
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
doCudaDotProducts_Independent_kernel(const unsigned int numberOfDotProducts,
                                     const unsigned int maxNumberOfDotProducts,
                                     const unsigned int dotProductSize,
                                     const float * const __restrict__ dev_dotProductData_LayoutLeft_A,
                                     const float * const __restrict__ dev_dotProductData_LayoutLeft_B,
                                     float * dev_dotProductResults) {
  unsigned int dotProductIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (dotProductIndex < numberOfDotProducts) {
    float sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < dotProductSize; ++entryIndex) {
      const unsigned int index = dotProductIndex +
        entryIndex * maxNumberOfDotProducts;
      sum +=
        dev_dotProductData_LayoutLeft_A[index] *
        dev_dotProductData_LayoutLeft_B[index];
    }
    dev_dotProductResults[dotProductIndex] = sum;
    dotProductIndex += blockDim.x * gridDim.x;
  }
}

__global__
void
doCudaDotProducts_Reduction_kernel(const unsigned int numberOfDotProducts,
                                   const unsigned int dotProductSize,
                                   const float * const __restrict__ dev_dotProductData_LayoutRight_A,
                                   const float * const __restrict__ dev_dotProductData_LayoutRight_B,
                                   float * dev_dotProductResults) {

  extern __shared__ float sharedMemory[];

  unsigned int dotProductIndex = blockIdx.x;
  while (dotProductIndex < numberOfDotProducts) {

    // goal: compute the contribution to the dot product from this thread
    const unsigned int shortcutIndex = dotProductIndex * dotProductSize;
    float partialSum = 0;
    unsigned int entryIndex = threadIdx.x;
    while (entryIndex < dotProductSize) {
      const unsigned int index = shortcutIndex + entryIndex;
      partialSum +=
        dev_dotProductData_LayoutRight_A[index] *
        dev_dotProductData_LayoutRight_B[index];
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
      atomicAdd(&dev_dotProductResults[dotProductIndex], partialSum);
    }

    // move on to the next dot product
    dotProductIndex += gridDim.x;
  }
}
#endif // RAW_CUDA

#ifdef RAW_CUDA
double
runCudaTest(const CudaStyle cudaStyle,
            const unsigned int numberOfThreadsPerBlock,
            const unsigned int numberOfRepeats,
            const unsigned int maxNumberOfCudaBlocks,
            const unsigned int numberOfDotProducts,
            const unsigned int maxNumberOfDotProducts,
            const unsigned int dotProductSize,
            const unsigned int memorySize,
            const vector<float> & correctResults,
            const ClearCacheStyle clearCacheStyle,
            const int * const dev_junkDataToClearTheCache,
            const unsigned int junkDataSize,
            const float * const dev_dotProductData_A,
            const float * const dev_dotProductData_B,
            int * const dev_junkDataCounter,
            unsigned int * const totalNumberOfRepeats,
            float * const dev_dotProductResults,
            vector<float> * const dotProductResults) {
  const unsigned int numberOfBlocks =
    min(maxNumberOfCudaBlocks,
        (unsigned int)ceil(numberOfDotProducts/float(numberOfThreadsPerBlock)));

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
      doCudaDotProducts_Independent_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(numberOfDotProducts,
                                   maxNumberOfDotProducts,
                                   dotProductSize,
                                   dev_dotProductData_A,
                                   dev_dotProductData_B,
                                   dev_dotProductResults);
    } else if (cudaStyle == CudaStyle_Reduction) {
      doCudaDotProducts_Reduction_kernel<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        numberOfThreadsPerBlock * sizeof(float)>>>(numberOfDotProducts,
                                                   dotProductSize,
                                                   dev_dotProductData_A,
                                                   dev_dotProductData_B,
                                                   dev_dotProductResults);
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
  checkCudaError(cudaMemcpy(&dotProductResults->at(0), dev_dotProductResults,
                            numberOfDotProducts * sizeof(float),
                            cudaMemcpyDeviceToHost));
  // check the results
  checkAnswer(correctResults, *dotProductResults,
              dotProductSize, memorySize,
              convertCudaStyleToString(cudaStyle));

  // scrub the results
  std::fill(dotProductResults->begin(),
            dotProductResults->end(),
            std::numeric_limits<float>::quiet_NaN());
  checkCudaError(cudaMemcpy(dev_dotProductResults, &dotProductResults->at(0),
                            numberOfDotProducts * sizeof(float),
                            cudaMemcpyHostToDevice));

  return totalElapsedTime;
}

double
runSwitchingCudaTest(const unsigned int numberOfRepeats,
                     const unsigned int maxNumberOfCudaBlocks,
                     const unsigned int numberOfDotProducts,
                     const unsigned int maxNumberOfDotProducts,
                     const unsigned int dotProductSize,
                     const unsigned int memorySize,
                     const vector<float> & correctResults,
                     const ClearCacheStyle clearCacheStyle,
                     const int * const dev_junkDataToClearTheCache,
                     const unsigned int junkDataSize,
                     const float * const dev_dotProductData_LayoutLeft_A,
                     const float * const dev_dotProductData_LayoutLeft_B,
                     const float * const dev_dotProductData_LayoutRight_A,
                     const float * const dev_dotProductData_LayoutRight_B,
                     int * const dev_junkDataCounter,
                     unsigned int * const totalNumberOfRepeats,
                     float * const dev_dotProductResults,
                     vector<float> * const dotProductResults) {
  // if i can't saturate occupancy, do the reduction version
  // i got this number by just looking at where the plots crossed, where
  //  the reduction style actually starts beating the independent.
  if (numberOfDotProducts < 200) {
    const unsigned int numberOfThreadsPerBlock =
      std::min(unsigned(1024),
               unsigned(ceil(dotProductSize / 32.)) * 32);
    return
      runCudaTest(CudaStyle_Reduction,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfDotProducts,
                  maxNumberOfDotProducts,
                  dotProductSize,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  dev_dotProductData_LayoutRight_A,
                  dev_dotProductData_LayoutRight_B,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_dotProductResults,
                  dotProductResults);
  } else {
    const unsigned int numberOfThreadsPerBlock = 1024;
    return
      runCudaTest(CudaStyle_Independent,
                  numberOfThreadsPerBlock,
                  numberOfRepeats,
                  maxNumberOfCudaBlocks,
                  numberOfDotProducts,
                  maxNumberOfDotProducts,
                  dotProductSize,
                  memorySize,
                  correctResults,
                  clearCacheStyle,
                  dev_junkDataToClearTheCache,
                  junkDataSize,
                  dev_dotProductData_LayoutLeft_A,
                  dev_dotProductData_LayoutLeft_B,
                  dev_junkDataCounter,
                  totalNumberOfRepeats,
                  dev_dotProductResults,
                  dotProductResults);
  }
}
#endif // RAW_CUDA



#ifdef ENABLE_KOKKOS


template <class DeviceType, class KokkosDataA, class KokkosDataB>
double
runKokkosTest(const unsigned int numberOfRepeats,
              const unsigned int memorySize,
              const unsigned int numCells,
              const unsigned int numFields,
              const unsigned int numPoints,
              const unsigned int dimVec,
              const vector<float> & dotProductData_LayoutRight_A,
              const vector<float> & dotProductData_LayoutRight_B,
              const vector<float> & correctResults,
              const string & kokkosFlavor,
              const ClearCacheStyle clearCacheStyle,
              const vector<int> & junkDataToClearTheCache,
              size_t * junkDataCounter,
              unsigned int * const totalNumberOfRepeats,
              vector<float> * dotProductResults) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef typename KokkosDataA::HostMirror              KokkosDataA_Host;
  typedef typename KokkosDataB::HostMirror              KokkosDataB_Host;
  typedef Kokkos::View<float**, DeviceType>             KokkosDotProductResults;
  typedef typename KokkosDotProductResults::HostMirror  KokkosDotProductResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  KokkosDataA dev_kokkosDotProductData_A("kokkos data A",
                                         numCells, numFields, numPoints, dimVec);
  KokkosDataA_Host kokkosDotProductData_A =
    Kokkos::create_mirror_view(dev_kokkosDotProductData_A);

  KokkosDataB dev_kokkosDotProductData_B("kokkos data B",
                                         numCells, numPoints, dimVec);
  KokkosDataB_Host kokkosDotProductData_B =
    Kokkos::create_mirror_view(dev_kokkosDotProductData_B);

  KokkosDotProductResults dev_kokkosDotProductResults("kokkos dot product results",
                                                      numCells, numFields);
  KokkosDotProductResults_Host kokkosDotProductResults =
    Kokkos::create_mirror_view(dev_kokkosDotProductResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);

  // copy the data into the device views and ship them over
  for (int cl = 0; cl < numCells; ++cl) {
    int clDim = cl * numFields * numPoints * dimVec;
    for (int lbf = 0; lbf < numFields; ++lbf) {
      int lbfDim = lbf * numPoints * dimVec;
      for (int qp = 0; qp < numPoints; ++qp) {
        int qpDim = qp * dimVec;
        for (int iVec = 0; iVec < dimVec; ++iVec) {
          kokkosDotProductData_A(cl, lbf, qp, iVec) = 
            dotProductData_LayoutRight_A[clDim + lbfDim + qpDim + iVec];
        }
      }
    }
  }

  for (int cl = 0; cl < numCells; ++cl) {
    int clDim = cl * numPoints * dimVec;
    for (int qp = 0; qp < numPoints; ++qp) {
      int qpDim = qp * dimVec;
      for (int iVec = 0; iVec < dimVec; ++iVec) {
        kokkosDotProductData_B(cl, qp, iVec) = 
          dotProductData_LayoutRight_B[clDim + qpDim + iVec];
      }
    }
  }
  Kokkos::deep_copy(dev_kokkosDotProductData_A, kokkosDotProductData_A);
  Kokkos::deep_copy(dev_kokkosDotProductData_B, kokkosDotProductData_B);

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
  KokkosFunctor_Independent<DeviceType,
                            KokkosDataA,
                            KokkosDataB,
                            KokkosDotProductResults>
    kokkosFunctor_Independent(numCells,
                              numPoints,
                              dimVec,
                              dev_kokkosDotProductData_A,
                              dev_kokkosDotProductData_B,
                              dev_kokkosDotProductResults);

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
    Kokkos::parallel_for(numCells * numFields, kokkosFunctor_Independent);

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
  Kokkos::deep_copy(kokkosDotProductResults, dev_kokkosDotProductResults);
  for (unsigned int cl = 0; cl < numCells; ++cl) {
    const unsigned int clInd = cl * numFields;
    for (unsigned int lbf = 0; lbf < numFields; ++lbf) {
        dotProductResults->at(clInd + lbf) =
          kokkosDotProductResults(cl, lbf);
      }
  }
  
  // check the results
  checkAnswer(correctResults, *dotProductResults,
              numPoints * dimVec, memorySize,
              kokkosFlavor);

  // scrub the results
  std::fill(dotProductResults->begin(),
            dotProductResults->end(),
            std::numeric_limits<float>::quiet_NaN());

  return totalElapsedTime;
}

#endif // ENABLE_KOKKOS



int main(int argc, char* argv[]) {

#ifdef ENABLE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

  // ===============================================================
  // ********************** < input> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const vector<unsigned int> dotProductSizes =
    {{8, 16, 32, 64, 128, 256, 512, 1024, 2048}};
  const array<float, 2> memorySizeExtrema = {{1e6, 1e9}};
  const unsigned int numberOfMemorySizes = 20;
#ifdef RAW_CUDA
  const unsigned int maxNumberOfCudaBlocks = unsigned(1e4);
#endif
  const ClearCacheStyle clearCacheStyle =
    ClearCacheAfterEveryRepeat;
  const unsigned int numberOfRepeats =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? 10 : 250;
  const string machineName = "shadowfax";
  const string prefix = "data/ContractDataFieldVector_";
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </input> ******************************
  // ===============================================================

  // derive some values from the inputs
  const unsigned int numberOfDotProductSizes = dotProductSizes.size();
  const string clearCacheStyleString =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? "clearCache" :
    "dontClearCache";
  const string suffix = "_" + clearCacheStyleString + "_" + machineName;

  // create the actual sizes
  vector<unsigned int> memorySizes(numberOfMemorySizes);
  const unsigned int numFields = 8;
  for (unsigned int memorySizeIndex = 0;
       memorySizeIndex < numberOfMemorySizes; ++memorySizeIndex) {
    const float percent = memorySizeIndex / float(numberOfMemorySizes - 1);
    const float minLog = log10(memorySizeExtrema[0]);
    const float maxLog = log10(memorySizeExtrema[1]);
    const float thisLog = minLog + percent * (maxLog - minLog);
    const unsigned int maxDotProductSize = dotProductSizes.back();
    // memory size is linear on a log scale, but rounded to a multiple of the
    //  largest dot product size
    const unsigned int desiredMemorySizeInBytes = pow(10., thisLog);
    // now, in this amount of memory i have to fit two vectors of data
    // that are multiples of the max dot product size, and it has to be divisible by numFields
    const unsigned int memorySizeInBytes =
      unsigned(desiredMemorySizeInBytes /
               float(4 * sizeof(float) * maxDotProductSize * numFields)) *
      4 * sizeof(float) * maxDotProductSize * numFields;
    memorySizes[memorySizeIndex] = memorySizeInBytes;
  }

  // create a c++11 random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  // these are just containers for storing the numbers we'll be plotting.
  // i feel a little dirty using a vector<vector>, but i don't want to introduce
  //  a dependence on eigen or something for a real matrix.
  vector<vector<float> >
    dotProductSizeMatrix(numberOfDotProductSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    numberOfDotProductsMatrix(numberOfDotProductSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    memorySizeMatrix(numberOfDotProductSizes,
                     vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    serialTimesMatrix(numberOfDotProductSizes,
                      vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    ompUncollapsedTimesMatrix(numberOfDotProductSizes,
                   vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    ompCollapsedTimesMatrix(numberOfDotProductSizes,
                   vector<float>(numberOfMemorySizes, 0));

#ifdef RAW_CUDA
  vector<vector<float> >
    cudaIndependent_TimesMatrix(numberOfDotProductSizes,
                                vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaReduction_TimesMatrix(numberOfDotProductSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaSwitchingTimesMatrix(numberOfDotProductSizes,
                             vector<float>(numberOfMemorySizes, 0));
#endif

#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfDotProductSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfDotProductSizes,
                                     vector<float>(numberOfMemorySizes, 0));
#endif

  // create some junk data to use in clearing the cache
  size_t junkDataCounter = 0;
  const size_t junkDataSize = 1e7;
  vector<int> junkDataToClearTheCache(junkDataSize, 0);
  for (unsigned int i = 0; i < junkDataSize/100; ++i) {
    junkDataToClearTheCache[(rand() / float(RAND_MAX))*junkDataSize] = 1;
  }

#ifdef RAW_CUDA
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
#endif // RAW_CUDA

  unsigned int totalNumberOfRepeats = 0;

  // for each dot product size
  for (unsigned int dotProductSizeIndex = 0;
       dotProductSizeIndex < numberOfDotProductSizes;
       ++dotProductSizeIndex) {
    const unsigned int dotProductSize = dotProductSizes[dotProductSizeIndex];
    const unsigned int dimVec = 8;
    const unsigned int numPoints = dotProductSize / dimVec;

    const timespec thisSizesTic = getTimePoint();

    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumCells =
      memorySizes.back() / 4 / sizeof(float) / dotProductSize / numFields;
    vector<float> dotProductData_LayoutRight_A(maxNumCells * numFields * dotProductSize);
    vector<float> dotProductData_LayoutRight_B(maxNumCells * dotProductSize);
#ifdef RAW_CUDA
    vector<float> dotProductData_LayoutLeft_A(dotProductData_LayoutRight_A.size());
    vector<float> dotProductData_LayoutLeft_B(dotProductData_LayoutRight_B.size());
#endif
    for (unsigned int cl = 0; cl < maxNumCells; ++cl) {
      for (unsigned int lbf = 0; lbf < numFields; ++lbf) {
        for (unsigned int entryIndex = 0;
             entryIndex < dotProductSize; ++entryIndex) {
          const unsigned int ALayoutRightIndex =
            cl * numFields * dotProductSize + lbf * dotProductSize + entryIndex;
          const unsigned int BLayoutRightIndex =
            cl * dotProductSize + entryIndex;
          dotProductData_LayoutRight_A[ALayoutRightIndex] =
            randomNumberGenerator(randomNumberEngine);
          dotProductData_LayoutRight_B[BLayoutRightIndex] =
            randomNumberGenerator(randomNumberEngine);
#ifdef RAW_CUDA
        const unsigned int layoutLeftIndex =
          entryIndex * maxNumberOfDotProducts + dotProductIndex;
        dotProductData_LayoutLeft_A[layoutLeftIndex] =
          dotProductData_LayoutRight_A[layoutRightIndex];
        dotProductData_LayoutLeft_B[layoutLeftIndex] =
          dotProductData_LayoutRight_B[layoutRightIndex];
#endif
      }
    }
  }
    // Results will be layout right
    vector<float> dotProductResults(maxNumCells * numFields,
                                    std::numeric_limits<float>::quiet_NaN());

#ifdef RAW_CUDA
    // now, because we'll be working with cuda stuff, also allocate the inputs
    //  and output on the gpu and copy them over
    float * dev_dotProductData_LayoutRight_A;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutRight_A,
                              maxNumberOfDotProducts * dotProductSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutRight_A,
                              &dotProductData_LayoutRight_A[0],
                              maxNumberOfDotProducts * dotProductSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_dotProductData_LayoutRight_B;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutRight_B,
                              maxNumberOfDotProducts * dotProductSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutRight_B,
                              &dotProductData_LayoutRight_B[0],
                              maxNumberOfDotProducts * dotProductSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_dotProductResults;
    checkCudaError(cudaMalloc((void **) &dev_dotProductResults,
                              maxNumberOfDotProducts * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductResults, &dotProductResults[0],
                              maxNumberOfDotProducts * sizeof(float),
                              cudaMemcpyHostToDevice));
    // make and populate the LayoutLeft versions
    float * dev_dotProductData_LayoutLeft_A;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutLeft_A,
                              maxNumberOfDotProducts * dotProductSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutLeft_A,
                              &dotProductData_LayoutLeft_A[0],
                              maxNumberOfDotProducts * dotProductSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_dotProductData_LayoutLeft_B;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutLeft_B,
                              maxNumberOfDotProducts * dotProductSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutLeft_B,
                              &dotProductData_LayoutLeft_B[0],
                              maxNumberOfDotProducts * dotProductSize * sizeof(float),
                              cudaMemcpyHostToDevice));
#endif // RAW_CUDA

    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex];
      const unsigned int numCells =
        memorySize / 4 / sizeof(float) / dotProductSize / numFields;
      if (memorySize != 4 * sizeof(float) * numCells * numFields * dotProductSize) {
        fprintf(stderr, "invalid memory size of %u for dot product size of "
                "%u because it doesn't divide evenly, remainder is %zu\n",
                memorySize, dotProductSize,
                memorySize % (4 * sizeof(float) * dotProductSize));
        exit(1);
      }

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
          for (int cl = 0; cl < numCells; cl++) {
            int clDimLeft = cl * numFields * numPoints * dimVec;
            int clDimRight = cl * numPoints * dimVec;
            int clDimOut = cl * numFields;
            for (int lbf = 0; lbf < numFields; lbf++) {
              int lbfDim = lbf * numPoints * dimVec;
              float tmpVal = 0;
              for (int qp = 0; qp < numPoints; qp++) {
                int qpDim = qp * dimVec;
                for (int iVec = 0; iVec < dimVec; iVec++) {
                  tmpVal +=
                    dotProductData_LayoutRight_A[clDimLeft + lbfDim + qpDim + iVec] *
                    dotProductData_LayoutRight_B[clDimRight + qpDim + iVec];
                } // D-loop
              } // P-loop
              dotProductResults[clDimOut + lbf] = tmpVal;
            } // F-loop
          } // C-loop

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            serialTimesMatrix[dotProductSizeIndex][memorySizeIndex] += elapsedTime;

            junkDataCounter +=
              std::accumulate(junkDataToClearTheCache.begin(),
                              junkDataToClearTheCache.end(), size_t(0));
          }
        }
        if (clearCacheStyle == DontClearCacheAfterEveryRepeat) {
          const timespec toc = getTimePoint();
          const float elapsedTime = getElapsedTime(tic, toc) / numberOfRepeats;
          serialTimesMatrix[dotProductSizeIndex][memorySizeIndex] = elapsedTime;
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do serial> ***************************
      // ===============================================================

      const vector<float> correctResults = dotProductResults;
      // scrub the results
      std::fill(dotProductResults.begin(),
                dotProductResults.end(),
                std::numeric_limits<float>::quiet_NaN());

      // ===============================================================
      // ********************** < do omp uncollapsed> ******************
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

#pragma omp parallel for default(none)                                  \
  shared(dotProductData_LayoutRight_A, dotProductData_LayoutRight_B,    \
         dotProductResults)
          for (int cl = 0; cl < numCells; cl++) {
            int clDimLeft = cl * numFields * numPoints * dimVec;
            int clDimRight = cl * numPoints * dimVec;
            int clDimOut = cl * numFields;
            for (int lbf = 0; lbf < numFields; lbf++) {
              int lbfDim = lbf * numPoints * dimVec;
              float tmpVal = 0;
              for (int qp = 0; qp < numPoints; qp++) {
                int qpDim = qp * dimVec;
                for (int iVec = 0; iVec < dimVec; iVec++) {
                  tmpVal +=
                    dotProductData_LayoutRight_A[clDimLeft + lbfDim + qpDim + iVec] *
                    dotProductData_LayoutRight_B[clDimRight + qpDim + iVec];
                } // D-loop
              } // P-loop
              dotProductResults[clDimOut + lbf] = tmpVal;
            } // F-loop
          } // C-loop

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompUncollapsedTimesMatrix[dotProductSizeIndex][memorySizeIndex] += elapsedTime;

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
          ompUncollapsedTimesMatrix[dotProductSizeIndex][memorySizeIndex] = elapsedTime;
          // check the results
          checkAnswer(correctResults, dotProductResults,
                      dotProductSize, memorySize,
                      string("omp uncollapsed"));
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp uncollapsed> ******************
      // ===============================================================

      // scrub the results
      std::fill(dotProductResults.begin(),
                dotProductResults.end(),
                std::numeric_limits<float>::quiet_NaN());

      // ===============================================================
      // ********************** < do omp collapsed> ********************
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

#pragma omp parallel for default(none)                                  \
  shared(dotProductData_LayoutRight_A, dotProductData_LayoutRight_B,    \
         dotProductResults)
          for (int elementIndex = 0; elementIndex < numCells * numFields; ++elementIndex) {
            int cl = elementIndex % numCells;
            int lbf = elementIndex / numCells;

            int clDimLeft = cl * numFields * numPoints * dimVec;
            int clDimRight = cl * numPoints * dimVec;
            int clDimOut = cl * numFields;

            int lbfDim = lbf * numPoints * dimVec;

            float tmpVal = 0;
            for (int qp = 0; qp < numPoints; qp++) {
              int qpDim = qp * dimVec;
              for (int iVec = 0; iVec < dimVec; iVec++) {
                tmpVal += 
                  dotProductData_LayoutRight_A[clDimLeft + lbfDim + qpDim + iVec] *
                  dotProductData_LayoutRight_B[clDimRight + qpDim + iVec];
              } // D-loop
            } // P-loop
            dotProductResults[clDimOut + lbf] = tmpVal;
          }

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompCollapsedTimesMatrix[dotProductSizeIndex][memorySizeIndex] += elapsedTime;

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
          ompCollapsedTimesMatrix[dotProductSizeIndex][memorySizeIndex] = elapsedTime;
          // check the results
          checkAnswer(correctResults, dotProductResults,
                      dotProductSize, memorySize,
                      string("omp collapsed"));
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp collapsed> ********************
      // ===============================================================

      // scrub the results
      std::fill(dotProductResults.begin(),
                dotProductResults.end(),
                std::numeric_limits<float>::quiet_NaN());

#ifdef RAW_CUDA
      checkCudaError(cudaMemcpy(dev_dotProductResults, &dotProductResults[0],
                                maxNumberOfDotProducts * sizeof(float),
                                cudaMemcpyHostToDevice));

      // ===============================================================
      // ***************** < do cuda independent> **********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock = 1024;

        cudaIndependent_TimesMatrix[dotProductSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Independent,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfDotProducts,
                      maxNumberOfDotProducts,
                      dotProductSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_dotProductData_LayoutLeft_A,
                      dev_dotProductData_LayoutLeft_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_dotProductResults,
                      &dotProductResults);

      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda independent> **********************
      // ===============================================================

      // ===============================================================
      // ***************** < do cuda reductions> ***********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock =
          std::min(unsigned(1024),
                   unsigned(ceil(dotProductSize / 32.)) * 32);

        cudaReduction_TimesMatrix[dotProductSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Reduction,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfDotProducts,
                      maxNumberOfDotProducts,
                      dotProductSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_dotProductData_LayoutRight_A,
                      dev_dotProductData_LayoutRight_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_dotProductResults,
                      &dotProductResults);

      }
      cudaSwitchingTimesMatrix[dotProductSizeIndex][memorySizeIndex] =
        runSwitchingCudaTest(numberOfRepeats,
                             maxNumberOfCudaBlocks,
                             numberOfDotProducts,
                             maxNumberOfDotProducts,
                             dotProductSize,
                             memorySize,
                             correctResults,
                             clearCacheStyle,
                             dev_junkDataToClearTheCache,
                             junkDataSize,
                             dev_dotProductData_LayoutLeft_A,
                             dev_dotProductData_LayoutLeft_B,
                             dev_dotProductData_LayoutRight_A,
                             dev_dotProductData_LayoutRight_B,
                             dev_junkDataCounter,
                             &totalNumberOfRepeats,
                             dev_dotProductResults,
                             &dotProductResults);
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do cuda reductions> ***********************
      // ===============================================================
#endif // RAW_CUDA

#ifdef ENABLE_KOKKOS
      // ===============================================================
      // ***************** < do kokkos> ********************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosDataA;
        typedef Kokkos::View<float***, Kokkos::LayoutRight,
                             DeviceType>                   KokkosDataB;
        kokkosOmpTimesMatrix[dotProductSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType, KokkosDataA, 
                                 KokkosDataB>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numFields,
                                              numPoints,
                                              dimVec,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &dotProductResults);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosDataA;
        typedef Kokkos::View<float***, Kokkos::LayoutLeft,
                             DeviceType>                   KokkosDataB;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaIndependentTimesMatrix[dotProductSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType, KokkosDataA, 
                                 KokkosDataB>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numFields,
                                              numPoints,
                                              dimVec,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &dotProductResults);
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
#endif // ENABLE_KOKKOS

      dotProductSizeMatrix[dotProductSizeIndex][memorySizeIndex] =
        dotProductSize;
      numberOfDotProductsMatrix[dotProductSizeIndex][memorySizeIndex] =
        numCells;
      memorySizeMatrix[dotProductSizeIndex][memorySizeIndex] =
        memorySize;

    }

    const timespec thisSizesToc = getTimePoint();
    const float thisSizesElapsedTime =
      getElapsedTime(thisSizesTic, thisSizesToc);
    printf("completed %4u repeats of dot products of size %4u "
           "in %7.2f seconds\n", numberOfRepeats,
           dotProductSize, thisSizesElapsedTime);

#ifdef RAW_CUDA
    checkCudaError(cudaFree(dev_dotProductData_LayoutLeft_A));
    checkCudaError(cudaFree(dev_dotProductData_LayoutLeft_B));
    checkCudaError(cudaFree(dev_dotProductData_LayoutRight_A));
    checkCudaError(cudaFree(dev_dotProductData_LayoutRight_B));
    checkCudaError(cudaFree(dev_dotProductResults));
#endif

  }
  writeTimesMatrixToFile(dotProductSizeMatrix,
                         prefix + string("dotProductSize") + suffix);
  writeTimesMatrixToFile(numberOfDotProductsMatrix,
                         prefix + string("numberOfDotProducts") + suffix);
  writeTimesMatrixToFile(memorySizeMatrix,
                         prefix + string("memorySize") + suffix);
  writeTimesMatrixToFile(serialTimesMatrix,
                         prefix + string("serialTimes") + suffix);
  writeTimesMatrixToFile(ompUncollapsedTimesMatrix,
                         prefix + string("ompUncollapsedTimes") + suffix);
  writeTimesMatrixToFile(ompCollapsedTimesMatrix,
                         prefix + string("ompTimes") + suffix);


#ifdef RAW_CUDA
  writeTimesMatrixToFile(cudaIndependent_TimesMatrix,
                         prefix + string("cudaIndependentTimes") + suffix);
  writeTimesMatrixToFile(cudaReduction_TimesMatrix,
                         prefix + string("cudaReductionTimes") + suffix);
  writeTimesMatrixToFile(cudaSwitchingTimesMatrix,
                         prefix + string("cudaSwitchingTimes") + suffix);
#endif

#ifdef ENABLE_KOKKOS
  writeTimesMatrixToFile(kokkosOmpTimesMatrix,
                         prefix + string("kokkosOmpTimes") + suffix);
  writeTimesMatrixToFile(kokkosCudaIndependentTimesMatrix,
                         prefix + string("kokkosCudaIndependentTimes") + suffix);
#endif

#if defined RAW_CUDA
  // Note, we assume that if RAW_CUDA is defined so is ENABLE_KOKKOS here
  const unsigned int numberOfMethods = 8;
#elif defined ENABLE_KOKKOS
  const unsigned int numberOfMethods = 5;
#else
  const unsigned int numberOfMethods = 3;
#endif

  const size_t junkDataSum =
    std::accumulate(junkDataToClearTheCache.begin(),
                    junkDataToClearTheCache.end(), size_t(0));
  {
    int temp = 0;
#ifdef RAW_CUDA
    checkCudaError(cudaMemcpy(&temp,
                              dev_junkDataCounter,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
#endif
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
      numberOfDotProductSizes;
    if (junkDataCounter != expectedDataCounter) {
      fprintf(stderr, "for ClearCacheAfterEveryRepeat, invalid "
              "junkDataCounter = %zu (%e), it should be %zu (%e)\n",
              junkDataCounter, float(junkDataCounter),
              expectedDataCounter, float(expectedDataCounter));
      exit(1);
    }
  }

  const unsigned int expectedTotalNumberOfRepeats = numberOfMethods *
    (numberOfRepeats + 1) * numberOfMemorySizes * numberOfDotProductSizes;
  if (totalNumberOfRepeats != expectedTotalNumberOfRepeats) {
    fprintf(stderr, "invalid totalNumberOfRepeats = %u (%e), it should be "
            "%u (%e)\n",
            totalNumberOfRepeats, float(totalNumberOfRepeats),
            expectedTotalNumberOfRepeats, float(expectedTotalNumberOfRepeats));
    exit(1);
  }

#ifdef ENABLE_KOKKOS
  Kokkos::finalize();
#endif

  return 0;
}
