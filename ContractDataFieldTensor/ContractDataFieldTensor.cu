// -*- C++ -*-
// ArrayOfDotProducts.cc
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
#include "../Utilities.hpp"

#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#include "ContractDataFieldTensorFunctors.hpp"
#endif // ENABLE_KOKKOS

enum ClearCacheStyle {ClearCacheAfterEveryRepeat,
                      DontClearCacheAfterEveryRepeat};


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





#ifdef ENABLE_KOKKOS


template <class DeviceType, class KokkosDotProductData_Left, class KokkosDotProductData_Right>
double
runKokkosTest(const unsigned int cellSize,
	      const unsigned int numberOfRepeats,
              const unsigned int memorySize,
              const unsigned int numCells,
              const unsigned int l,
              const unsigned int q,
              const unsigned int d1,
              const unsigned int d2,
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

  typedef typename KokkosDotProductData_Left::HostMirror     KokkosDotProductData_Host_Left;
  typedef typename KokkosDotProductData_Right::HostMirror     KokkosDotProductData_Host_Right;

  typedef Kokkos::View<float**, DeviceType>              KokkosDotProductResults;
  typedef typename KokkosDotProductResults::HostMirror  KokkosDotProductResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  KokkosDotProductData_Left dev_kokkosDotProductData_A("kokkos data A",
                                                  numCells,
                                                  l, q, d1, d2);
  KokkosDotProductData_Host_Left kokkosDotProductData_A =
    Kokkos::create_mirror_view(dev_kokkosDotProductData_A);

  KokkosDotProductData_Right dev_kokkosDotProductData_B("kokkos data B",
                                                  numCells,
                                                  q,
                                                  d1,
                                                  d2);
  KokkosDotProductData_Host_Right kokkosDotProductData_B =
    Kokkos::create_mirror_view(dev_kokkosDotProductData_B);

  KokkosDotProductResults dev_kokkosDotProductResults("kokkos dot product results",
                                                      numCells,l);
  KokkosDotProductResults_Host kokkosDotProductResults =
    Kokkos::create_mirror_view(dev_kokkosDotProductResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);

  // copy the data into the device views and ship them over
  for (unsigned int dotProductIndex = 0;
       dotProductIndex < numCells; ++dotProductIndex) {
    for (unsigned int lbf = 0; lbf < l; ++lbf) {
      for(unsigned int qp = 0; qp < q; ++qp) {
        for(unsigned int iTens1 = 0; iTens1 < d1; ++iTens1) {
          for(unsigned int iTens2 = 0; iTens2 < d2; ++iTens2) {
            kokkosDotProductData_A(dotProductIndex, lbf, qp, iTens1, iTens2) =
            dotProductData_LayoutRight_A[dotProductIndex * l * q * d1 * d2 +
            lbf * q * d1 * d2 + qp * d1 * d2 + iTens1 * d2 + iTens2];
            kokkosDotProductData_B(dotProductIndex, qp, iTens1, iTens2) =
            dotProductData_LayoutRight_B[dotProductIndex * q * d1 * d2 +
            qp * d1 * d2 + iTens1 * d2 + iTens2];
          }
        }
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
    KokkosFunctor_Independent<DeviceType, KokkosDotProductData_Left,
                            KokkosDotProductData_Right, KokkosDotProductResults>
    kokkosFunctor_Independent(dev_kokkosDotProductData_A,
    dev_kokkosDotProductData_B,
    dev_kokkosDotProductResults,
    numCells, l, q, d1, d2);

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
    Kokkos::parallel_for(numCells, kokkosFunctor_Independent);

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
  for (unsigned int dotProductIndex = 0;
       dotProductIndex < numCells; ++dotProductIndex) {
         for(unsigned int lbf = 0; lbf < l; ++lbf) {
    dotProductResults->at(dotProductIndex*l+ lbf) =
      kokkosDotProductResults(dotProductIndex, lbf);
    }
  }
  // check the results
  checkAnswer(correctResults, *dotProductResults,
              cellSize, memorySize,
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
  const vector<unsigned int> contractionSizes =
    {{8, 27, 64, 125, 216, 343, 512, 729, 1000}};
  const array<float, 2> memorySizeExtrema = {{1e6, 1e9}};
  const unsigned int numberOfMemorySizes = 20;
  // const unsigned int maxNumberOfCudaBlocks = unsigned(1e4);
  const unsigned int l = 8;

  unsigned int cellSize;
  const ClearCacheStyle clearCacheStyle =
    ClearCacheAfterEveryRepeat;
  const unsigned int numberOfRepeats =
    (clearCacheStyle == ClearCacheAfterEveryRepeat) ? 10 : 250;
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
  vector<vector <unsigned int>> memorySizes(numberOfMemorySizes,
                                          vector<unsigned int>(numberOfContractionSizes, 0));
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
    for(unsigned int contractionIndex = 0; contractionIndex < numberOfContractionSizes;
        ++contractionIndex){

          cellSize = 2*l*(contractionSizes[contractionIndex])*sizeof(float)
                    + 2*contractionSizes[contractionIndex]*sizeof(float);
          const unsigned int memorySizeInBytes =
          unsigned(desiredMemorySizeInBytes /
               cellSize) * cellSize;
          (memorySizes[memorySizeIndex]).at(contractionIndex) = memorySizeInBytes;
   }
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
#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfContractionSizes,
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
  for (unsigned int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {
    const unsigned int contractionSize = contractionSizes[contractionSizeIndex];
    
    const unsigned int q = pow(contractionSize, 1/3);
    const unsigned int d1 = q;
    const unsigned int d2 = q;
    const timespec thisSizesTic = getTimePoint();

    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumberOfContractions =
      memorySizes.back().at(contractionSizeIndex) / 4 / sizeof(float) / contractionSize;
    vector<float> contractionData_LayoutRight_A(maxNumberOfContractions * l * contractionSize);
    vector<float> contractionData_LayoutRight_B(maxNumberOfContractions * contractionSize);
    vector<float> contractionData_LayoutLeft_A(contractionData_LayoutRight_A.size());
    vector<float> contractionData_LayoutLeft_B(contractionData_LayoutRight_B.size());

    for (unsigned int contractionIndex = 0;
         contractionIndex < maxNumberOfContractions; ++contractionIndex) {
      for (unsigned int entryIndex = 0;
           entryIndex < contractionSize; ++entryIndex) {

        const unsigned int layoutRightIndex_B =
          contractionIndex * contractionSize + entryIndex;

        const unsigned int layoutLeftIndex_B =
          entryIndex * contractionSize + contractionIndex;

        contractionData_LayoutRight_B[layoutRightIndex_B] =
          randomNumberGenerator(randomNumberEngine);

        contractionData_LayoutLeft_B[layoutLeftIndex_B] =
          randomNumberGenerator(randomNumberEngine);

        for(unsigned int lbf = 0; lbf < l; ++lbf) {

          const unsigned int layoutRightIndex_A =
          contractionIndex * contractionSize * l + lbf * contractionSize + entryIndex;

          const unsigned int layoutLeftIndex_A =
          entryIndex * contractionSize * l + lbf * contractionSize + contractionIndex;

          contractionData_LayoutRight_A[layoutRightIndex_A] =
            randomNumberGenerator(randomNumberEngine);

          contractionData_LayoutLeft_A[layoutLeftIndex_A] =
            randomNumberGenerator(randomNumberEngine);
        }
      }
    }
    vector<float> contractionResults(maxNumberOfContractions*l,
                                    std::numeric_limits<float>::quiet_NaN());


    // now, because we'll be working with cuda stuff, also allocate the inputs
    //  and output on the gpu and copy them over
    float * dev_contractionData_LayoutRight_A;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutRight_A,
                              maxNumberOfContractions * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutRight_A,
                              &contractionData_LayoutRight_A[0],
                              maxNumberOfContractions * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_LayoutRight_B;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutRight_B,
                              maxNumberOfContractions * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutRight_B,
                              &contractionData_LayoutRight_B[0],
                              maxNumberOfContractions * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_contractionResults;
    checkCudaError(cudaMalloc((void **) &dev_contractionResults,
                              maxNumberOfContractions * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionResults, &contractionResults[0],
                              maxNumberOfContractions * sizeof(float),
                              cudaMemcpyHostToDevice));
    // make and populate the LayoutLeft versions
    float * dev_contractionData_LayoutLeft_A;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutLeft_A,
                              maxNumberOfContractions * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutLeft_A,
                              &contractionData_LayoutLeft_A[0],
                              maxNumberOfContractions * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_contractionData_LayoutLeft_B;
    checkCudaError(cudaMalloc((void **) &dev_contractionData_LayoutLeft_B,
                              maxNumberOfContractions * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_contractionData_LayoutLeft_B,
                              &contractionData_LayoutLeft_B[0],
                              maxNumberOfContractions * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));

    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex].at(contractionSizeIndex);
      const unsigned int numCells =
       memorySize / (2*sizeof(float)*contractionSize+2*sizeof(float)*l*contractionSize);
      if (memorySize != numCells * (2*sizeof(float)*contractionSize + 2*sizeof(float)*l*contractionSize)) {
        fprintf(stderr, "invalid memory size of %u for dot product size of "
                "%u because it doesn't divide evenly, remainder is %zu\n",
                memorySize, contractionSize,
                memorySize % (4 * sizeof(float) * contractionSize));
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

          for (int cl = 0; cl < numCells; cl++) {
            for (int lbf = 0; lbf < l; lbf++) {
              double tmpVal = 0;
              for (int qp = 0; qp < q; qp++) {
                for (int iTens1 = 0; iTens1 < d1; iTens1++) {
                  for (int iTens2 =0; iTens2 < d2; iTens2++) {
                    tmpVal += contractionData_LayoutRight_A[cl*l*q*d1*d2+lbf*q*d1*d2+qp*d1*d2+iTens1*d2+iTens2] *
                    contractionData_LayoutRight_B[cl*q*d1*d2+qp*d1*d2+iTens1*d2+iTens2];
                  } // D2-loop
                } // D1-loop
              } // P-loop
              contractionResults[cl*l+ lbf] = tmpVal;
            } // F-loop
          } // C-loop

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
     /*  {

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
  shared(dotProductData_LayoutRight_A, dotProductData_LayoutRight_B,    \
         dotProductResults)
          for (unsigned int dotProductIndex = 0;
               dotProductIndex < numberOfDotProducts;
               ++dotProductIndex) {
            const unsigned int shortcutIndex = dotProductIndex * dotProductSize;
            float sum = 0;
            for (unsigned int entryIndex = 0;
                 entryIndex < dotProductSize; ++entryIndex) {
              sum +=
                dotProductData_LayoutRight_A[shortcutIndex + entryIndex] *
                dotProductData_LayoutRight_B[shortcutIndex + entryIndex];
            }
            dotProductResults[dotProductIndex] = sum;
          }

          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompTimesMatrix[dotProductSizeIndex][memorySizeIndex] += elapsedTime;

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
          ompTimesMatrix[dotProductSizeIndex][memorySizeIndex] = elapsedTime;
          // check the results
          checkAnswer(correctResults, dotProductResults,
                      dotProductSize, memorySize,
                      string("omp"));
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp> ******************************
      // ===============================================================

      // scrub the results
      std::fill(dotProductResults.begin(),
                dotProductResults.end(),
                std::numeric_limits<float>::quiet_NaN());
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
*/
#ifdef ENABLE_KOKKOS
      // ===============================================================
      // ***************** < do kokkos> ********************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      /*
      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float**, Kokkos::LayoutRight,
                             DeviceType>                   KokkosDotProductData;
        kokkosOmpTimesMatrix[dotProductSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosDotProductData>(numberOfDotProducts,
                                              numberOfRepeats,
                                              dotProductSize,
                                              memorySize,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &dotProductResults);
      } */
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float*****, Kokkos::LayoutLeft,
                             DeviceType>                   KokkosContractionData_Left;

        typedef Kokkos::View<float****, Kokkos::LayoutLeft,
                             DeviceType>                   KokkosContractionData_Right;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaIndependentTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosContractionData_Left,
                        KokkosContractionData_Right>(cellSize,
			                      numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              l,
                                              q,
                                              d1,
                                              d2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &contractionResults);
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
#endif // ENABLE_KOKKOS

      contractionSizeMatrix[contractionSizeIndex][memorySizeIndex] =
        contractionSize;
      numberOfContractionsMatrix[contractionSizeIndex][memorySizeIndex] =
        numCells;
      memorySizeMatrix[contractionSizeIndex][memorySizeIndex] =
        memorySize;

    }

    const timespec thisSizesToc = getTimePoint();
    const float thisSizesElapsedTime =
      getElapsedTime(thisSizesTic, thisSizesToc);
    printf("completed %4u repeats of dot products of size %4u "
           "in %7.2f seconds\n", numberOfRepeats,
           contractionSize, thisSizesElapsedTime);

    checkCudaError(cudaFree(dev_contractionData_LayoutLeft_A));
    checkCudaError(cudaFree(dev_contractionData_LayoutLeft_B));
    checkCudaError(cudaFree(dev_contractionData_LayoutRight_A));
    checkCudaError(cudaFree(dev_contractionData_LayoutRight_B));
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
#ifdef ENABLE_KOKKOS
  writeTimesMatrixToFile(kokkosOmpTimesMatrix,
                         prefix + string("kokkosOmpTimes") + suffix);
  writeTimesMatrixToFile(kokkosCudaIndependentTimesMatrix,
                         prefix + string("kokkosCudaIndependentTimes") + suffix);
#endif

#ifdef ENABLE_KOKKOS
  const unsigned int numberOfMethods = 7;
#else
  const unsigned int numberOfMethods = 5;
#endif

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

#ifdef ENABLE_KOKKOS
  Kokkos::finalize();
#endif

  return 0;
}
