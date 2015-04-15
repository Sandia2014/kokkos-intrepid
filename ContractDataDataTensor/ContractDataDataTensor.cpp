// -*- C++ -*-
// ContractDataDataTensor.cu
// a huge comparison of different ways of doing ContractDataDataTensor
// Ellen Hui (outline stolen from Jeff Amelang), 2015

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

#define ENABLE_KOKKOS

#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif // ENABLE_KOKKOS

#include "Utilities.hpp"
#include "ContractDataDataTensorFunctors.hpp"


enum KokkosStyle {KokkosStyle_Independent,
                  KokkosStyle_TeamStride,
                  KokkosStyle_Depth1Reduction,
                  KokkosStyle_Depth2Reduction,
                  KokkosStyle_Depth3Reduction,
                  KokkosStyle_OmpTeamsize2,
                  KokkosStyle_OmpTeamsize12};

enum ClearCacheStyle {ClearCacheAfterEveryRepeat,
                      DontClearCacheAfterEveryRepeat};

// Use cases, from Carter, Irina, and Denis
//
// DataData use case dd1:  t1 = t2 = 3, p = 8
// DataData use case dd2:  t1 = t2 = 3, p = 27
// DataData use case dd3:  t1 = t2 = 6, p = 27

struct UseCase {
  const unsigned int _dim1;
  const unsigned int _dim2;
  const unsigned int _numPoints;

  UseCase(const unsigned int dim1, const unsigned int dim2, const unsigned int numPoints)
    : _dim1(dim1), _dim2(dim2), _numPoints(numPoints)
  {
    //nothing
  }

  int getSize() {
    return _dim1 * _dim2 * _numPoints;
  }
};



#ifdef ENABLE_KOKKOS


template <class DeviceType, class KokkosInputData>
double
runKokkosTest(const unsigned int numberOfRepeats,
              const unsigned int memorySize,
              const unsigned int numCells,
              const unsigned int numPoints,
              const unsigned int dim1,
              const unsigned int dim2,
              const vector<float> & contractionData_LayoutRight_A,
              const vector<float> & contractionData_LayoutRight_B,
              const vector<float> & correctResults,
              const string & kokkosFlavor,
              const ClearCacheStyle clearCacheStyle,
              const vector<int> & junkDataToClearTheCache,
              size_t * junkDataCounter,
              unsigned int * const totalNumberOfRepeats,
              vector<float> * calcResults,
              KokkosStyle kokkosStyle) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

  typedef Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename KokkosInputData::HostMirror     KokkosInputData_Host;
  typedef Kokkos::View<float*, DeviceType>              KokkosCalcResults;
  typedef typename KokkosCalcResults::HostMirror  KokkosCalcResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  KokkosInputData dev_kokkosInputData_A("kokkos data A",
                                                  numCells, numPoints, dim1, dim2);
  KokkosInputData_Host kokkosInputData_A =
    Kokkos::create_mirror_view(dev_kokkosInputData_A);

  KokkosInputData dev_kokkosInputData_B("kokkos data B",
                                                  numCells, numPoints, dim1, dim2);
  KokkosInputData_Host kokkosInputData_B =
    Kokkos::create_mirror_view(dev_kokkosInputData_B);

  KokkosCalcResults dev_kokkosCalcResults("kokkos contraction results",
                                                      numCells);
  KokkosCalcResults_Host kokkosCalcResults =
    Kokkos::create_mirror_view(dev_kokkosCalcResults);

  KokkosJunkVector dev_kokkosJunkDataToClearTheCache("kokkos junk data to clear cache",
                                                     junkDataSize);
  KokkosJunkVector_Host kokkosJunkDataToClearTheCache =
    Kokkos::create_mirror_view(dev_kokkosJunkDataToClearTheCache);

  // copy the data into the device views and ship them over
  for (int cl = 0; cl < numCells; ++cl) {
    int clDim = cl * numPoints * dim1 * dim2;
    for (int qp = 0; qp < numPoints; ++qp) {
      int qpDim = qp * dim1 * dim2;
      for (int iTens1 = 0; iTens1 < dim1; ++iTens1) {
        int iTens1Dim = iTens1 * dim2;
        for (int iTens2 = 0; iTens2 < dim2; ++iTens2) {
          kokkosInputData_A(cl, qp, iTens1, iTens2) =
            contractionData_LayoutRight_A[clDim + qpDim + iTens1Dim + iTens2];
          kokkosInputData_B(cl, qp, iTens1, iTens2) =
            contractionData_LayoutRight_B[clDim + qpDim + iTens1Dim + iTens2];
        }
      }
    }
  }

  Kokkos::deep_copy(dev_kokkosInputData_A, kokkosInputData_A);
  Kokkos::deep_copy(dev_kokkosInputData_B, kokkosInputData_B);

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


  double totalElapsedTime;
  if (kokkosStyle == KokkosStyle_Independent) {

    // breaking formatting convention because holy freak that's long
    ContractDataDataTensorIndependentFunctor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }

      // actually do the calculation
      Kokkos::parallel_for(numCells, contractDataDataTensorFunctor);

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

  else if (kokkosStyle == KokkosStyle_TeamStride) {

    ContractDataDataTensorTeamStrideFunctor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    const team_policy reduction_policy(numCells, 32);
    //const team_policy reduction_policy(numCells,
    //    team_policy::team_size_max(ContractDataDataTensorTeamStrideFunctor
    //      <DeviceType, KokkosInputData, KokkosInputData, KokkosCalcResults>));

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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

  else if (kokkosStyle == KokkosStyle_OmpTeamsize2) {

    ContractDataDataTensorTeamStrideFunctor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);


    const team_policy reduction_policy(numCells, 2);
    //const team_policy reduction_policy(numCells,
    //    team_policy::team_size_max(ContractDataDataTensorTeamStrideFunctor
    //      <DeviceType, KokkosInputData, KokkosInputData, KokkosCalcResults>));

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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

  else if (kokkosStyle == KokkosStyle_OmpTeamsize12) {

    ContractDataDataTensorTeamStrideFunctor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    const team_policy reduction_policy(numCells, 12);
    //const team_policy reduction_policy(numCells,
    //    team_policy::team_size_max(ContractDataDataTensorTeamStrideFunctor
    //      <DeviceType, KokkosInputData, KokkosInputData, KokkosCalcResults>));

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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




  else if (kokkosStyle == KokkosStyle_Depth1Reduction) {
    ContractDataDataTensor_TeamDepth1Functor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    //const team_policy reduction_policy(numCells, dim1 * dim2);
    const team_policy reduction_policy(numCells, dim2);

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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

  else if (kokkosStyle == KokkosStyle_Depth2Reduction) {
    ContractDataDataTensor_TeamDepth2Functor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    const team_policy reduction_policy(numCells, dim1 * dim2);

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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


  else if (kokkosStyle == KokkosStyle_Depth3Reduction) {
    ContractDataDataTensor_TeamDepth3Functor<DeviceType,
      KokkosInputData,
      KokkosInputData,
      KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
            dim1,
            dim2,
            dev_kokkosInputData_A,
            dev_kokkosInputData_B,
            dev_kokkosCalcResults);

    const team_policy reduction_policy(numCells, dim1 * dim2 * numPoints);

    timespec tic;
    totalElapsedTime = 0;
    for (unsigned int repeatIndex = 0;
        repeatIndex < numberOfRepeats + 1; ++repeatIndex) {
      *totalNumberOfRepeats = *totalNumberOfRepeats + 1;
      if ((clearCacheStyle == DontClearCacheAfterEveryRepeat &&
            repeatIndex == 1) ||
          clearCacheStyle == ClearCacheAfterEveryRepeat) {
        tic = getTimePoint();
      }


      Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
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

  else {
    printf("Illegal kokkosStyle!");
    exit(1);
  }



  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosCalcResults, dev_kokkosCalcResults);
  for (unsigned int contractionIndex = 0;
       contractionIndex < numCells; ++contractionIndex) {
    calcResults->at(contractionIndex) =
      kokkosCalcResults(contractionIndex);
  }
  // check the results
  checkAnswer(correctResults, *calcResults,
              numPoints * dim1 * dim2, memorySize,
              kokkosFlavor);

  // scrub the results
  std::fill(calcResults->begin(),
            calcResults->end(),
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
  //

  vector<UseCase> contractionSizes;
  contractionSizes.push_back(UseCase(3, 3, 8));
  contractionSizes.push_back(UseCase(3, 3, 27));
  contractionSizes.push_back(UseCase(6, 6, 27));

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
  const string prefix = "data/ContractDataDataTensor_";
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
    // times 2 because otherwise we have division issues
    const unsigned int maxContractionSize = contractionSizes.back().getSize() * 2;
    // memory size is linear on a log scale, but rounded to a multiple of the
    // largest contraction size
    const unsigned int desiredMemorySizeInBytes = pow(10., thisLog);
    // now, in this amount of memory i have to fit two vectors of data
    // that are multiples of the max contraction size
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
    numberOfcontractionsMatrix(numberOfContractionSizes,
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

#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosOmpTeamSize2TimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosOmpTeamSize12TimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaTeamStrideTimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaTeamDepth1TimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaTeamDepth2TimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaTeamDepth3TimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));
#endif

  // create some junk data to use in clearing the cache
  size_t junkDataCounter = 0;
  const size_t junkDataSize = 1e7;
  vector<int> junkDataToClearTheCache(junkDataSize, 0);
  for (unsigned int i = 0; i < junkDataSize/100; ++i) {
    junkDataToClearTheCache[(rand() / float(RAND_MAX))*junkDataSize] = 1;
  }

  unsigned int totalNumberOfRepeats = 0;

  // for each contraction size
  for (unsigned int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {

    UseCase useCase = contractionSizes[contractionSizeIndex];
    const unsigned int contractionSize = useCase.getSize();
    const unsigned int numPoints =  useCase._numPoints;
    const unsigned int dimSize1 =  useCase._dim1;
    const unsigned int dimSize2  = useCase._dim2;

    const timespec thisSizesTic = getTimePoint();

    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumberOfcontractions =
      memorySizes.back() / 4 / sizeof(float) / contractionSize;
    vector<float> contractionData_LayoutRight_A(maxNumberOfcontractions * contractionSize);
    vector<float> contractionData_LayoutRight_B(contractionData_LayoutRight_A.size());
    vector<float> contractionData_LayoutLeft_A(contractionData_LayoutRight_A.size());
    vector<float> contractionData_LayoutLeft_B(contractionData_LayoutRight_B.size());

    // make some input data
    for (unsigned int contractionIndex = 0;
         contractionIndex < maxNumberOfcontractions; ++contractionIndex) {
      for (unsigned int entryIndex = 0;
           entryIndex < contractionSize; ++entryIndex) {
        const unsigned int layoutRightIndex =
          contractionIndex * contractionSize + entryIndex;
        contractionData_LayoutRight_A[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        contractionData_LayoutRight_B[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        const unsigned int layoutLeftIndex =
          entryIndex * maxNumberOfcontractions + contractionIndex;
        contractionData_LayoutLeft_A[layoutLeftIndex] =
          contractionData_LayoutRight_A[layoutRightIndex];
        contractionData_LayoutLeft_B[layoutLeftIndex] =
          contractionData_LayoutRight_B[layoutRightIndex];
      }
    }

    // make result matrix
    vector<float> calcResults(maxNumberOfcontractions,
                                    std::numeric_limits<float>::quiet_NaN());

    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex];


      const unsigned int numCells =
        memorySize / 4 / sizeof(float) / contractionSize;
      if (memorySize != 4 * sizeof(float) * numCells * contractionSize) {
        fprintf(stderr, "invalid memory size of %u for contraction size of "
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


  for (int cl=0; cl < numCells; cl++) {
    int clDim = cl * numPoints * dimSize1 * dimSize2;
    double tmp = 0;
    for (int qp=0; qp < numPoints; qp++) {
      int qpDim = qp * dimSize1 * dimSize2;
      for (int iTens1=0; iTens1 < dimSize1; iTens1++) {
        int iTens1Dim = iTens1 * dimSize2;
        for (int iTens2=0; iTens2 < dimSize2; iTens2++) {
          tmp += contractionData_LayoutRight_A[clDim + qpDim + iTens1Dim + iTens2] *
                 contractionData_LayoutRight_B[clDim + qpDim + iTens1Dim + iTens2];
        }
      }
    }
    calcResults[cl] = tmp;
  }


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

      const vector<float> correctResults = calcResults;
      // scrub the results
      std::fill(calcResults.begin(),
                calcResults.end(),
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
#pragma omp parallel                                   \
  shared(contractionData_LayoutRight_A, contractionData_LayoutRight_B,    \
         calcResults)

  for (int cl=0; cl < numCells; cl++) {
    double tmp = 0;
    for (int qp=0; qp < numPoints; qp++) {
      for (int iTens1=0; iTens1 < dimSize1; iTens1++) {
        for (int iTens2=0; iTens2 < dimSize2; iTens2++) {
          tmp += contractionData_LayoutRight_A[cl * numPoints * dimSize1 * dimSize2 + qp * dimSize1 * dimSize1 + iTens1 * dimSize2 + iTens2] *
                 contractionData_LayoutRight_B[cl * numPoints * dimSize1 * dimSize2 + qp * dimSize1 * dimSize2 + iTens1 * dimSize2 + iTens2];
        }
      }
    }
    calcResults[cl] = tmp;
  }




          if (clearCacheStyle == ClearCacheAfterEveryRepeat) {
            const timespec toc = getTimePoint();
            const float elapsedTime = getElapsedTime(tic, toc);
            ompTimesMatrix[contractionSizeIndex][memorySizeIndex] += elapsedTime;

            // attempt to scrub all levels of cache
#pragma omp parallel                       \
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
          checkAnswer(correctResults, calcResults,
                      contractionSize, memorySize,
                      string("omp"));
        }
      }
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </do omp> ******************************
      // ===============================================================

      // scrub the results
      std::fill(calcResults.begin(),
                calcResults.end(),
                std::numeric_limits<float>::quiet_NaN());

#ifdef ENABLE_KOKKOS
      // ===============================================================
      // ***************** < do kokkos> ********************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#if 0
      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        kokkosOmpTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults, 
                                              KokkosStyle_Independent);
      }


      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        kokkosOmpTeamSize2TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults, 
                                              KokkosStyle_OmpTeamsize2);
      }

      {
        typedef Kokkos::OpenMP                             DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        kokkosOmpTeamSize12TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults, 
                                              KokkosStyle_OmpTeamsize12);
      }
#endif

      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTeamDepth1TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda depth 1 reduction"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              KokkosStyle_Depth1Reduction);
      }

      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTeamDepth2TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda depth 2 reduction"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              KokkosStyle_Depth2Reduction);
      }

      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTeamDepth3TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda depth 3 reduction"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              KokkosStyle_Depth3Reduction);
      }

      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutLeft,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaIndependentTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda independent"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              KokkosStyle_Independent);
      }
      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutRight,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTeamStrideTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize1,
                                              dimSize2,
                                              contractionData_LayoutRight_A,
                                              contractionData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda team stride"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              KokkosStyle_TeamStride);
      }



      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
#endif // ENABLE_KOKKOS

      contractionSizeMatrix[contractionSizeIndex][memorySizeIndex] =
        contractionSize;
      numberOfcontractionsMatrix[contractionSizeIndex][memorySizeIndex] =
        numCells;
      memorySizeMatrix[contractionSizeIndex][memorySizeIndex] =
        memorySize;

    }

    const timespec thisSizesToc = getTimePoint();
    const float thisSizesElapsedTime =
      getElapsedTime(thisSizesTic, thisSizesToc);
    printf("completed %4u repeats of contractions of size %4u "
           "in %7.2f seconds\n", numberOfRepeats,
           contractionSize, thisSizesElapsedTime);


  }
  writeTimesMatrixToFile(contractionSizeMatrix,
                         prefix + string("contractionSize") + suffix);
  writeTimesMatrixToFile(numberOfcontractionsMatrix,
                         prefix + string("numberOfcontractions") + suffix);
  writeTimesMatrixToFile(memorySizeMatrix,
                         prefix + string("memorySize") + suffix);
  writeTimesMatrixToFile(serialTimesMatrix,
                         prefix + string("serialTimes") + suffix);
  writeTimesMatrixToFile(ompTimesMatrix,
                         prefix + string("ompTimes") + suffix);


#ifdef ENABLE_KOKKOS
  writeTimesMatrixToFile(kokkosOmpTimesMatrix,
                         prefix + string("kokkosOmpTimes") + suffix);

  writeTimesMatrixToFile(kokkosOmpTeamSize2TimesMatrix,
                         prefix + string("kokkosOmpTeamSize2Times") + suffix);

  writeTimesMatrixToFile(kokkosOmpTeamSize12TimesMatrix,
                         prefix + string("kokkosOmpTeamSize12Times") + suffix);

  writeTimesMatrixToFile(kokkosCudaIndependentTimesMatrix,
                         prefix + string("kokkosCudaIndependentTimes") + suffix);

  writeTimesMatrixToFile(kokkosCudaTeamStrideTimesMatrix,
                         prefix + string("kokkosCudaTeamStrideTimes") + suffix);


  writeTimesMatrixToFile(kokkosCudaTeamDepth1TimesMatrix,
                         prefix + string("kokkosCudaTeamDepth1Times") + suffix);

  writeTimesMatrixToFile(kokkosCudaTeamDepth2TimesMatrix,
                         prefix + string("kokkosCudaTeamDepth2Times") + suffix);

  writeTimesMatrixToFile(kokkosCudaTeamDepth3TimesMatrix,
                         prefix + string("kokkosCudaTeamDepth3Times") + suffix);
#endif

#ifdef ENABLE_KOKKOS
  const unsigned int numberOfMethods = 7;
#else
  const unsigned int numberOfMethods = 2;
#endif

  const size_t junkDataSum =
    std::accumulate(junkDataToClearTheCache.begin(),
                    junkDataToClearTheCache.end(), size_t(0));
  {
    int temp = 0;

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
