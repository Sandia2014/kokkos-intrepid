// -*- C++ -*-
// ContractDataDataTensor.cu
// a huge comparison of different ways of doing ContractDataDataTensor
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

#define ENABLE_KOKKOS
#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#endif // ENABLE_KOKKOS

enum CudaStyle {CudaStyle_Independent,
                CudaStyle_Reduction};

enum ClearCacheStyle {ClearCacheAfterEveryRepeat,
                      DontClearCacheAfterEveryRepeat};

string
convertCudaStyleToString(const CudaStyle cudaStyle) {
  switch (cudaStyle) {
  case CudaStyle_Independent:
    return string("CudaStyle_Independent");
  case CudaStyle_Reduction:
    return string("CudaStyle_Reduction");
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
            const vector<float> & calcResults,
            const unsigned int contractionSize,
            const unsigned int memorySize,
            const string flavorName) {
  for (unsigned int dotProductIndex = 0;
       dotProductIndex < correctResults.size();
       ++dotProductIndex) {
    if (std::abs(correctResults[dotProductIndex] -
                 calcResults[dotProductIndex]) /
        std::abs(correctResults[dotProductIndex]) > 1e-4) {
      fprintf(stderr, "invalid answer for dot product index %u for "
              "flavor %s, "
              "should be %e but we have %e, "
              "contractionSize = %u, memorySize = %8.2e\n",
              dotProductIndex, flavorName.c_str(),
              correctResults[dotProductIndex],
              calcResults[dotProductIndex],
              contractionSize, float(memorySize));
      exit(1);
    }
  }
}


#ifdef ENABLE_KOKKOS

typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;
typedef Kokkos::DefaultExecutionSpace       Device ;
typedef Kokkos::HostSpace::execution_space  Host ;



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



template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensor_TeamFunctor {
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  ContractDataDataTensor_TeamFunctor( int numPoints,
      int dim1,
      int dim2,
      LeftViewType leftInput,
      RightViewType rightInput,
      OutputViewType output) :
    _leftInput(leftInput),
    _rightInput(rightInput),
    _output(output),
    _numPoints(numPoints),
    _dim1(dim1),
    _dim2(dim2)
  {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& thread) const {

    const unsigned int elementIndex = thread.league_rank();
    const unsigned int dim = thread.team_rank();

    float sum = 0;
    float tsum = 0;


    for (unsigned int qp=0; qp < _numPoints; ++qp) {
      sum +=  _leftInput(elementIndex, qp, dim/_dim2, dim%_dim2) *
        _rightInput(elementIndex, qp, dim/_dim2, dim%_dim2);
    }

    //sum = 0;

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2),
        [&] (const unsigned int& dim, float& localsum) {
        localsum += sum;
      }, tsum);


    // FIXME everyone is writing this?
    _output(elementIndex) = tsum;
  }

private:
  ContractDataDataTensor_TeamFunctor();
};


template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  ContractDataDataTensorFunctor( int numPoints,
      int dim1,
      int dim2,
      LeftViewType leftInput,
      RightViewType rightInput,
      OutputViewType output) :
    _leftInput(leftInput),
    _rightInput(rightInput),
    _output(output),
    _numPoints(numPoints),
    _dim1(dim1),
    _dim2(dim2)
  {
    // Nothing to do
  }

  // Parallelize over c-loop
  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {

    double tmp = 0;
    for (int qp=0; qp < _numPoints; qp++) {
      for (int iTens1=0; iTens1 < _dim1; iTens1++) {
        for (int iTens2=0; iTens2 < _dim2; iTens2++) {
          tmp += _leftInput(elementIndex, qp, iTens1, iTens2) *
                  _rightInput(elementIndex, qp, iTens1, iTens2);
        }
      }
    }
    _output(elementIndex) = tmp;
  }
private:
  ContractDataDataTensorFunctor();
};




template <class DeviceType, class KokkosInputData>
double
runKokkosTest(const unsigned int numberOfRepeats,
              const unsigned int memorySize,
              const unsigned int numCells,
              const unsigned int numPoints,
              const unsigned int dim1,
              const unsigned int dim2,
              const vector<float> & dotProductData_LayoutRight_A,
              const vector<float> & dotProductData_LayoutRight_B,
              const vector<float> & correctResults,
              const string & kokkosFlavor,
              const ClearCacheStyle clearCacheStyle,
              const vector<int> & junkDataToClearTheCache,
              size_t * junkDataCounter,
              unsigned int * const totalNumberOfRepeats,
              vector<float> * calcResults,
              bool doTeam) {

  const unsigned int junkDataSize = junkDataToClearTheCache.size();

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

  KokkosCalcResults dev_kokkosCalcResults("kokkos dot product results",
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
            dotProductData_LayoutRight_A[clDim + qpDim + iTens1Dim + iTens2];
          kokkosInputData_B(cl, qp, iTens1, iTens2) = 
            dotProductData_LayoutRight_B[clDim + qpDim + iTens1Dim + iTens2];
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
  if (!doTeam) {

    // breaking formatting convention because holy freak that's long
    ContractDataDataTensorFunctor<DeviceType,
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

  else {
    ContractDataDataTensor_TeamFunctor<DeviceType,
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


  // copy over the results from the device to the host
  Kokkos::deep_copy(kokkosCalcResults, dev_kokkosCalcResults);
  for (unsigned int dotProductIndex = 0;
       dotProductIndex < numCells; ++dotProductIndex) {
    calcResults->at(dotProductIndex) =
      kokkosCalcResults(dotProductIndex);
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
  const vector<unsigned int> contractionSizes =
    {{25, 100, 500, 1000, 2000}};
    //{{8, 16, 32, 64, 128, 256, 512, 1024, 2048}};
  const array<float, 2> memorySizeExtrema = {{1e6, 1e9}};
  const unsigned int numberOfMemorySizes = 20;
  const unsigned int dimSize = 5;


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
    numberOfDotProductsMatrix(numberOfContractionSizes,
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

#ifdef RAW_CUDA
  vector<vector<float> >
    cudaIndependent_TimesMatrix(numberOfContractionSizes,
                                vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaReduction_TimesMatrix(numberOfContractionSizes,
                              vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    cudaSwitchingTimesMatrix(numberOfContractionSizes,
                             vector<float>(numberOfMemorySizes, 0));
#endif

#ifdef ENABLE_KOKKOS
  vector<vector<float> >
    kokkosOmpTimesMatrix(numberOfContractionSizes,
                         vector<float>(numberOfMemorySizes, 0));
  vector<vector<float> >
    kokkosCudaIndependentTimesMatrix(numberOfContractionSizes,
                                     vector<float>(numberOfMemorySizes, 0));

  vector<vector<float> >
    kokkosCudaTeamTimesMatrix(numberOfContractionSizes,
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
  for (unsigned int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {
    const unsigned int contractionSize = contractionSizes[contractionSizeIndex];
    //const unsigned int dimVec = 8;
    const unsigned int numPoints = contractionSize / (dimSize * dimSize);
    //const unsigned int numPoints = contractionSize / dimVec;

    const timespec thisSizesTic = getTimePoint();

    // allocate and initialize the largest amount of memory we'll need, then on
    //  each size we'll just use subsets of this memory.
    const unsigned int maxNumberOfDotProducts =
      memorySizes.back() / 4 / sizeof(float) / contractionSize;
    vector<float> dotProductData_LayoutRight_A(maxNumberOfDotProducts * contractionSize);
    vector<float> dotProductData_LayoutRight_B(dotProductData_LayoutRight_A.size());
    vector<float> dotProductData_LayoutLeft_A(dotProductData_LayoutRight_A.size());
    vector<float> dotProductData_LayoutLeft_B(dotProductData_LayoutRight_B.size());
    for (unsigned int dotProductIndex = 0;
         dotProductIndex < maxNumberOfDotProducts; ++dotProductIndex) {
      for (unsigned int entryIndex = 0;
           entryIndex < contractionSize; ++entryIndex) {
        const unsigned int layoutRightIndex =
          dotProductIndex * contractionSize + entryIndex;
        dotProductData_LayoutRight_A[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        dotProductData_LayoutRight_B[layoutRightIndex] =
          randomNumberGenerator(randomNumberEngine);
        const unsigned int layoutLeftIndex =
          entryIndex * maxNumberOfDotProducts + dotProductIndex;
        dotProductData_LayoutLeft_A[layoutLeftIndex] =
          dotProductData_LayoutRight_A[layoutRightIndex];
        dotProductData_LayoutLeft_B[layoutLeftIndex] =
          dotProductData_LayoutRight_B[layoutRightIndex];
      }
    }
    vector<float> calcResults(maxNumberOfDotProducts,
                                    std::numeric_limits<float>::quiet_NaN());

#ifdef RAW_CUDA
    // now, because we'll be working with cuda stuff, also allocate the inputs
    //  and output on the gpu and copy them over
    float * dev_dotProductData_LayoutRight_A;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutRight_A,
                              maxNumberOfDotProducts * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutRight_A,
                              &dotProductData_LayoutRight_A[0],
                              maxNumberOfDotProducts * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_dotProductData_LayoutRight_B;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutRight_B,
                              maxNumberOfDotProducts * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutRight_B,
                              &dotProductData_LayoutRight_B[0],
                              maxNumberOfDotProducts * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_calcResults;
    checkCudaError(cudaMalloc((void **) &dev_calcResults,
                              maxNumberOfDotProducts * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_calcResults, &calcResults[0],
                              maxNumberOfDotProducts * sizeof(float),
                              cudaMemcpyHostToDevice));
    // make and populate the LayoutLeft versions
    float * dev_dotProductData_LayoutLeft_A;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutLeft_A,
                              maxNumberOfDotProducts * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutLeft_A,
                              &dotProductData_LayoutLeft_A[0],
                              maxNumberOfDotProducts * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    float * dev_dotProductData_LayoutLeft_B;
    checkCudaError(cudaMalloc((void **) &dev_dotProductData_LayoutLeft_B,
                              maxNumberOfDotProducts * contractionSize * sizeof(float)));
    checkCudaError(cudaMemcpy(dev_dotProductData_LayoutLeft_B,
                              &dotProductData_LayoutLeft_B[0],
                              maxNumberOfDotProducts * contractionSize * sizeof(float),
                              cudaMemcpyHostToDevice));
#endif // RAW_CUDA

    // for each memory size
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      const unsigned int memorySize = memorySizes[memorySizeIndex];


      const unsigned int numCells =
        memorySize / 4 / sizeof(float) / contractionSize;
      if (memorySize != 4 * sizeof(float) * numCells * contractionSize) {
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


  for (int cl=0; cl < numCells; cl++) {
    double tmp = 0;
    for (int qp=0; qp < numPoints; qp++) {
      for (int iTens1=0; iTens1 < dimSize; iTens1++) {
        for (int iTens2=0; iTens2 < dimSize; iTens2++) {
          tmp += dotProductData_LayoutRight_A[cl * numPoints * dimSize * dimSize + qp * dimSize * dimSize + iTens1 * dimSize + iTens2] *
                 dotProductData_LayoutRight_B[cl * numPoints * dimSize * dimSize + qp * dimSize * dimSize + iTens1 * dimSize + iTens2];
        }
      }
    }
    calcResults[cl] = tmp;
  }


#if 0

          // do the actual calculation
          for (int cl = 0; cl < numCells; cl++) {
            int clDim = cl * numPoints * dimVec;
            float tmpVal = 0;
            for (int qp = 0; qp < numPoints; qp++) {
              int qpDim = qp * dimVec;
              for (int iVec = 0; iVec < dimVec; iVec++) {
                tmpVal += 
                  dotProductData_LayoutRight_A[clDim + qpDim + iVec] *
                  dotProductData_LayoutRight_B[clDim + qpDim + iVec];
              } // D-loop
            } // P-loop
            calcResults[cl] = tmpVal;
          } // C-loop

#endif
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
#pragma omp parallel for default(none)                                  \
  shared(dotProductData_LayoutRight_A, dotProductData_LayoutRight_B,    \
         calcResults)
//          for (int cl = 0; cl < numCells; cl++) {
//            int clDim = cl * numPoints * dimVec;
//            float tmpVal = 0;
//            for (int qp = 0; qp < numPoints; qp++) {
//              int qpDim = qp * dimVec;
//              for (int iVec = 0; iVec < dimVec; iVec++) {
//                tmpVal += 
//                  dotProductData_LayoutRight_A[clDim + qpDim + iVec] *
//                  dotProductData_LayoutRight_B[clDim + qpDim + iVec];
//              } // D-loop
//            } // P-loop
//            calcResults[cl] = tmpVal;
//          } // C-loop


  for (int cl=0; cl < numCells; cl++) {
    double tmp = 0;
    for (int qp=0; qp < numPoints; qp++) {
      for (int iTens1=0; iTens1 < dimSize; iTens1++) {
        for (int iTens2=0; iTens2 < dimSize; iTens2++) {
          tmp += dotProductData_LayoutRight_A[cl * numPoints * dimSize * dimSize + qp * dimSize * dimSize + iTens1 * dimSize + iTens2] *
                 dotProductData_LayoutRight_B[cl * numPoints * dimSize * dimSize + qp * dimSize * dimSize + iTens1 * dimSize + iTens2];
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

#ifdef RAW_CUDA
      checkCudaError(cudaMemcpy(dev_calcResults, &calcResults[0],
                                maxNumberOfDotProducts * sizeof(float),
                                cudaMemcpyHostToDevice));

      // ===============================================================
      // ***************** < do cuda independent> **********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      {
        const unsigned int numberOfThreadsPerBlock = 1024;

        cudaIndependent_TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Independent,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfDotProducts,
                      maxNumberOfDotProducts,
                      contractionSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_dotProductData_LayoutLeft_A,
                      dev_dotProductData_LayoutLeft_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_calcResults,
                      &calcResults);

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
                   unsigned(ceil(contractionSize / 32.)) * 32);

        cudaReduction_TimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runCudaTest(CudaStyle_Reduction,
                      numberOfThreadsPerBlock,
                      numberOfRepeats,
                      maxNumberOfCudaBlocks,
                      numberOfDotProducts,
                      maxNumberOfDotProducts,
                      contractionSize,
                      memorySize,
                      correctResults,
                      clearCacheStyle,
                      dev_junkDataToClearTheCache,
                      junkDataSize,
                      dev_dotProductData_LayoutRight_A,
                      dev_dotProductData_LayoutRight_B,
                      dev_junkDataCounter,
                      &totalNumberOfRepeats,
                      dev_calcResults,
                      &calcResults);

      }
      cudaSwitchingTimesMatrix[contractionSizeIndex][memorySizeIndex] =
        runSwitchingCudaTest(numberOfRepeats,
                             maxNumberOfCudaBlocks,
                             numberOfDotProducts,
                             maxNumberOfDotProducts,
                             contractionSize,
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
                             dev_calcResults,
                             &calcResults);
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
                             DeviceType>                   KokkosInputData;
        kokkosOmpTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize,
                                              dimSize,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos openmp"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults, 
                                              false);
      }

      {
        typedef Kokkos::Cuda                               DeviceType;
        typedef Kokkos::View<float****, Kokkos::LayoutLeft,
                             DeviceType>                   KokkosInputData;
        // i pass in the layout right version even though this is the cuda
        //  version because it gets copied into the view inside the function.
        kokkosCudaTeamTimesMatrix[contractionSizeIndex][memorySizeIndex] =
          runKokkosTest<DeviceType,
                        KokkosInputData>(numberOfRepeats,
                                              memorySize,
                                              numCells,
                                              numPoints,
                                              dimSize,
                                              dimSize,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              true);
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
                                              dimSize,
                                              dimSize,
                                              dotProductData_LayoutRight_A,
                                              dotProductData_LayoutRight_B,
                                              correctResults,
                                              string("Kokkos cuda"),
                                              clearCacheStyle,
                                              junkDataToClearTheCache,
                                              &junkDataCounter,
                                              &totalNumberOfRepeats,
                                              &calcResults,
                                              false);
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ***************** </do kokkos> ********************************
      // ===============================================================
#endif // ENABLE_KOKKOS

      contractionSizeMatrix[contractionSizeIndex][memorySizeIndex] =
        contractionSize;
      numberOfDotProductsMatrix[contractionSizeIndex][memorySizeIndex] =
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

#ifdef RAW_CUDA
    checkCudaError(cudaFree(dev_dotProductData_LayoutLeft_A));
    checkCudaError(cudaFree(dev_dotProductData_LayoutLeft_B));
    checkCudaError(cudaFree(dev_dotProductData_LayoutRight_A));
    checkCudaError(cudaFree(dev_dotProductData_LayoutRight_B));
    checkCudaError(cudaFree(dev_calcResults));
#endif

  }
  writeTimesMatrixToFile(contractionSizeMatrix,
                         prefix + string("contractionSize") + suffix);
  writeTimesMatrixToFile(numberOfDotProductsMatrix,
                         prefix + string("numberOfDotProducts") + suffix);
  writeTimesMatrixToFile(memorySizeMatrix,
                         prefix + string("memorySize") + suffix);
  writeTimesMatrixToFile(serialTimesMatrix,
                         prefix + string("serialTimes") + suffix);
  writeTimesMatrixToFile(ompTimesMatrix,
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

  writeTimesMatrixToFile(kokkosCudaTeamTimesMatrix,
                         prefix + string("kokkosCudaTeamTimes") + suffix);
#endif

#if defined RAW_CUDA
  // Note, we assume that if RAW_CUDA is defined so is ENABLE_KOKKOS here
  const unsigned int numberOfMethods = 8;
#elif defined ENABLE_KOKKOS
  const unsigned int numberOfMethods = 5;
#else
  const unsigned int numberOfMethods = 2;
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
