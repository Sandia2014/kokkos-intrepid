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

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>

void
checkAnswer(const vector<float> & correctResults,
            const vector<float> & calcResults,
            const unsigned int contractionSize,
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
              "contractionSize = %u\n",
              dotProductIndex, flavorName.c_str(),
              correctResults[dotProductIndex],
              calcResults[dotProductIndex],
              contractionSize);
      exit(1);
    }
  }
}

// Team stuff

typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;
typedef Kokkos::DefaultExecutionSpace       Device ;
typedef Kokkos::HostSpace::execution_space  Host ;

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorFunctor {
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

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& thread) const {

    unsigned int elementIndex = thread.league_rank();

    float sum;
    float tsum =0;
    for (unsigned int qp=0; qp < _numPoints; ++qp) {

      sum = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2),
          [&] (const unsigned int& dim, float& sum) {
              sum +=  _leftInput(elementIndex, qp, dim/_dim2, dim%_dim2) *
                      _rightInput(elementIndex, qp, dim/_dim2, dim%_dim2);
        }, sum);

      thread.team_barrier();

      tsum += sum;

    }

    _output(elementIndex) = tsum;
  }

private:
  ContractDataDataTensorFunctor();
};


int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);

  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  const unsigned int numCells = 1e3;
  const unsigned int numPoints = 100;
  const unsigned int dim1 = 10;
  const unsigned int dim2 = 10;

  typedef Kokkos::Cuda                             DeviceType;
  typedef Kokkos::View<float****, DeviceType>           KokkosInputData;
  typedef typename KokkosInputData::HostMirror          KokkosInputData_Host;
  typedef Kokkos::View<float*, DeviceType>              KokkosCalcResults;
  typedef typename KokkosCalcResults::HostMirror        KokkosCalcResults_Host;

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

  vector<float> inputA(numCells * numPoints * dim1 * dim2);
  vector<float> inputB(numCells * numPoints * dim1 * dim2);
  vector<float> serialResults(numCells);
  vector<float> kokkosResults(numCells);

  for (int cl = 0; cl < numCells; ++cl) {
    int clDim = cl * numPoints * dim1 * dim2;
    for (int qp = 0; qp < numPoints; ++qp) {
      int qpDim = qp * dim1 * dim2;
      for (int iTens1 = 0; iTens1 < dim1; ++iTens1) {
        int iTens1Dim = iTens1 * dim2;
        for (int iTens2 = 0; iTens2 < dim2; ++iTens2) {
          const float randA = randomNumberGenerator(randomNumberEngine);
          const float randB = randomNumberGenerator(randomNumberEngine);
          kokkosInputData_A(cl, qp, iTens1, iTens2) = randA;
          inputA[clDim + qpDim + iTens1Dim + iTens2] = randA;
          kokkosInputData_B(cl, qp, iTens1, iTens2) = randB;
          inputB[clDim + qpDim + iTens1Dim + iTens2] = randB;
        }
      }
    }
  }

  Kokkos::deep_copy(dev_kokkosInputData_A, kokkosInputData_A);
  Kokkos::deep_copy(dev_kokkosInputData_B, kokkosInputData_B);


  for (int cl=0; cl < numCells; cl++) {
    double tmp = 0;
    unsigned int clDim = cl * numPoints * dim1 * dim2;

    for (int qp=0; qp < numPoints; qp++) {
      unsigned int qpDim = qp * dim1 * dim2;

      for (int iTens1=0; iTens1 < dim1; iTens1++) {
        unsigned int iTens1Dim = iTens1 * dim2;

        for (int iTens2=0; iTens2 < dim2; iTens2++) {
          tmp +=  inputA[clDim + qpDim + iTens1Dim + iTens2] *
                  inputB[clDim + qpDim + iTens1Dim + iTens2];
        }
      }
    }
    serialResults[cl] = tmp;
  }


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




  const team_policy reduction_policy(numCells, dim1 * dim2);
  //const team_policy reduction_policy(numCells, numPoints);

  Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
  Kokkos::fence();

  Kokkos::deep_copy(kokkosCalcResults, dev_kokkosCalcResults);
  for (unsigned int cl = 0; cl < numCells; ++cl) {
    kokkosResults[cl] = kokkosCalcResults(cl);
  }
  // check the results
  checkAnswer(serialResults, kokkosResults,
              numPoints * dim1 * dim2,
              "kokkos cuda reduction");

  printf("yay!");
  Kokkos::finalize();

  return 0;
}
