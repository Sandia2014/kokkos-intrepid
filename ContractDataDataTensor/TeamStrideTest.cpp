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

#include <omp.h>
#include <Kokkos_Core.hpp>

// kokkos setup
typedef Kokkos::Cuda DeviceType;
typedef Kokkos::DefaultExecutionSpace       Device ;
typedef Kokkos::HostSpace::execution_space  Host ;
typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
typedef typename team_policy::member_type team_member;

// Teamstride functor
template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorTeamStrideFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;
  ContractDataDataTensorTeamStrideFunctor( int numPoints,
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

    const unsigned int cell = thread.league_rank();
    const unsigned int tIndex = thread.team_rank();

    const unsigned int dim12 = _dim1 * _dim2;
    const unsigned int cellSize = _numPoints * dim12;

    float sum = 0;
    float tsum = 0;

    for (unsigned int innerIdx = tIndex; innerIdx < cellSize; innerIdx += thread.team_size() ) {
      // Sane arithmetic version:
      // const unsigned int qp = innerIdx / (_dim1 * _dim2);
      // const unsigned int iTens1 = (innerIdx % (_dim1 * _dim2)) / _dim2;
      // const unsigned int iTens2 = innerIdx % _dim2;

      // Optimized arithmetic version:
      const unsigned int qp = innerIdx / dim12;
      const unsigned int idxDivDim2 = innerIdx / _dim2;
      const unsigned int iTens1 = idxDivDim2 - _dim1 * qp;
      const unsigned int iTens2 = innerIdx - idxDivDim2 *_dim2;

      sum +=  _leftInput(cell, qp, iTens1, iTens2) *
       _rightInput(cell, qp, iTens1, iTens2);
    }

    thread.team_barrier();

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, thread.team_size()),
        [&] (const unsigned int& dummy, float& localsum) {
        localsum += sum;
      }, tsum);

    thread.team_barrier();
    _output(cell) = tsum;
  }
private:
  ContractDataDataTensorTeamStrideFunctor();
};



int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // dimension size parameters
  const unsigned int numCells = 1000;
  const unsigned int numPoints = 100;
  const unsigned int dim1 = 10;
  const unsigned int dim2 = 10;

  // make input
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  vector<float> leftInput(numCells * numPoints * dim1 * dim2);
  vector<float> rightInput(numCells * numPoints * dim1 * dim2);
  for (unsigned int idx = 0; idx < leftInput.size(); ++idx) {
    leftInput[idx] = randomNumberGenerator(randomNumberEngine);
    rightInput[idx] = randomNumberGenerator(randomNumberEngine);
  }

  // output vectors
  vector<float> serialResults(numCells, 0);
  vector<float> kokkosResults(numCells, 0);

  // Do serial
  for (int cl=0; cl < numCells; cl++) {
    int clDim = cl * numPoints * dim1 * dim2;
    double tmp = 0;
    for (int qp=0; qp < numPoints; qp++) {
      int qpDim = qp * dim1 * dim2;
      for (int iTens1=0; iTens1 < dim1; iTens1++) {
        int iTens1Dim = iTens1 * dim2;
        for (int iTens2=0; iTens2 < dim2; iTens2++) {
          tmp += leftInput[clDim + qpDim + iTens1Dim + iTens2] *
                 rightInput[clDim + qpDim + iTens1Dim + iTens2];
        }
      }
    }
    serialResults[cl] = tmp;
  }

  // more Kokkos setup
  typedef Kokkos::View<float****, Kokkos::LayoutRight, DeviceType> KokkosInputData;

  typedef typename KokkosInputData::HostMirror     KokkosInputData_Host;
  typedef Kokkos::View<float*, DeviceType>              KokkosCalcResults;
  typedef typename KokkosCalcResults::HostMirror  KokkosCalcResults_Host;
  typedef Kokkos::View<int*, DeviceType>                KokkosJunkVector;
  typedef typename KokkosJunkVector::HostMirror         KokkosJunkVector_Host;

  // make host and device views, populate and ship over
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

  for (unsigned int c =0; c < numCells; ++c) {
    for (unsigned int p =0; p < numPoints; ++p) {
      for (unsigned int t1 =0; t1 < dim1; ++t1) {
        for (unsigned int t2 =0; t2 < dim2; ++t2) {
          const unsigned int idx = c * numPoints * dim1 * dim2 + p * dim1 * dim2 + t1 * dim2 + dim2;
          kokkosInputData_A(c, p, t1, t2) = leftInput[idx];
          kokkosInputData_B(c, p, t1, t2) = rightInput[idx];
        }
      }
    }
  }

  Kokkos::deep_copy(dev_kokkosInputData_A, kokkosInputData_A);
  Kokkos::deep_copy(dev_kokkosInputData_B, kokkosInputData_B);

  // teamstride functor
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

  // launch kernel and copy results back over
  const team_policy reduction_policy(numCells, 32);
  Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );
  Kokkos::fence();

  Kokkos::deep_copy(kokkosCalcResults, dev_kokkosCalcResults);
  for (unsigned int c =0; c < numCells; ++c) {
    kokkosResults[c] = kokkosCalcResults(c);
  }

  for (unsigned int c=0; c < numCells; ++c) {
    if (std::abs(serialResults[c] - kokkosResults[c]) /
        std::abs(serialResults[c]) > 1e-4) {
      fprintf(stderr, "invalid answer for dot product index %u, "
              "should be %e but we have %e\n ",
              c,
              serialResults[c],
              kokkosResults[c]);
      exit(1);
    }
  }

  Kokkos::finalize();
}
