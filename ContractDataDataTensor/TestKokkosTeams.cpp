#include <random>
#include <vector>
#include <cstdio>

#include <omp.h>

#include <Kokkos_Core.hpp>


// Setup
typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;
typedef Kokkos::DefaultExecutionSpace       Device ;
typedef Kokkos::HostSpace::execution_space  Host ;


// Kokkos TeamStride functor definition
template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorTeamStrideFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

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

    const unsigned int cellSize = _numPoints * _dim1 * _dim2;

    float sum = 0;
    float tsum = 0;

    for (unsigned int innerIdx = tIndex; innerIdx < cellSize; innerIdx += thread.team_size() ) {
        const unsigned int qp = innerIdx / (_dim1 * _dim2);
        const unsigned int iTens1 = (innerIdx % (_dim1 * _dim2)) / _dim2;
        const unsigned int iTens2 = innerIdx % _dim2;

        sum +=  _leftInput(cell, qp, iTens1, iTens2) *
        _rightInput(cell, qp, iTens1, iTens2);
    }

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, thread.team_size()),
        [&] (const unsigned int& dummy, float& localsum) {
        localsum += sum;
      }, tsum);

    _output(cell) = tsum;
  }
private:
  ContractDataDataTensorTeamStrideFunctor();
};


int main() {
  Kokkos::initialize();

  // Doesn't work!
  typedef Kokkos::OpenMP                                            DeviceType;

  // Works!
  // typedef Kokkos::Cuda                                              DeviceType;

  typedef Kokkos::View<float****, Kokkos::LayoutRight, DeviceType>  KokkosInputData;
  typedef typename KokkosInputData::HostMirror                      KokkosInputData_Host;

  typedef Kokkos::View<float*, DeviceType>                          KokkosCalcResults;
  typedef typename KokkosCalcResults::HostMirror                    KokkosCalcResults_Host;


  const unsigned int numCells   = 100;
  const unsigned int numPoints  = 100;
  const unsigned int dim1       = 10;
  const unsigned int dim2       = 10;

  // Kokkos Views
  KokkosInputData dev_kokkosInputData_A("kokkos data A", numCells, numPoints, dim1, dim2);
  KokkosInputData_Host kokkosInputData_A = Kokkos::create_mirror_view(dev_kokkosInputData_A);

  KokkosInputData dev_kokkosInputData_B("kokkos data B", numCells, numPoints, dim1, dim2);
  KokkosInputData_Host kokkosInputData_B = Kokkos::create_mirror_view(dev_kokkosInputData_B);

  KokkosCalcResults dev_kokkosCalcResults("kokkos dot product results", numCells);
  KokkosCalcResults_Host kokkosCalcResults = Kokkos::create_mirror_view(dev_kokkosCalcResults);

  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  // Put data into views
  for (int cl = 0; cl < numCells; ++cl) {
    for (int qp = 0; qp < numPoints; ++qp) {
      for (int iTens1 = 0; iTens1 < dim1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < dim2; ++iTens2) {
          kokkosInputData_A(cl, qp, iTens1, iTens2) = randomNumberGenerator(randomNumberEngine);
          kokkosInputData_B(cl, qp, iTens1, iTens2) = randomNumberGenerator(randomNumberEngine);
        }
      }
    }
  }

  // Ship to device
  Kokkos::deep_copy(dev_kokkosInputData_A, kokkosInputData_A);
  Kokkos::deep_copy(dev_kokkosInputData_B, kokkosInputData_B);

  // Making a Teamstride functor
  ContractDataDataTensorTeamStrideFunctor<DeviceType, KokkosInputData, KokkosInputData, KokkosCalcResults>
        contractDataDataTensorFunctor(numPoints,
                                      dim1,
                                      dim2,
                                      dev_kokkosInputData_A,
                                      dev_kokkosInputData_B,
                                      dev_kokkosCalcResults);


  // Set Policy
  const team_policy reduction_policy(numCells, 12);

  // Do the calculation
  Kokkos::parallel_for( reduction_policy, contractDataDataTensorFunctor );

  // Fence and finalize Kokkos
  Kokkos::fence();
  Kokkos::finalize();

  printf("done!\n");

}
