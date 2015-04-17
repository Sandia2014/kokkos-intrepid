// ContractDataDataTensorFunctor.hpp
//
// Various flavors of contractdatadatatensor functors
// Ellen Hui, 2015


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

typedef Kokkos::DefaultExecutionSpace       Device ;
typedef Kokkos::HostSpace::execution_space  Host ;

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensor_TeamDepth2Functor {
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;

  ContractDataDataTensor_TeamDepth2Functor( int numPoints,
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

    // A team does one cell
    const unsigned int cellIndex = thread.league_rank();

    float sum = 0;
    // Each of the _dim1 * _dim2 threads qp dimension
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2),
        [&] (const unsigned int threadIndex, float& localsum) {
          const unsigned int dim1 = threadIndex / _dim2;
          const unsigned int dim2 = threadIndex % _dim2;

          for (unsigned int qp = 0; qp < _numPoints; ++qp) {
            localsum +=  _leftInput(cellIndex, qp, dim1, dim2) *
              _rightInput(cellIndex, qp, dim1, dim2);
          }

      }, sum);

    if (thread.team_rank() == 0) {
      _output(cellIndex) = sum;
    }
  }

private:
  ContractDataDataTensor_TeamDepth2Functor();
};



template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensor_TeamDepth1Functor {
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;
  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;

  ContractDataDataTensor_TeamDepth1Functor( int numPoints,
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

    // A team does one cell
    const unsigned int cellIndex = thread.league_rank();

    float sum = 0;
    // Each of the _dim1 threads contracts the qp and d1 dimensions
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim2),
        [&] (const unsigned int dim2, float& localsum) {
          for (unsigned int qp=0; qp < _numPoints; ++qp) {
            for (unsigned int d1=0; d1 < _dim1; ++d1) {
              localsum +=  _leftInput(cellIndex, qp, d1, dim2) *
                _rightInput(cellIndex, qp, d1, dim2);
            }
          }
      }, sum);

    if (thread.team_rank() == 0) {
      _output(cellIndex) = sum;
    }
  }

private:
  ContractDataDataTensor_TeamDepth1Functor();
};



template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorIndependentFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;
  ContractDataDataTensorIndependentFunctor( int numPoints,
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

    // Each thread does one cell, contracting the qp, d1, and d2 dimensions
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
  ContractDataDataTensorIndependentFunctor();
};



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
  ContractDataDataTensorTeamStrideFunctor(int numPoints,
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

    // A team does one cell
    const unsigned int cellIndex = thread.league_rank();

    // Some useful derived constants that we'll reuse
    const unsigned int dim12 = _dim1 * _dim2;
    const unsigned int cellSize = _numPoints * dim12;

    float sum = 0;

    Kokkos::parallel_reduce
      (Kokkos::TeamThreadLoop(thread,cellSize),
       [&](const unsigned int indexWithinContraction, float & localsum) {

        // Calculate the next element to add (striding by teamsize)
        const unsigned int qp = indexWithinContraction / dim12;
        const unsigned int indexWithinTens1Tens2Thing =
          indexWithinContraction - qp * dim12;
        const unsigned int iTens1 = indexWithinTens1Tens2Thing / _dim2;
        const unsigned int iTens2 = indexWithinTens1Tens2Thing - iTens1*_dim2;

        localsum +=  _leftInput(cellIndex, qp, iTens1, iTens2) *
          _rightInput(cellIndex, qp, iTens1, iTens2);
      } , sum );

    if (thread.team_rank() == 0) {
      _output(cellIndex) = sum;
    }
  }

private:
  ContractDataDataTensorTeamStrideFunctor();
};


template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensor_TeamDepth3Functor {
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;
  ContractDataDataTensor_TeamDepth3Functor( int numPoints,
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

    // A team does one cell
    const unsigned int cellIndex = thread.league_rank();

    float sum = 0;

    // Each of the _dim1 * _dim2 * _numPoints threads does one multiply
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2 * _numPoints),
        [&] (const unsigned int threadIndex, float& localsum) {
          const unsigned int qp = threadIndex / (_dim1 * _dim2);
          const unsigned int dim1 = threadIndex % (_dim1 * _dim2) / _dim2;
          const unsigned int dim2 = threadIndex % _dim2;
          localsum +=  _leftInput(cellIndex, qp, dim1, dim2) *
          _rightInput(cellIndex, qp, dim1, dim2);
        }, sum);

    if (thread.team_rank() == 0) {
      _output(cellIndex) = sum;
    }
  }

private:
  ContractDataDataTensor_TeamDepth3Functor();
};



