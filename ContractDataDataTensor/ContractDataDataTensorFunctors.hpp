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


template <class DeviceType, class KokkosJunkVector>
struct KokkosFunctor_ClearCache {

  typedef size_t     value_type;
  typedef DeviceType device_type;
  typedef typename Kokkos::TeamPolicy<DeviceType> team_policy;
  typedef typename team_policy::member_type team_member;

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

    const unsigned int elementIndex = thread.league_rank();
    const unsigned int dim1 = thread.team_rank() / _dim2;
    const unsigned int dim2 = thread.team_rank() % _dim2;

    float sum = 0;
    float tsum = 0;


    for (unsigned int qp=0; qp < _numPoints; ++qp) {
        sum +=  _leftInput(elementIndex, qp, dim1, dim2) *
          _rightInput(elementIndex, qp, dim1, dim2);
    }

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2),
        [&] (const unsigned int& dim, float& localsum) {
        localsum += sum;
      }, tsum);

    // FIXME everyone is writing this?
    _output(elementIndex) = tsum;
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

    const unsigned int elementIndex = thread.league_rank();
    const unsigned int dim2 = thread.team_rank();

    float sum = 0;
    float tsum = 0;


    for (unsigned int qp=0; qp < _numPoints; ++qp) {
      for (unsigned int d1=0; d1 < _dim1; ++d1) {
        sum +=  _leftInput(elementIndex, qp, d1, dim2) *
          _rightInput(elementIndex, qp, d1, dim2);
      }
    }

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim2),
        [&] (const unsigned int& dim, float& localsum) {
        localsum += sum;
      }, tsum);

    // FIXME everyone is writing this?
    _output(elementIndex) = tsum;
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



// kokkos setup
// typedef Kokkos::Cuda                                DeviceType;

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

    const unsigned int cellIndex = thread.league_rank();

    const unsigned int dim12 = _dim1 * _dim2;
    const unsigned int cellSize = _numPoints * dim12;

    double sum = 0;

    Kokkos::parallel_reduce
      (Kokkos::TeamThreadLoop(thread,cellSize),
       [&](const unsigned int indexWithinContraction, double & localsum) {
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

#if 0

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& thread) const {

    const unsigned int cell = thread.league_rank();
    const unsigned int dim12 = _dim1 * _dim2;
    const unsigned int cellSize = _numPoints * dim12;

    float sum = 0;

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, cellSize),
        [&] (const unsigned int innerIdx, float& localsum) {
          const unsigned int qp = innerIdx / dim12;
          const unsigned int indexWithinTens1Tens2 = innerIdx - qp * dim12;
          const unsigned int iTens1 = indexWithinTens1Tens2 / _dim2;
          const unsigned int iTens2 = indexWithinTens1Tens2 - iTens1*_dim2;

          localsum +=  _leftInput(cell, qp, iTens1, iTens2) *
                       _rightInput(cell, qp, iTens1, iTens2);
        },
        sum);

    if (thread.team_rank() == 0) {
      _output(cell) = sum;
    }
  }
#endif
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

    const unsigned int elementIndex = thread.league_rank();
    const unsigned int threadIndex = thread.team_rank();

    const unsigned int qp = threadIndex / (_dim1 * _dim2);
    const unsigned int dim1 = threadIndex % (_dim1 * _dim2) / _dim2;
    const unsigned int dim2 = threadIndex % _dim2;

    float sum = 0;
    float tsum = 0;


    sum +=  _leftInput(elementIndex, qp, dim1, dim2) *
            _rightInput(elementIndex, qp, dim1, dim2);

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _dim1 * _dim2 * _numPoints),
        [&] (const unsigned int& dim, float& localsum) {
        localsum += sum;
      }, tsum);

    // FIXME everyone is writing this?
    _output(elementIndex) = tsum;
  }

private:
  ContractDataDataTensor_TeamDepth3Functor();
};



