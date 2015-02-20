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
struct ContractDataDataTensor_TeamDepth2Functor {
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

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

    const unsigned int dim12 = _dim1 * _dim2;
    const unsigned int cellSize = _numPoints * dim12;

    float sum = 0;
    float tsum = 0;

    for (unsigned int innerIdx = tIndex; innerIdx < cellSize; innerIdx += thread.team_size() ) {
      // Sane arithmetic version:
      // qp = innerIdx / (_dim1 * _dim2)
      // iTens1 = (innerIdx % (_dim1 * _dim2)) / _dim2
      // iTens2 = innerIdx % _dim2

      // Optimized arithmetic version:
      const unsigned int qp = innerIdx / dim12;
      const unsigned int idxDivDim2 = innerIdx / _dim2;
      const unsigned int iTens1 = idxDivDim2 - _dim1 * qp;      
      const unsigned int iTens2 = innerIdx - idxDivDim2 *_dim2;

      sum +=  _leftInput(cell, qp, iTens1, iTens2) *
       _rightInput(cell, qp, iTens1, iTens2);
    }

    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, thread.team_size()),
        [&] (const unsigned int& dummy, float& localsum) {
        localsum += sum;
      }, tsum);

    // FIXME everyone is writing this?
    _output(cell) = tsum;
  }
private:
  ContractDataDataTensorTeamStrideFunctor();
};

