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


template<class DeviceType, class DataViewType, class FieldViewType, class OutputViewType>
struct ContractDataFieldScalarFunctor {
  typedef DeviceType device_type;
  FieldViewType _inputFields;
  DataViewType _inputData;
  OutputViewType _output;
  int _numPoints;
  int _numFields;

  ContractDataFieldScalarFunctor(int numPoints,
      int numFields,
      FieldViewType inputFields,
      DataViewType inputData,
      OutputViewType output) :
    _inputFields(inputFields),
    _inputData(inputData),
    _output(output),
    _numPoints(numPoints),
    _numFields(numFields)
  {
    // Nothing to do
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    int cl = elementIndex;// / _numFields;
    //int lbf = elementIndex % _numFields;

    for (int lbf = 0; lbf < _numFields; lbf ++) {
      double tmpVal = 0;
      for (int qp = 0; qp < _numPoints; qp++) {
        tmpVal += _inputFields(cl, lbf, qp) * _inputData(cl,  qp);
      }
      _output(cl, lbf) = tmpVal;
    } // P-loop
  }

private:
  ContractDataFieldScalarFunctor();
};

