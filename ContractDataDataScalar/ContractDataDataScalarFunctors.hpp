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

template <class DeviceType, class KokkosDotProductData,
          class KokkosDotProductResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _dotProductSize;
  KokkosDotProductData _data_A;
  KokkosDotProductData _data_B;
  KokkosDotProductResults _results;

  KokkosFunctor_Independent(const unsigned int dotProductSize,
                            KokkosDotProductData data_A,
                            KokkosDotProductData data_B,
                            KokkosDotProductResults results) :
    _dotProductSize(dotProductSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int dotProductIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _dotProductSize;
         ++entryIndex) {
      sum +=
        _data_A(dotProductIndex, entryIndex) *
        _data_B(dotProductIndex, entryIndex);
    }
    _results(dotProductIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};
