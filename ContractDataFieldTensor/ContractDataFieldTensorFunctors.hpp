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

#ifdef ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif // ENABLE_KOKKOS

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
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _c;
  int _l;
  int _q;
  int _d1;
  int _d2;

  KokkosFunctor_Independent(LeftViewType leftInput,
  RightViewType rightInput,
  OutputViewType output,
  int c,
  int l,
  int q,
  int d1,
  int d2) :
  _leftInput(leftInput),
  _rightInput(rightInput),
  _output(output),
  _c(c),
  _l(l),
  _q(q),
  _d1(d1),
  _d2(d2)
  {
    // Nothing to do
  }

  // Parallelize over c-loop
  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    for (int lbf = 0; lbf < _l; lbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < _q; qp++) {
        for (int iTens1 = 0; iTens1 < _d1; iTens1++) {
          for (int iTens2 =0; iTens2 < _d2; iTens2++) {
            tmpVal += _leftInput(elementIndex, lbf,qp,iTens1,iTens2) *
            _rightInput(elementIndex, qp, iTens1, iTens2);
          } // D2-loop
        } // D1-loop
      } // P-loop
      _output(elementIndex, lbf) = tmpVal;
    } // F-loop
  }
};


