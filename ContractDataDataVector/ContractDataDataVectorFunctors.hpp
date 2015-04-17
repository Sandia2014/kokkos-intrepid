


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

template <class DeviceType, class KokkosInputView,
          class KokkosResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _numPoints;
  const unsigned int _dimVec;
  KokkosInputView _leftView;
  KokkosInputView _rightView;
  KokkosResults _results;

  KokkosFunctor_Independent(const unsigned int numPoints,
                            const unsigned int dimVec,
                            KokkosInputView leftView,
                            KokkosInputView rightView,
                            KokkosResults results) :
    _numPoints(numPoints),
    _dimVec(dimVec),
    _leftView(leftView), _rightView(rightView),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int cl) const {

    float tmpVal = 0;
    // Each thread does one contraction independently
    for (int qp = 0; qp < _numPoints; qp++) {
      for (int iVec = 0; iVec < _dimVec; iVec++) {
        tmpVal +=
          _leftView(cl, qp, iVec) *
          _rightView(cl, qp, iVec);;
      } // D-loop
    } // P-loop
    _results(cl) = tmpVal;

  }

private:
  KokkosFunctor_Independent();

};
