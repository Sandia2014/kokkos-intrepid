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

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;
  LeftViewType _lbfeftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _c;
  int _lbf;
  int _numPoints;
  int _dim1;
  int _dim2;

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
  _lbf(l),
  _numPoints(q),
  _dim1(d1),
  _dim2(d2)
  {
    // Nothing to do
  }

  // Parallelize over c-loop
  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    for (int lbf = 0; lbf < _lbf; lbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < _numPoints; qp++) {
        for (int iTens1 = 0; iTens1 < _dim1; iTens1++) {
          for (int iTens2 =0; iTens2 < _dim2; iTens2++) {
            tmpVal += _leftInput(elementIndex, lbf,qp,iTens1,iTens2) *
            _rightInput(elementIndex, qp, iTens1, iTens2);
          } // D2-loop
        } // D1-loop
      } // P-loop
      _output(elementIndex, lbf) = tmpVal;
    } // F-loop
  }
};


