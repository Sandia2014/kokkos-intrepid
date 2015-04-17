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

template <class DeviceType, class KokkosDataA, class KokkosDataB,
          class KokkosDotProductResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _numCells;
  const unsigned int _numPoints;
  const unsigned int _dimVec;
  KokkosDataA _data_A;
  KokkosDataB _data_B;
  KokkosDotProductResults _results;

  KokkosFunctor_Independent(const unsigned int numCells,
                            const unsigned int numPoints,
                            const unsigned int dimVec,
                            KokkosDataA data_A,
                            KokkosDataB data_B,
                            KokkosDotProductResults results) :
    _numCells(numCells),
    _numPoints(numPoints), 
    _dimVec(dimVec),
    _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    const unsigned int cl = elementIndex % _numCells;
    const unsigned int lbf = elementIndex / _numCells;

    float tmpVal = 0;
    for (int qp = 0; qp < _numPoints; qp++) {
      for (int iVec = 0; iVec < _dimVec; iVec++) {
        tmpVal += 
          _data_A(cl, lbf, qp, iVec) *
          _data_B(cl, qp, iVec);
      } // D-loop
    } // P-loop
    _results(cl, lbf) = tmpVal;
    
  }

private:
  KokkosFunctor_Independent();

};
