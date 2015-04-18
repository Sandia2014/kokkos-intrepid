// -*- C++ -*-
// ArrayOfDotProducts.cc
// a huge comparison of different ways of doing an array of dot products
// Jeff Amelang, 2014

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

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numCells;
  int _numPoints;
  int _numLeftFields;
  int _numRightFields;
  int _dimVec;

  KokkosFunctor_Independent(LeftViewType leftInput,
  RightViewType rightInput,
  OutputViewType output,
  int c,
  int l,
  int r,
  int q,
  int i) :
  _leftInput(leftInput),
  _rightInput(rightInput),
  _output(output),
  _numCells(c),
  _numPoints(q),
  _numLeftFields(l),
  _numRightFields(r),
  _dimVec(i)
  {
    // Nothing to do
  }

  // Parallelize over c-loop
  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    int cellNum = elementIndex / (_numLeftFields*_numRightFields);
    int fieldsIndex = elementIndex % (_numLeftFields*_numRightFields);
    int leftFieldNum = fieldsIndex / _numRightFields;
    int rightFieldNum = fieldsIndex % _numRightFields;

    // Each thread does one element of one output cell
    double tmpVal = 0;
    for (int qp = 0; qp < _numPoints; qp++) {
      for (int iVec = 0; iVec < _dimVec; iVec++) {
        tmpVal += _leftInput(cellNum, leftFieldNum, qp, iVec)*_rightInput(cellNum,rightFieldNum,qp, iVec);
      } //D-loop
    } // P-loop

    _output(cellNum, leftFieldNum, rightFieldNum) = tmpVal;
  }
};
