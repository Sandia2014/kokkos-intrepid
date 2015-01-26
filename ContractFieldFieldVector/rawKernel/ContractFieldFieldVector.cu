/*
*
* Framework shamelessly stolen from Ellen Hui
*/


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>
#include <assert.h>

// header file for openmp
#include <omp.h>

// header files for kokkos
#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>

//Pre-C++11 timing (thanks jeff)
double getElapsedTime(const timespec start, const timespec end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}

void serial(double* leftInput, double* rightInput, double* output,
int c, int l, int r, int q, int i) {

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < l; lbf++) {
      for (int rbf = 0; rbf < r; rbf++) {
        double tmpVal = 0;
        for (int qp = 0; qp < q; qp++) {
          for (int iVec = 0; iVec < i; iVec++) {
            tmpVal += leftInput[cl*l*q*i+lbf*q*i+qp*i+iVec]*rightInput[cl*r*q*i+rbf*q*i+qp*i+iVec];
          } //D-loop
        } // P-loop
        output[cl*l*r+lbf*r+rbf] = tmpVal;
      } // R-loop
    } // L-loop
    }
}

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldVectorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numCells;
  int _numPoints;
  int _numLeftFields;
  int _numRightFields;
  int _dimVec;

  ContractFieldFieldVectorFunctor(LeftViewType leftInput,
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

    int matrixIndex = elementIndex / (_numRightFields*_numLeftFields);
    int rbf = matrixIndex % _numRightFields;
    int lbf = matrixIndex / _numRightFields;

    double tmpVal = 0;
    for (int qp = 0; qp < _numPoints; qp++) {
      for (int iVec = 0; iVec < _dimVec; iVec++) {
        tmpVal += _leftInput(matrixIndex, qp, iVec, lbf)*_rightInput(matrixIndex, qp, iVec, rbf);
      } //D-loop
    } // P-loop
    _output(matrixIndex, lbf, rbf) = tmpVal;
  }
};



int main(int argc, char* argv[]) {
  int c=1, l=1, r=10, q = 10, i = 10;
  const int repeats = 10;

  timespec tic;
  timespec toc;

  Kokkos::initialize();

  double* leftInput = new double[c * l * q * i];
  double* rightInput = new double[c * r * q * i];
  double* serialOutput = new double[c * l * r];

  typedef Kokkos::View<double ****, Kokkos::LayoutRight, Kokkos::Cuda> dev_input_t;
  typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda> dev_output_t;
  typedef typename dev_input_t::HostMirror host_input_t;
  typedef typename dev_output_t::HostMirror host_output_t;

  dev_input_t d_inputLeft("left", c, q, i, l);
  dev_input_t d_inputRight("right", c, q, i, r);
  dev_output_t d_output("out", c, l, r);

  host_input_t h_inputLeft = Kokkos::create_mirror_view(d_inputLeft);
  host_input_t h_inputRight = Kokkos::create_mirror_view(d_inputRight);
  host_output_t h_output = Kokkos::create_mirror_view(d_output);


  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < q; ++qp) {
      for(int ivec = 0; ivec < i; ++ivec){
        for(int rbf = 0; rbf < r; ++rbf) {
          double tmp1 = (double)std::rand();
          rightInput[cl * q * i * r + rbf * q * i + qp * i + ivec] = tmp1;
          h_inputRight(cl,qp, ivec, rbf) = tmp1;
        }

        for(int lbf = 0; lbf < l; ++lbf) {
          double tmp2 = (double)std::rand();
          leftInput[cl * q * i * l + lbf * q * i + qp * i + ivec] = tmp2;
          h_inputLeft(cl, qp, ivec, lbf) = tmp2;
        }
      }
    }
  }

  for (int cl=0; cl < c; cl++) {
    for(int rbf =0; rbf < r; rbf++) {
      for(int lbf = 0; lbf<l; lbf++) {
        serialOutput[cl * r * l + lbf*r + rbf] = 0;
        h_output(cl, lbf, rbf) = 0;
      }
    }
  }


  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(leftInput, rightInput, serialOutput, c, l, r,q,i);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;


  Kokkos::deep_copy(d_inputLeft, h_inputLeft);
  Kokkos::deep_copy(d_inputRight, h_inputRight);
  Kokkos::deep_copy(d_output, h_output);


  ContractFieldFieldVectorFunctor<Kokkos::Cuda, dev_input_t, dev_input_t, dev_output_t>
  kokkosFunctor(d_inputLeft, d_inputRight, d_output, c, l, r, q, i);

  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c*l*r, kokkosFunctor);
    Kokkos::fence();
  }

  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c*l*r, kokkosFunctor);
    Kokkos::fence();
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_kokkosCuda = getElapsedTime(tic, toc);

  Kokkos::deep_copy(h_output, d_output);

  for (int cl=0; cl < c; cl++) {
    for(int lbf = 0; lbf < l; lbf++){
      for(int rbf = 0; rbf < r; rbf++){

      double err = serialOutput[cl*l*r + lbf*r + rbf] / h_output(cl, lbf,rbf);
      if ((abs(err) - 1) > 1.0e-6) {
        std::cerr << "output mismatch at" << cl*l*r+lbf*r+rbf << std::endl;
        std::cerr << "Serial is" << serialOutput[cl*l*r + lbf*r + rbf] << "kokkos is" << h_output(cl,lbf,rbf) << std::endl;
      }
      }
  }
  }
  std::cout << "kokkos cuda time: " << elapsedTime_kokkosCuda << std::endl;
  std::cout << "kokkos cuda speedup: " << elapsedTime_serial/elapsedTime_kokkosCuda << std::endl;

  Kokkos::finalize();

}
