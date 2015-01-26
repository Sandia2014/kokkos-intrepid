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
int c, int p, int t1, int t2) {

  for (int cl=0; cl < c; cl++) {
    double tmp = 0;
    for (int qp=0; qp < p; qp++) {
      for (int iTens1=0; iTens1 < t1; iTens1++) {
        for (int iTens2=0; iTens2 < t2; iTens2++) {
          tmp += leftInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] *
          rightInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2];
        }
      }
    }
    output[cl] = tmp;
  }
}

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractDataDataTensorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _numPoints;
  int _dim1;
  int _dim2;

  ContractDataDataTensorFunctor(LeftViewType leftInput,
  RightViewType rightInput,
  OutputViewType output,
  int numPoints,
  int dim1,
  int dim2) :
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
};



int main(int argc, char* argv[]) {
  const int c=100000, p=10, t1=10, t2=10;
  const int repeats = 10;

  timespec tic;
  timespec toc;

  Kokkos::initialize();

  double* leftInput = new double[c * p * t1 * t2];
  double* rightInput = new double[c * p * t1 * t2];
  double* serialOutput = new double[c];

  typedef Kokkos::View<double ****, Kokkos::LayoutLeft, Kokkos::Cuda> dev_input_t;
  typedef Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::Cuda> dev_output_t;
  typedef typename dev_input_t::HostMirror host_input_t;
  typedef typename dev_output_t::HostMirror host_output_t;

  dev_input_t d_inputLeft("left", c, p, t1, t2);
  dev_input_t d_inputRight("right", c, p, t1, t2);
  dev_output_t d_output("out", c);

  host_input_t h_inputLeft = Kokkos::create_mirror_view(d_inputLeft);
  host_input_t h_inputRight = Kokkos::create_mirror_view(d_inputRight);
  host_output_t h_output = Kokkos::create_mirror_view(d_output);


  for (int cl=0; cl < c; cl++) {
    for (int qp=0; qp < p; qp++) {
      for (int iTens1=0; iTens1 < t1; iTens1++) {
        for (int iTens2=0; iTens2 < t2; iTens2++) {
          double tmp1 = (double)std::rand();
          double tmp2 = (double)std::rand();
          leftInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] = tmp1;
          rightInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] = tmp2;
          h_inputLeft(cl, qp, iTens1, iTens2) = tmp1;
          h_inputRight(cl, qp, iTens1, iTens2) = tmp2;
        }
      }
    }
  }

  for (int cl=0; cl < c; cl++) {
    serialOutput[cl] = 0;
    h_output(cl) = 0;
  }


  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(leftInput, rightInput, serialOutput, c, p, t1, t2);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;


  Kokkos::deep_copy(d_inputLeft, h_inputLeft);
  Kokkos::deep_copy(d_inputRight, h_inputRight);
  Kokkos::deep_copy(d_output, h_output);


  ContractDataDataTensorFunctor<Kokkos::Cuda, dev_input_t, dev_input_t, dev_output_t>
  kokkosFunctor(d_inputLeft, d_inputRight, d_output, p, t1, t2);

  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c, kokkosFunctor);
    Kokkos::fence();
  }

  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c, kokkosFunctor);
    Kokkos::fence();
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_kokkosCuda = getElapsedTime(tic, toc);

  Kokkos::deep_copy(h_output, d_output);

  for (int cl=0; cl < c; cl++) {
    double err = serialOutput[cl] / h_output(cl);
    if ((abs(err) - 1) > 1.0e-6) {
      std::cerr << "output mismatch at" << cl << std::endl;
      std::cerr << "err: " << err << std::endl;
    }
  }
  std::cout << "kokkos cuda time: " << elapsedTime_kokkosCuda << std::endl;
  std::cout << "kokkos cuda speedup: " << elapsedTime_serial/elapsedTime_kokkosCuda << std::endl;

  Kokkos::finalize();

}
