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
int c, int l, int q, int d1, int d2) {

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < l; lbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < q; qp++) {
        for (int iTens1 = 0; iTens1 < d1; iTens1++) {
          for (int iTens2 =0; iTens2 < d2; iTens2++) {
            tmpVal += leftInput[cl*l*q*d1*d2+lbf*q*d1*d2+qp*d1*d2+iTens1*d2+iTens2] *
            rightInput[cl*q*d1*d2+qp*d1*d2+iTens1*d2+iTens2];
          } // D2-loop
        } // D1-loop
      } // P-loop
      outputFields[cl*l+ lbf] = tmpVal;
    } // F-loop
  } // C-loop
}

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldVectorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftInput;
  RightViewType _rightInput;
  OutputViewType _output;
  int _c;
  int _l;
  int _q;
  int _d1;
  int _d2;

  ContractFieldFieldVectorFunctor(LeftViewType leftInput,
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
};



int main(int argc, char* argv[]) {
  int c=10000, l=10, d1=10, d2 = 10, q = 10;
  const int repeats = 10;

  timespec tic;
  timespec toc;

  Kokkos::initialize();

  double* leftInput = new double[c * l * q * d1 * d2];
  double* rightInput = new double[c * q * d1 * d2];
  double* serialOutput = new double[c * l];

  typedef Kokkos::View<double *****, Kokkos::LayoutLeft, Kokkos::Cuda> dev_input_t_left;
  typedef Kokkos::View<double ****, Kokkos::LayoutLeft, Kokkos::Cuda> dev_input_t_right;
  typedef Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::Cuda> dev_output_t;
  typedef typename dev_input_t_left::HostMirror host_input_t_left;
  typedef typename dev_input_t_right::HostMirror host_input_t_right;
  typedef typename dev_output_t::HostMirror host_output_t;

  dev_input_t_left d_inputLeft("left", c, l, q, d1, d2);
  dev_input_t_right d_inputRight("right", c, q, d1, d2);
  dev_output_t d_output("out", c, l);

  host_input_t h_inputLeft = Kokkos::create_mirror_view(d_inputLeft);
  host_input_t h_inputRight = Kokkos::create_mirror_view(d_inputRight);
  host_output_t h_output = Kokkos::create_mirror_view(d_output);


  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < q; ++qp) {
      for(int dim1 = 0; dim1 < d1; ++dim1){
        for(int dim2 = 0; dim2 < d2; ++dim2) {
          double tmp1 = (double)std::rand();
          rightInput[cl * q * d1 * d2 + qp * d1 * d2 + dim1 * d2 + dim2] = tmp1;
          h_inputRight(cl,qp, dim1, dim2) = tmp1;

          for(int lbf = 0; lbf < l; ++lbf) {
            double tmp2 = (double)std::rand();
            leftInput[cl * l * q * d1 * d2 + lbf * q * d1 * d2 + qp * d1 * d2 + dim1 * d2 + dim2] = tmp2;
            h_inputLeft(cl, lbf, qp, dim1, dim2) = tmp2;
          }
        }
      }
    }
  }

  for (int cl=0; cl < c; cl++) {
      for(int lbf = 0; lbf<l; lbf++) {
        serialOutput[cl * l + lbf] = 0;
        h_output(cl, lbf) = 0;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(leftInput, rightInput, serialOutput, c, l, q,d1,d2);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;


  Kokkos::deep_copy(d_inputLeft, h_inputLeft);
  Kokkos::deep_copy(d_inputRight, h_inputRight);
  Kokkos::deep_copy(d_output, h_output);


  ContractFieldFieldVectorFunctor<Kokkos::Cuda, dev_input_t_left, dev_input_t_right, dev_output_t>
  kokkosFunctor(d_inputLeft, d_inputRight, d_output, c, l, q, d1, d2);

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
