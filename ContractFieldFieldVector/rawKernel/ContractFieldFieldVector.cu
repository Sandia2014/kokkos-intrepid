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
int c, int p, int l, int r, int q, int i) {

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < l; lbf++) {
      for (int rbf = 0; rbf < r; rbf++) {
        double tmpVal = 0;
        for (int qp = 0; qp < q; qp++) {
          for (int iVec = 0; iVec < i; iVec++) {
            tmpVal += leftFields(cl, lbf, qp, iVec)*rightFields(cl, rbf, qp, iVec);
          } //D-loop
        } // P-loop
        outputFields[cl * lbf * rbf] = tmpVal;
      } // R-loop
    } // L-loop
  } // C-loop
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
  int numPoints,
  int dimVec,
  int numLeftFields,
  int numRightFields) :
  _leftInput(leftInput),
  _rightInput(rightInput),
  _output(output),
  _numCells(numCells),
  _numPoints(numPoints),
  _numLeftFields(numLeftFields),
  _numRightFields(numRightFields),
  _dimVec(dimVec)
  {
    // Nothing to do
  }

  // Parallelize over c-loop
  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {

    int matrixIndex = elementIndex % _numCells;
    int rbf = matrixIndex % _numRightFields;
    int lbf = matrixIndex % _numLeftFields;

    double tmpVal = 0;
    for (int qp = 0; qp < numPoints; qp++) {
      for (int iVec = 0; iVec < dimVec; iVec++) {
        tmpVal += leftFields(cl, qp, iVec, lbf)*rightFields(cl, qp, iVec, rbf);
      } //D-loop
    } // P-loop
    outputFields(cl, lbf, rbf) = tmpVal;
  }
};



int main(int argc, char* argv[]) {
  int c=10000, l=10, r=10, q = 10, i = 10;
  const int repeats = 10;

  timespec tic;
  timespec toc;

  Kokkos::initialize();

  double* leftInput = new double[c * l * q * i];
  double* rightInput = new double[c * r * q * i];
  double* serialOutput = new double[c * l * r];

  typedef Kokkos::View<double ****, Kokkos::LayoutLeft, Kokkos::Cuda> dev_input_t;
  typedef Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::Cuda> dev_output_t;
  typedef typename dev_input_t::HostMirror host_input_t;
  typedef typename dev_output_t::HostMirror host_output_t;

  dev_input_t d_inputLeft("left", c, q, i, l);
  dev_input_t d_inputRight("right", c, q, i, r);
  dev_output_t d_output("out", c, l, r);

  host_input_t h_inputLeft = Kokkos::create_mirror_view(d_inputLeft);
  host_input_t h_inputRight = Kokkos::create_mirror_view(d_inputRight);
  host_output_t h_output = Kokkos::create_mirror_view(d_output);


  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < p; ++qp) {
      for(int ivec = 0; ivec < i; ++ivec){
        for(int rbf = 0; rbf < r; ++rbf) {
          double tmp1 = (double)std::rand();
          rightInput[cl * p * i * r + rbf * p * i + qp * i + ivec] = tmp2;
          h_inputRight(cl,qp, ivec, rbf) = tmp2;
        }

        for(int lbf = 0; lbf < l; ++lbf) {
          double tmp2 = (double)std::rand();
          leftInput[cl * p * i * l + lbf * p * i + qp * i + ivec] = tmp2;
          h_inputLeft(cl, qp, ivec, lbf) = tmp2;
        }
      }
    }
  }

  for (int cl=0; cl < c; cl++) {
    for(int rbf =0; rbf < r; rbf++) {
      for(int lbf = 0; lbf<l; lbf++) {
        serialOutput[cl * rbf * lbf] = 0;
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


  ContractDataDataTensorFunctor<Kokkos::Cuda, dev_input_t, dev_input_t, dev_output_t>
  kokkosFunctor(d_inputLeft, d_inputRight, d_output, c,q, l, r,i);

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
