/*
 *
 * Contract array of matrices (c, l, p) to array of vectors (c, p) to get array
 * of vectors (c, l).
 *
 * cached: (c, l, p) by (c, p)
 * coalesced: (c, p, l) by (c, p) parallelized over c and l
 *
 */


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <Kokkos_Core.hpp>
#include <unistd.h>
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


void serial(double* inputFields, double* inputData, double* output,
            int c, int p, int f) {

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < p; qp++) {
        tmpVal += inputFields[cl * f * p + lbf * p + qp]*inputData[cl * p + qp];
      } // P-loop
      output[cl * f + lbf] = tmpVal;
    } // F-loop
  } // C-loop
}

template<class DeviceType, class FieldViewType, class DataViewType, class OutputViewType>
struct ContractFieldFieldScalarFunctor {
  typedef DeviceType device_type;
  FieldViewType _inputFields;
  DataViewType _inputData;
  OutputViewType _output;
  int _numPoints;
  int _numFields;

  ContractFieldFieldScalarFunctor(FieldViewType inputFields,
      DataViewType inputData,
      OutputViewType output,
      int numPoints,
      int numFields) :
    _inputFields(inputFields),
    _inputData(inputData),
    _output(output),
    _numPoints(numPoints),
    _numFields(numFields)
  {
    // Nothing to do
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    int cl = elementIndex / _numFields;
    int lbf = elementIndex % _numFields;

    double tmpVal = 0;
    for (int qp = 0; qp < _numPoints; qp++) {
      tmpVal += _inputFields(cl, qp, lbf) * _inputData(cl,  qp);
    } // P-loop
    _output(cl, lbf) = tmpVal;
  }
};



int main(int argc, char* argv[]) {
  int c=1000000, p=10, f=10;
  int repeats = 10;

  timespec tic;
  timespec toc;

  Kokkos::initialize();


  // Setup

  double* inputFields = new double[c * p * f];
  double* inputData = new double[c * p];
  double* serialOutput = new double[c * f];

  typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda> cuda3d_t;
  typedef Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::Cuda> cuda2d_t;
  typedef typename cuda3d_t::HostMirror host_cuda3d_t;
  typedef typename cuda2d_t::HostMirror host_cuda2d_t;

  cuda3d_t cuda_inputFields("inputFields", c, p, f);
  cuda2d_t cuda_inputData("inputData", c, p);
  cuda2d_t cuda_output("output", c, f);

  host_cuda3d_t hostCuda_inputFields = Kokkos::create_mirror_view(cuda_inputFields);
  host_cuda2d_t hostCuda_inputData = Kokkos::create_mirror_view(cuda_inputData);
  host_cuda2d_t hostCuda_output = Kokkos::create_mirror_view(cuda_output);

  double tmp;
  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      for (int qp = 0; qp < p; qp++) {
        tmp = (double)std::rand();
        inputFields[cl * f * p + lbf * p + qp] = tmp;
        hostCuda_inputFields(cl, qp, lbf) = tmp;
      } // P-loop
    } // F-loop
  } // C-loop

  for (int cl = 0; cl < c; cl++) {
    for (int qp = 0; qp < p; qp++) {
      tmp = (double)std::rand();
      inputData[cl * p + qp] = tmp;
      hostCuda_inputData(cl, qp) = tmp;
    } // P-loop
  } // C-loop

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      serialOutput[cl * f + lbf] = 0;
      hostCuda_output(cl, lbf) = 0;
    } // F-loop
  } // C-loop


  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(inputFields, inputData, serialOutput, c, p, f);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;


  // Kokkos

  Kokkos::deep_copy(cuda_inputFields, hostCuda_inputFields);
  Kokkos::deep_copy(cuda_inputData, hostCuda_inputData);
  Kokkos::deep_copy(cuda_output, hostCuda_output);

  ContractFieldFieldScalarFunctor<Kokkos::Cuda, cuda3d_t, cuda2d_t, cuda2d_t>
    kokkosFunctor(cuda_inputFields, cuda_inputData, cuda_output, p, f);

  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c*f, kokkosFunctor);
    Kokkos::fence();
  }


  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    Kokkos::parallel_for(c*f, kokkosFunctor);
    Kokkos::fence();
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_kokkosCuda = getElapsedTime(tic, toc);

  Kokkos::deep_copy(hostCuda_output, cuda_output);

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      double err = serialOutput[cl * f + lbf] / hostCuda_output(cl, lbf);
      if ((abs(err) - 1) > 1.0e-6) {
        std::cerr << "output mismatch at" << cl << ", "<< lbf << std::endl;
        std::cerr << "diff: " << err << std::endl;
        std::cerr << "serial: " << serialOutput[cl * f + lbf] << std::endl;
        std::cerr << "kokkos: " << hostCuda_output(cl, lbf) << std::endl;
        exit(0);
      }
    } // F-loop
  } // C-loop

  std::cout << "kokkos cuda time: " << elapsedTime_kokkosCuda << std::endl;
  std::cout << "kokkos cuda speedup: " << elapsedTime_serial/elapsedTime_kokkosCuda << std::endl;

  Kokkos::finalize();
}
