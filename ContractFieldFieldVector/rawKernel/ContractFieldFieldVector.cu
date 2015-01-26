// -*- C++ -*-
// matrixMultiplication.cc
// a huge comparison of doing naive and tiled matrix multiplication using many
//  different methods and technologies

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>

// header file for openmp
#include <omp.h>

// header files for kokkos
#include <Kokkos_Core.hpp>
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <cuda_runtime.h>

using std::string;
using std::vector;
using Intrepid::FieldContainer;

typedef Intrepid::RealSpaceTools<double> rst;

#define BLOCK_SIZE 64;

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

__global__
void
cudaDoContractFieldFieldVector_kernel(const double * const __restrict__ d_left, const double * const __restrict__ d_right,
double * d_out,
int numCells,
int numPoints,
int dimVec,
int numLeftFields,
int numRightFields) {

  int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(myID < (numCells * numLeftFields * numRightFields)) {
    int matrixIndex = elementIndex % numCells;
    int rbf = matrixIndex % numRightFields;
    int lbf = matrixIndex % numLeftFields;

    double tmpVal = 0;
    for (int qp = 0; qp < numPoints; qp++) {
      for (int iVec = 0; iVec < dimVec; iVec++) {
        tmpVal += leftFields(cl, qp, iVec, lbf)*rightFields(cl, qp, iVec, rbf);
      } //D-loop
    } // P-loop
    outputFields(cl, lbf, rbf) = tmpVal;
  }
}

void
cudaDoContractFieldFieldVector(double * h_out,
double * h_inLeft,
double * h_inRight,
int numCells,
int numPoints,
int dimVec,
int numLeftFields,
int numRightFields,
timespec * tic,
timespec * toc) {

  double * d_right;
  double * d_left;
  double * d_out;

  cudaMalloc(&d_right, sizeof(double) * numCells  * numPoints * numRightFields * dimVec);

  cudaMalloc(&d_left, sizeof(double) * numCells * numPoints * numLeftFields * dimVec);

  cudaMalloc(&d_out, sizeof(double) * numCells * numRightFields * numLeftFields);

  cudaMemset(d_out, 0, sizeof(double) * numCells * numRightFields * numLeftFields);

  cudaMemcpy(d_right, h_inRight,
  sizeof(double) * numCells * numPoints * numRightFields, cudaMemcpyHostToDevice);

  cudaMemcpy(d_left, h_inLeft,
  sizeof(double) * numCells * numPoints * numLeftFields, cudaMemcpyHostToDevice);


  dim3 blockSize(1024);
  dim3 gridSize((numCells * numLeftFields * numRightFields / 1024) + 1);

  clock_gettime(CLOCK_MONOTONIC, tic);
  cudaDoContractFieldFieldVector_kernel<<<gridSize, blockSize>>>(d_left,
  d_right, d_out, numCells, numPoints, dimVec, numLeftFields, numRightFields);

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, toc);
  cudaMemcpy(h_out, d_out, sizeof(double) * numCells * numLeftFields * numRightFields, cudaMemcpyDeviceToHost);

}

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldVectorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftFields;
  RightViewType _rightFields;
  OutputViewType _outputFields;
  int _numCells;
  int _numPoints;
  int _numLeftFields;
  int _numRightFields;
  int _dimVec;

  ContractFieldFieldVectorFunctor(LeftViewType leftFields,
  RightViewType rightFields,
  OutputViewType outputFields,
  int numCells,
  int numPoints,
  int dimVec,
  int numLeftFields,
  int numRightFields) :
  _leftFields(leftFields),
  _rightFields(rightFields),
  _outputFields(outputFields),
  _numPoints(numPoints),
  _numLeftFields(numLeftFields),
  _numRightFields(numRightFields),
  _dimVec(dimVec)
  {
    // Nothing to do
  }

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

// Serial ContractFieldFieldVector.  Contracts FieldContainers of doubles.
void ContractFieldFieldVectorSerial(FieldContainer<double> &  outputFields,
const FieldContainer<double> &              leftFields,
const FieldContainer<double> &              rightFields,
double *                                    time = 0) {

  int numCells        = leftFields.dimension(0);
  int numLeftFields   = leftFields.dimension(1);
  int numRightFields  = rightFields.dimension(1);
  int numPoints       = leftFields.dimension(2);
  int dimVec          = leftFields.dimension(3);

  for (int cl = 0; cl < numCells; cl++) {
    for (int lbf = 0; lbf < numLeftFields; lbf++) {
      for (int rbf = 0; rbf < numRightFields; rbf++) {
        double tmpVal = 0;
        for (int qp = 0; qp < numPoints; qp++) {
          for (int iVec = 0; iVec < dimVec; iVec++) {
            tmpVal += leftFields(cl, lbf, qp, iVec)*rightFields(cl, rbf, qp, iVec);
          } //D-loop
        } // P-loop
        outputFields(cl, lbf, rbf) = tmpVal;
      } // R-loop
    } // L-loop
  } // C-loop
}


/*
* Kokkos Cuda ContractFieldFieldVector.
*
* Contracts two Kokkos Cuda host views (two double *** tensors -> one double
* *** tensor). Since
*
* Note that all input and output is in Kokkos host views --- the user is
* responsible for getting the data in and out of them.
*/
template <class DeviceType, class input_view_t, class output_view_t, class input_host_t, class output_host_t>
void ContractFieldFieldVectorKokkos(output_host_t &   outHost,
const input_host_t &                      leftHost,
const input_host_t &                      rightHost,
output_view_t &                           outDevice,
input_view_t &                            leftDevice,
input_view_t &                            rightDevice,
double *                                  time = 0) {

  // get sizes
  int numCells        = leftFields.dimension(0);
  int numPoints       = leftFields.dimension(1);
  int dimVec          = leftFields.dimension(2);
  int numLeftFields   = leftFields.dimension(3);
  int numRightFields  = rightFields.dimension(3);

  // Deep copy Kokkos host views into device views
  Kokkos::deep_copy(leftDevice, leftHost);
  Kokkos::deep_copy(rightDevice, rightHost);
  Kokkos::deep_copy(outDevice, outHost);

  timespec tic;
  if(time != 0)
    clock_gettime(CLOCK_MONOTONIC, &tic);

    ContractFieldFieldVectorFunctor<DeviceType, input_view_t, input_view_t, output_view_t>
    kokkosFunctor(leftDevice, rightDevice, outDevice, numCells, numPoints, dimVec,
    numLeftFields, numRightFields);

    Kokkos::parallel_for(numCells * numRightFields * numLeftFields, kokkosFunctor);

    Kokkos::fence();

    timespec toc;
    if(time !=0){
      clock_gettime(CLOCK_MONOTONIC, &toc);
      *time += getElapsedTime(tic, toc);
    }

    Kokkos::deep_copy(outHost, outDevice);
  }



  int main(int argc, char* argv[]) {
    int c=10000, p=10, l=10, r=10, q = 10, i = 10;

    FieldContainer<double> in_c_l_q_i(c, l, q, i);
    FieldContainer<double> in_c_r_q_i(c, r, q, i);
    FieldContainer<double> out1_c_l_r(c, l, r);
    FieldContainer<double> out2_c_l_r(c, l, r);
    double zero = Intrepid::INTREPID_TOL*100000.0;

    // fill with random numbers
    for (int i=0; i<in_c_l_q_i.size(); i++) {
      in_c_l_p[i] = Teuchos::ScalarTraits<double>::random();
    }
    for (int i=0; i<in_c_r_q_i.size(); i++) {
      in_c_r_p[i] = Teuchos::ScalarTraits<double>::random();
    }
    std::cout << "Created vectors" << std::endl;

    // ===============================================================
    // ********************** < Kokkos setup> ************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // Doing all of this here might throw off the timing -- we're not counting the
    // cost of the copy into Kokkos or the deep copy from Kokkos host to Kokkos
    // device.

    Kokkos::initialize();

    // Kokkos Cuda views
    typedef Kokkos::View<double ****, Kokkos::Cuda> cuda_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::Cuda> cuda_output_view_t;
    typedef typename cuda_input_view_t::HostMirror cuda_input_host_t;
    typedef typename cuda_output_view_t::HostMirror cuda_output_host_t;

    // Kokkos OpenMP views
    typedef Kokkos::View<double ****, Kokkos::OpenMP> omp_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::OpenMP> omp_output_view_t;
    typedef typename omp_input_view_t::HostMirror omp_input_host_t;
    typedef typename omp_output_view_t::HostMirror omp_output_host_t;


    //Cuda arrays

    double * cudaRight = new double[c * r * q * i];
    double * cudaLeft = new double[c * l * q * i];

    double * cudaOut = new double[c * l * r];


    // Make equivalent Kokkos views
    cuda_input_view_t cuda_kokkosLeft("left_input", c,q,i,l);
    cuda_input_view_t cuda_kokkosRight("right_input", c,q,i,r);
    cuda_output_view_t cuda_kokkosOut("output", c, l, r );

    omp_input_view_t omp_kokkosLeft("left_input", c, q,i,l);
    omp_input_view_t omp_kokkosRight("right_input",  c, q,i,r);
    omp_output_view_t omp_kokkosOut("output", c,l ,r);

    // And their host mirrors

    cuda_input_host_t cuda_hostLeft = Kokkos::create_mirror_view(cuda_kokkosLeft);
    cuda_input_host_t cuda_hostRight = Kokkos::create_mirror_view(cuda_kokkosRight);
    cuda_output_host_t cuda_hostOut = Kokkos::create_mirror_view(cuda_kokkosOut);

    omp_input_host_t omp_hostLeft = Kokkos::create_mirror_view(omp_kokkosLeft);
    omp_input_host_t omp_hostRight = Kokkos::create_mirror_view(omp_kokkosRight);
    omp_output_host_t omp_hostOut = Kokkos::create_mirror_view(omp_kokkosOut);

    // Copy into Kokkos host views and cuda
    // ORDERING IN KOKKOS GOES CL, QP, I, RBF/LBF
    for (int cl = 0; cl < c; ++cl) {
      for (int qp = 0; qp < p; ++qp) {
        for(int ivec = 0; ivec < i; ++ivec){
          for(int rbf = 0; rbf < r; ++rbf) {
            cuda_hostRight(cl,qp, ivec, rbf) = in_c_r_q_i(cl,rbf,qp,ivec);
            omp_hostRight(cl,qp, ivec, rbf) = in_c_r_p(cl,rbf,qp,ivec);

            cudaRight[cl * p * i * r + qp * i * r + ivec * r + rbf] = in_c_r_p(cl,rbf,qp,ivec);
          }

          for(int lbf = 0; lbf < l; ++lbf) {
            cuda_hostLeft(cl,qp, ivec, lbf) = in_c_l_p(cl,lbf,qp,ivec);
            omp_hostLeft(cl,rbf,qp,ivec) = in_c_l_p(cl,lbf,qp,ivec);

            cudaLeft[cl * p * i * l + qp * i * l + ivec * l + lbf] = in_c_l_p(cl,lbf,qp,ivec);
          }
        }
      }
    }



    // ===============================================================
    // ********************** </Kokkos setup> ************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    std::cout << "trying serial" << std::endl;

    //Warmup
    ContractFieldFieldVectorSerial(out2_c_l_r, in_c_l_p, in_c_r_p);

    timespec tic;
    clock_gettime(CLOCK_MONOTONIC, &tic);

    //repeat the calculation 5 times so we can average out some randomness
    for(int i = 0; i < 5; ++i){
      ContractFieldFieldVectorSerial(out2_c_l_r, in_c_l_p, in_c_r_p);
    }

    timespec toc;
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_serial = getElapsedTime(tic, toc);

    printf("trying kokkos openmp\n");

    //Warmpup
    ContractFieldFieldVectorKokkos<Kokkos::OpenMP, omp_input_view_t,
    omp_output_view_t, omp_input_host_t, omp_output_host_t>
    (omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
    omp_kokkosLeft, omp_kokkosRight);
    clock_gettime(CLOCK_MONOTONIC, &tic);
    //repeat the calculation 5 times so we can average out some randomness
    for(int i = 0; i < 5; ++i){
      ContractFieldFieldVectorKokkos<Kokkos::OpenMP, omp_input_view_t,
      omp_output_view_t, omp_input_host_t, omp_output_host_t>
      (omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
      omp_kokkosLeft, omp_kokkosRight);
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_kokkos_omp = getElapsedTime(tic, toc);
    // Copy out from kokkos output view (NOT timing this)
    for (int cl = 0; cl < c; ++cl) {
      for(int lbf = 0; lbf < l; ++lbf) {
        for(int rbf = 0; rbf < r; ++rbf) {
          out1_c_l_r(cl,lbf,rbf) = omp_hostOut(cl,lbf,rbf);
        }
      }
    }
    rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
    if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
      std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
      << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
    }
    std::cout << "kokkos omp speedup of " << elapsedTime_serial/elapsedTime_kokkos_omp << std::endl;


    printf("trying kokkos cuda\n");

    //Warmpup
    ContractFieldFieldVectorKokkos<Kokkos::Cuda, cuda_input_view_t,
    cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
    (cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
    cuda_kokkosLeft, cuda_kokkosRight);
    clock_gettime(CLOCK_MONOTONIC, &tic);
    //repeat the calculation 5 times so we can average out some randomness
    for(int i = 0; i < 5; ++i){
      ContractFieldFieldVectorKokkos<Kokkos::Cuda, cuda_input_view_t,
      cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
      (cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
      cuda_kokkosLeft, cuda_kokkosRight);
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_kokkos_cuda = getElapsedTime(tic, toc);
    // Copy out from kokkos output view (NOT timing this)
    for (int cl = 0; cl < c; ++cl) {
      for(int lbf = 0; lbf < l; ++lbf) {
        for(int rbf = 0; rbf < r; ++rbf) {
          out1_c_l_r(cl,lbf,rbf) = omp_hostOut(cl,lbf,rbf);
        }
      }
    }
    rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
    if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
      std::cout << "\n\nINCORRECT contractFieldFieldTensor (0): check COMP_CPP vs. COMP_KOKKOS; "
      << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
    }
    std::cout << "kokkos cuda speedup of " << elapsedTime_serial/elapsedTime_kokkos_cuda << std::endl;

    Kokkos::finalize();

    std::cout << "trying cuda major" << std::endl;
    //Now try the cuda version, start with warmup
    cudaDoContractFieldFieldVector(cudaOut,cudaLeft,cudaRight, c, p, i, l, r, &tic, &toc);
    double elapsedTime_cuda = 0;

    for(int i = 0; i < 5; ++i){
      cudaDoContractFieldFieldVector(cudaOut,cudaLeft,cudaRight, c, p, i, l, r, &tic, &toc);
      elapsedTime_cuda += getElapsedTime(tic,toc);
    }

    for (int cl = 0; cl < c; ++cl) {
      for(int lbf = 0; lbf < l; ++lbf) {
        for(int rbf = 0; rbf < r; ++rbf) {
          out1_c_l_r(cl,lbf,rbf) = cudaOut[cl * l * r + lbf * r + rbf];
        }
      }
    }

    rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
    if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
      std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
      << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
    }

    std::cout << "cuda speedup of " << elapsedTime_serial/elapsedTime_cuda << std::endl;


    return 0;
  }
