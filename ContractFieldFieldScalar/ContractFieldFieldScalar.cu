// -*- C++ -*-

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

#include <cuda_runtime.h>

using std::string;
using std::vector;

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

//Hacky random double... Right now it's just ints casted that way
double random_double() {
  return (double) rand();
}

__global__
void
cudaDocontractFieldFieldScalar_kernel(const double * const __restrict__ d_left, const double * const __restrict__ d_right,
    double * d_out,
    int numCells,
    int numLeftFields,
    int numRightFields,
    int numPoints) {

  int myID = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(myID < (numCells * numLeftFields * numRightFields)) {

    int myMatrix = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);
    int matrixRow = matrixIndex / numRightFields;
    int matrixCol = matrixIndex % numRightFields;
    double temp = 0;
    for (int qp = 0; qp < numPoints; qp++) {

      temp += d_left[myMatrix*numPoints*numLeftFields + numPoints*matrixRow + qp] *
        d_right[myMatrix*numPoints*numRightFields + qp*numRightFields + matrixCol];

    }
    d_out[myID]= temp;

  }

}

void
cudaDocontractFieldFieldScalar(double * h_out,
    double * h_inLeft,
    double * h_inRight,
    int numCells,
    int numLeftFields,
    int numRightFields,
    int numPoints,
    timespec * tic = NULL,
    timespec * toc = NULL) {

  double * d_right;
  double * d_left;
  double * d_out;

  cudaMalloc(&d_right, sizeof(double) * numCells * numPoints * numRightFields);
  cudaMalloc(&d_left, sizeof(double) * numCells * numPoints * numLeftFields);
  cudaMalloc(&d_out, sizeof(double) * numCells * numRightFields * numLeftFields);
  cudaMemset(d_out, 0, sizeof(double) * numCells * numRightFields * numLeftFields);

  cudaMemcpy(d_right, h_inRight,
      sizeof(double) * numCells * numPoints * numRightFields, cudaMemcpyHostToDevice);
  cudaMemcpy(d_left, h_inLeft,
      sizeof(double) * numCells * numPoints * numLeftFields, cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 gridSize((numCells * numLeftFields * numRightFields / 1024) + 1);

  if (tic != NULL)
    clock_gettime(CLOCK_MONOTONIC, tic);

  cudaDocontractFieldFieldScalar_kernel<<<gridSize, blockSize>>>(d_left,
      d_right, d_out, numCells, numLeftFields, numRightFields, numPoints);
  cudaDeviceSynchronize();

  if (toc != NULL)
    clock_gettime(CLOCK_MONOTONIC, toc);

  cudaFree(d_right);
  cudaFree(d_left);
  cudaFree(d_out);

  cudaMemcpy(h_out, d_out, sizeof(double) * numCells * numLeftFields * numRightFields, cudaMemcpyDeviceToHost);

}

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarFunctor {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numPoints;
	int _numLeftFields;
	int _numRightFields;

	contractFieldFieldScalarFunctor(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numLeftFields,
			int numRightFields,
			int numPoints) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numPoints(numPoints),
		_numLeftFields(numLeftFields),
		_numRightFields(numRightFields)
	{
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION
		void operator()(const unsigned int elementIndex) const {
			for (int lbf = 0; lbf < _numLeftFields; lbf++) {
				for (int rbf = 0; rbf < _numRightFields; rbf++) {
					double tmpVal = 0;
					for (int qp = 0; qp < _numPoints; qp++) {
						tmpVal += _leftFields(elementIndex, lbf, qp)*_rightFields(elementIndex, rbf, qp);
					} // P-loop
					_outputFields(elementIndex, lbf, rbf) = tmpVal;
				} // R-loop
			} // L-loop
		}
};

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarCudaFunctor {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numPoints;
	int _numLeftFields;
	int _numRightFields;

	contractFieldFieldScalarCudaFunctor(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numLeftFields,
			int numRightFields,
			int numPoints) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numPoints(numPoints),
		_numLeftFields(numLeftFields),
		_numRightFields(numRightFields)
	{
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION
	    void operator()(const unsigned int elementIndex) const {

		int myID = elementIndex;
		int myMatrix = myID / (_numLeftFields * _numRightFields);
		int matrixIndex = myID % (_numLeftFields * _numRightFields);

		int matrixRow = matrixIndex / _numRightFields;
		int matrixCol = matrixIndex % _numRightFields;

		double temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    temp += _leftFields(myMatrix, matrixRow, qp) * _rightFields(myMatrix, qp, matrixCol);
		}
		_outputFields(matrixIndex, matrixRow, matrixCol) = temp;
	    }
};


// Serial contractFieldFieldScalar.  Contracts arrays of doubles.
void contractFieldFieldScalarSerial(double * outputFields, // c, l, r
		double *             leftFields,  // c, l ,p
	  double *             rightFields,  // c, r, p
    int                  numCells,
    int                  numLeftFields,
    int                  numRightFields,
    int                  numPoints) {

  double tmpVal;
	for (int cl = 0; cl < numCells; cl++) {
		for (int lbf = 0; lbf < numLeftFields; lbf++) {
			for (int rbf = 0; rbf < numRightFields; rbf++) {
				tmpVal = 0;
				for (int qp = 0; qp < numPoints; qp++) {
					tmpVal += leftFields[cl * numLeftFields * numPoints + lbf * numPoints + qp]
                  * rightFields[cl * numPoints * numRightFields + rbf * numPoints + qp];
				} // P-loop
				outputFields[cl * numLeftFields * numRightFields + lbf * numRightFields + rbf] = tmpVal;
			} // R-loop
		} // L-loop
	} // C-loop
}

/*
 * Kokkos OpenMP contractFieldFieldScalar.
 *
 * Contracts two Kokkos OpenMP host views (two double *** tensors -> one double
 * *** tensor). Since
 *
 * Note that all input and output is in Kokkos host views --- the user is
 * responsible for getting the data in and out of them.
 */
template <class DeviceType, class input_view_t, class output_view_t, class input_host_t, class output_host_t>
void contractFieldFieldScalarKokkos(output_host_t &   outHost,
		const input_host_t &                      leftHost,
		const input_host_t &                      rightHost,
		output_view_t &                           outDevice,
		input_view_t &                            leftDevice,
		input_view_t &                            rightDevice,
		double *                                  time = NULL) {

	// get sizes
	int numCells        = leftHost.dimension(0);
	int numLeftFields   = leftHost.dimension(1);
	int numRightFields  = rightHost.dimension(1);
	int numPoints       = leftHost.dimension(2);
	
  // Deep copy Kokkos host views into device views
	Kokkos::deep_copy(leftDevice, leftHost);
	Kokkos::deep_copy(rightDevice, rightHost);
	Kokkos::deep_copy(outDevice, outHost);

	timespec tic;
	if(time != NULL)
		clock_gettime(CLOCK_MONOTONIC, &tic);

	contractFieldFieldScalarFunctor<DeviceType, input_view_t, input_view_t, output_view_t>
		kokkosFunctor(leftDevice, rightDevice, outDevice, numLeftFields,
		numRightFields, numPoints);

	Kokkos::parallel_for(numCells, kokkosFunctor);

	Kokkos::fence();

	timespec toc;
	if(time != NULL){
		clock_gettime(CLOCK_MONOTONIC, &toc);
		*time += getElapsedTime(tic, toc);
	}

	Kokkos::deep_copy(outHost, outDevice);
}

/*
 * Kokkos Cuda contractFieldFieldScalar.
 *
 * Contracts two Kokkos Cuda host views (two double *** tensors -> one double
 * *** tensor). Since
 *
 * Note that all input and output is in Kokkos host views --- the user is
 * responsible for getting the data in and out of them.
 */
template <class DeviceType, class input_view_t, class output_view_t, class input_host_t, class output_host_t>
void contractFieldFieldScalarKokkosCuda(output_host_t &   outHost,
		const input_host_t &                      leftHost,
		const input_host_t &                      rightHost,
		output_view_t &                           outDevice,
		input_view_t &                            leftDevice,
		input_view_t &                            rightDevice,
		double *                                  time = NULL) {

	// get sizes
	int numCells        = leftHost.dimension(0);
	int numLeftFields   = leftHost.dimension(1);
	int numRightFields  = rightHost.dimension(2);
	int numPoints       = leftHost.dimension(2);
	
  // Deep copy Kokkos host views into device views
	Kokkos::deep_copy(leftDevice, leftHost);
	Kokkos::deep_copy(rightDevice, rightHost);
	Kokkos::deep_copy(outDevice, outHost);

	timespec tic;
	if(time != NULL)
		clock_gettime(CLOCK_MONOTONIC, &tic);

	contractFieldFieldScalarCudaFunctor<DeviceType, input_view_t, input_view_t, output_view_t>
		kokkosCudaFunctor(leftDevice, rightDevice, outDevice, numLeftFields,
		numRightFields, numPoints);

	Kokkos::parallel_for(numCells * numLeftFields * numRightFields, kokkosCudaFunctor);

	Kokkos::fence();

	timespec toc;
	if(time != NULL){
		clock_gettime(CLOCK_MONOTONIC, &toc);
		*time += getElapsedTime(tic, toc);
	}

	Kokkos::deep_copy(outHost, outDevice);
}


int main(int argc, char* argv[]) {
	int c=10000, p=10, l=100, r=100;

  // Seet the random
  srand(time(NULL));

	// ===============================================================
	// ********************** < Kokkos setup> ************************
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

	// Doing all of this here might throw off the timing -- we're not counting the
	// cost of the copy into Kokkos or the deep copy from Kokkos host to Kokkos
	// device.

	Kokkos::initialize();

	// Kokkos Cuda views
	typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda> cuda_input_view_t;
	typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda> cuda_output_view_t;
	typedef typename cuda_input_view_t::HostMirror cuda_input_host_t;
	typedef typename cuda_output_view_t::HostMirror cuda_output_host_t;

	// Kokkos OpenMP views
	typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::OpenMP> omp_input_view_t;
	typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::OpenMP> omp_output_view_t;
	typedef typename omp_input_view_t::HostMirror omp_input_host_t;
	typedef typename omp_output_view_t::HostMirror omp_output_host_t;


	//Cuda and Serial arrays

  double * serialRight = new double [c * r * p];
  double * serialLeft = new double [c * l * p];

	double * cudaRight = new double[c * r * p];
	double * cudaLeft = new double[c * l * p];

  double * serialOut = new double[c * l * r];
	double * cudaOut = new double[c * l * r];

	// Make equivalent Kokkos views

	cuda_input_view_t cuda_kokkosLeft("left_input", c,l, p);
	cuda_input_view_t cuda_kokkosRight("right_input", c, p, r);
	cuda_output_view_t cuda_kokkosOut("output", c, l, r );

	omp_input_view_t omp_kokkosLeft("left_input", c, l, p);
	omp_input_view_t omp_kokkosRight("right_input",  c,r, p);
	omp_output_view_t omp_kokkosOut("output", c,l ,r);

	// And their host mirrors
	cuda_input_host_t cuda_hostLeft = Kokkos::create_mirror_view(cuda_kokkosLeft);
	cuda_input_host_t cuda_hostRight = Kokkos::create_mirror_view(cuda_kokkosRight);
	cuda_output_host_t cuda_hostOut = Kokkos::create_mirror_view(cuda_kokkosOut);

	omp_input_host_t omp_hostLeft = Kokkos::create_mirror_view(omp_kokkosLeft);
	omp_input_host_t omp_hostRight = Kokkos::create_mirror_view(omp_kokkosRight);
	omp_output_host_t omp_hostOut = Kokkos::create_mirror_view(omp_kokkosOut);


	// Make Kokkos host views and cuda
  double n;
	for (int cl = 0; cl < c; ++cl) {
		for (int qp = 0; qp < p; ++qp) {
			for(int rbf = 0; rbf < r; ++rbf) {
        n = random_double();
				cuda_hostRight(cl, qp, rbf) = n;
				omp_hostRight(cl,rbf,qp) = n;
        serialRight[cl * p * r + rbf * p + qp] = n;

				cudaRight[cl * p * r + r * qp + rbf] = n;
			}
			for(int lbf = 0; lbf < l; ++lbf) {
        n = random_double();
				cuda_hostLeft(cl, lbf, qp) = n;
				omp_hostLeft(cl,lbf, qp) = n;
				serialLeft[cl * p * l  + lbf * p + qp] = n;

				cudaLeft[cl * p * l + lbf * p + qp] = n;
			}
		}
	}



	// ===============================================================
	// ********************** </Kokkos setup> ************************
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

	std::cout << "trying serial" << std::endl;

	//Warmup
	contractFieldFieldScalarSerial(serialOut, serialLeft, serialRight, c, l, r, p);

	timespec tic;
	clock_gettime(CLOCK_MONOTONIC, &tic);

	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarSerial(serialOut, serialLeft, serialRight, c, l, r, p);
	}

	timespec toc;
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "serial elapsed time: " << elapsedTime_serial << " sec" << std::endl;

	printf("trying kokkos openmp\n");

	//Warmpup
	contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t,
		omp_output_view_t, omp_input_host_t, omp_output_host_t>
			(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
			 omp_kokkosLeft, omp_kokkosRight);
	clock_gettime(CLOCK_MONOTONIC, &tic);

	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t,
			omp_output_view_t, omp_input_host_t, omp_output_host_t>
				(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
				 omp_kokkosLeft, omp_kokkosRight);
	}

	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_kokkos_omp = getElapsedTime(tic, toc);


  //Disabling checking for now
  /*
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
			<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
  */

	std::cout << "kokkos omp speedup of " << elapsedTime_serial/elapsedTime_kokkos_omp << std::endl;


	printf("trying kokkos cuda\n");
 
	//Warmpup
	contractFieldFieldScalarKokkosCuda<Kokkos::Cuda, cuda_input_view_t,
		cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
			(cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
			 cuda_kokkosLeft, cuda_kokkosRight);

	clock_gettime(CLOCK_MONOTONIC, &tic);

	//repeat the calculation 5 times so we can average out some randomness
  double elapsedTime_kokkos_cuda_nocopy = 0;
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarKokkosCuda<Kokkos::Cuda, cuda_input_view_t,
			cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
				(cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
				 cuda_kokkosLeft, cuda_kokkosRight, &elapsedTime_kokkos_cuda_nocopy);
	}

	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_kokkos_cuda_with_copy = getElapsedTime(tic, toc);

// No checking right now
  /*
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
			<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
  */

	std::cout << "kokkos runtime of " << elapsedTime_kokkos_cuda_nocopy/5 << std::endl;

	Kokkos::finalize();

	std::cout << "trying cuda major" << std::endl;
	//Now try the cuda version, start with warmup
	cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p);

  double elapsedTime_cuda_nocopy = 0;
	for(int i = 0; i < 5; ++i){
		cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, &tic, &toc);
    elapsedTime_cuda_nocopy += getElapsedTime(tic, toc);
	}

  // Disabling checking for now 
#if 0
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
		<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
#endif

	std::cout << "cuda speedup of " << elapsedTime_serial/elapsedTime_cuda_nocopy << std::endl;


	return 0;
}
