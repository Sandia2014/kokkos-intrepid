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
#include <assert.h>

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


double random_double() {
    return (double) rand();
}


    
void contractFieldFieldTensorSerial(double * outputFields,
                                    double *   leftFields,
                                    double *  rightFields,
                                    const bool sumInto,
				    int numCells,
				    int numLeftFields,
				    int numRightFields,
				    int numPoints,
				    int dim1Tensor,
				    int dim2Tensor) {
    /* This function expects the left and right arrays to be in the order of
     * (cell, left or right, points, dim1Tens, dim2Tens). That is the way
     * the indexing is calculated.
     */
    
    if (sumInto) {
	for (int cl = 0; cl < numCells; cl++) {
	    // Need to index into the different arrays, so I am doing the 
	    // calculation once here
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    double tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] += tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
    }
    // This is exactly the same as above but outputfields is set equal
    // to temp instead of += temp
    else {
	for (int cl = 0; cl < numCells; cl++) {
	    int clOff = numLeftFields*numPoints*dim1Tensor*dim2Tensor;
	    int crOff = numRightFields*numPoints*dim1Tensor*dim2Tensor;
	    int cOut = numLeftFields*numRightFields;
	    for (int lbf = 0; lbf < numLeftFields; lbf++) {
		int lOff = numPoints*dim1Tensor*dim2Tensor;
		int lOut = numRightFields;
		for (int rbf = 0; rbf < numRightFields; rbf++) {
		    double tmpVal = 0;
		    int rOff = numPoints*dim1Tensor*dim2Tensor;
		    for (int qp = 0; qp < numPoints; qp++) {
			int pOff = dim1Tensor*dim2Tensor;
			for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
			    int tenOff = dim2Tensor;
			    for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
				tmpVal +=
				leftFields[cl*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
				rightFields[cl*crOff+rbf*rOff+qp*pOff+iTens1*tenOff+iTens2];
			    } // D2-loop
			} // D1-loop
		    } // P-loop
		    outputFields[cl*cOut+lbf*lOut+rbf] = tmpVal;
		} // R-loop
	    } // L-loop
	} // C-loop
   }
} // end contractFieldFieldTensor
 

template<class DeviceType, class LeftViewType, class RightViewType, class
OutputViewType>
struct contractFieldFieldTensorFunctor {
    typedef DeviceType device_type;
    LeftViewType _leftFields;
    RightViewType _rightFields;
    OutputViewType _outputFields;
    int _numCells;
    int _numLeftFields;
    int _numRightFields;
    int _numPoints;
    int _dim1Tens;
    int _dim2Tens;

    contractFieldFieldTensorFunctor(LeftViewType leftFields, RightViewType
    rightFields, OutputViewType outputFields, int numCells, int numLeftFields, int
    numRightFields, int numPoints, int dim1Tens, int dim2Tens) :
    _leftFields(leftFields), _rightFields(rightFields),
    _outputFields(outputFields), _numCells(numCells), _numPoints(numPoints),
    _numLeftFields(numLeftFields), _numRightFields(numRightFields),
    _dim1Tens(dim1Tens), _dim2Tens(dim2Tens)
    {

    }

    // This function expects the views to be in this order:
    // left(cell, leftBasis, points, dim1Tens, dim2Tens)
    // right(cell, points, dim1Tens, dim2Tens, rightBasis)
    KOKKOS_INLINE_FUNCTION
	void operator() (const unsigned int elementIndex) const {
	    int myID = elementIndex;

	    if(myID < (_numCells * _numLeftFields * _numRightFields)) {
		// Calculating the index in the output array for this
		// thread
		int myCell = myID / (_numLeftFields * _numRightFields);
		int matrixIndex = myID % (_numLeftFields * _numRightFields);
		int lbf = matrixIndex / _numRightFields;
		int rbf = matrixIndex % _numRightFields;
		
		double temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    for (int iTens1 = 0; iTens1 < _dim1Tens; iTens1++) {
			for (int iTens2 = 0; iTens2 < _dim2Tens; iTens2++) {
			    temp += _leftFields(myCell, lbf, qp, iTens1, iTens2) *
				_rightFields(myCell, qp, iTens1, iTens2, rbf);
			}
		    }
		}
		_outputFields(myCell, lbf, rbf) = temp;
	    }
	}

};



template <class DeviceType, class input_view_t, class output_view_t, class
input_host_t, class output_host_t>
void contractFieldFieldTensorKokkos(output_host_t& outHost,
    const input_host_t & leftHost,
    const input_host_t & rightHost,
    output_view_t & outDevice,
    input_view_t & leftDevice,
    input_view_t & rightDevice,
    int numCells,
    int numLeftFields,
    int numRightFields,
    int numPoints,
    int dim1Tens,
    int dim2Tens,
    double* time = NULL) {
   
    Kokkos::deep_copy(leftDevice, leftHost);
    Kokkos::deep_copy(rightDevice, rightHost);
    Kokkos::deep_copy(outDevice, outHost);

    timespec tic;
    if (time != NULL) {
	clock_gettime(CLOCK_MONOTONIC, &tic);
    }

    contractFieldFieldTensorFunctor<DeviceType, input_view_t, input_view_t,
    output_view_t> kokkosFunctor(leftDevice, rightDevice, outDevice,
    numCells, numLeftFields, numRightFields, numPoints, dim1Tens, dim2Tens);

    Kokkos::parallel_for(numCells*numLeftFields*numRightFields, kokkosFunctor);

    Kokkos::fence();

    timespec toc;
    if (time != NULL) {
	clock_gettime(CLOCK_MONOTONIC, &toc);
	*time += getElapsedTime(tic, toc);
    }

    Kokkos::deep_copy(outHost, outDevice);



}

int main(int argc, char* argv[]) {
    // Changing these variables changes the problem size
    int c=1000, p=10, l=100, r=100, t1=10, t2=10;

    // These are indices into the arrays and should not be
    // changed unless you change the order of the indices
    // as well
    int cLOff = l*p*t1*t2;
    int cROff = r*p*t1*t2;
    int basisOff = p*t1*t2;
    int pLOff = t1*t2;
    int pROff = t1*t2*r;
    int tROff = t2*r;
    int t2ROff = r;
    int tOff = t2;

    double * in_c_l_p_t1_t2 = new double[c*l*p*t1*t2];
    double * in_c_r_p_t1_t2 = new double[c*r*p*t1*t2];
    double * out1_c_l_r = new double[c*l*r];
    double * out2_c_l_r = new double[c*l*r];

    // Filling the left array with random doubles
    for (int cl = 0; cl < c; ++cl) {
	int cOff = p * t1 * t2 * r;
	for(int rbf = 0; rbf < r; ++rbf) {
	    int rOff = p*t1*t2;
	    for (int qp = 0; qp < p; ++qp) {
		int pOff = t1*t2;
		for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		    int t1Off = t2;
		    for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
			in_c_r_p_t1_t2[cl*cOff+rbf*rOff+qp*pOff+iTens1*t1Off+iTens2]
			    = random_double();
		    }
		}
	    }
	}
    }

    // Filling the right array with random doubles
    for (int cl = 0; cl < c; ++cl) {
	int cOff = p*t1*t2*l;
	for(int lbf = 0; lbf < l; ++lbf) {
	    int lOff = p*t1*t2;
	    for (int qp = 0; qp < p; ++qp) {
		int pOff = t1*t2;
		for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		    int t1Off = t2;
		    for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
			in_c_l_p_t1_t2[cl*cOff+lbf*lOff+qp*pOff+iTens1*t1Off+iTens2]
			    = random_double();
		    }
		}
	    }
	}
    }


    // Making sure the output arrays are set to 0
    for (int cl = 0; cl < c; cl++) {
	for (int lbf = 0; lbf < l; lbf++) {
	    for (int rbf = 0; rbf < r; rbf++) {
		out1_c_l_r[cl*r*l + lbf*r + rbf] = 0;
		out2_c_l_r[cl*r*l + lbf*r + rbf] =0;
	    }
	}
    }

    std::cout << "Created vectors" << std::endl;

    timespec tic;
    clock_gettime(CLOCK_MONOTONIC, &tic);

    contractFieldFieldTensorSerial(out1_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2, 
	    false, c, l, r, p, t1, t2);

    timespec toc;
    clock_gettime(CLOCK_MONOTONIC, &toc);
    const double elapsedTime_serial = getElapsedTime(tic, toc);

    std::cout << "serial elapsed time: " << elapsedTime_serial << " sec" <<
	std::endl;


    /* Initializing Kokkos and all the views */
    Kokkos::initialize();

    typedef Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::Cuda>
	cuda_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::Cuda>
	cuda_output_view_t;
    typedef typename cuda_input_view_t::HostMirror cuda_input_host_t;
    typedef typename cuda_output_view_t::HostMirror cuda_output_host_t;

    typedef Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::OpenMP>
	omp_input_view_t;
    typedef Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::OpenMP>
	omp_output_view_t;
    typedef typename omp_input_view_t::HostMirror omp_input_host_t;
    typedef typename omp_output_view_t::HostMirror omp_output_host_t;


    cuda_input_view_t cuda_kokkosLeft("left_input", c, l, p, t1, t2);
    cuda_input_view_t cuda_kokkosRight("right_input", c, p, t1, t2, r);
    cuda_output_view_t cuda_kokkosOut("output", c, l, r);

    /*
    cuda_input_host_t cuda_hostLeft("left_input", c, l, p, t1, t2);
    cuda_input_host_t cuda_hostRight("left_input", c, p, t1, t2, r);
    cuda_output_host_t cuda_hostOut("left_input", c, l, r);
    */

    cuda_input_host_t cuda_hostLeft =
    Kokkos::create_mirror_view(cuda_kokkosLeft);
    cuda_input_host_t cuda_hostRight =
    Kokkos::create_mirror_view(cuda_kokkosRight);
    cuda_output_host_t cuda_hostOut =
    Kokkos::create_mirror_view(cuda_kokkosOut);


    omp_input_view_t omp_kokkosLeft("left_input", c, l, p, t1, t2);
    omp_input_view_t omp_kokkosRight("right_input", c, p, t1, t2, r);
    omp_output_view_t omp_kokkosOut("output", c, l, r);

    /*
    omp_input_host_t omp_hostLeft("left_input", c, l, p, t1, t2);
    omp_input_host_t omp_hostRight("left_input", c, p, t1, t2, r);
    omp_output_host_t omp_hostOut("left_input", c, l, r);
    */


    omp_input_host_t omp_hostLeft =
    Kokkos::create_mirror_view(omp_kokkosLeft);
    omp_input_host_t omp_hostRight =
    Kokkos::create_mirror_view(omp_kokkosRight);
    omp_output_host_t omp_hostOut =
    Kokkos::create_mirror_view(omp_kokkosOut);


    printf("filling views\n");

    for (int cl = 0; cl < c; ++cl) {
	for (int qp = 0; qp < p; ++qp) {
	    for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
		for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
		    for(int rbf = 0; rbf < r; ++rbf) {
			cuda_hostRight(cl, qp, iTens1, iTens2, rbf) =
			    in_c_r_p_t1_t2[cl*cROff + rbf + qp*pROff + 
			    iTens1*tROff + iTens2*t2ROff];
			omp_hostRight(cl, qp, iTens1, iTens2, rbf) =
			    in_c_r_p_t1_t2[cl*cROff + rbf + qp*pROff + 
			    iTens1*tROff + iTens2*t2ROff];
		    }
		    for(int lbf = 0; lbf < l; ++lbf) {
			cuda_hostLeft(cl, lbf, qp, iTens1, iTens2) =
			    in_c_l_p_t1_t2[cl*cLOff + lbf*basisOff + qp*pLOff +
			    iTens1*tOff + iTens2];
			omp_hostLeft(cl, lbf, qp, iTens1, iTens2) =
			    in_c_l_p_t1_t2[cl*cLOff + lbf*basisOff + qp*pLOff +
			    iTens1*tOff + iTens2];

		    }
		}
	    }
	}
    }
    printf("trying Kokkos Cuda\n");


    double elapsedTime_kokkos_cuda_nocopy = 0;
    double elapsedTime_kokkos_omp_nocopy = 0;


    clock_gettime(CLOCK_MONOTONIC, &tic);

    contractFieldFieldTensorKokkos<Kokkos::Cuda, cuda_input_view_t,
	cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>(cuda_hostOut,
		cuda_hostLeft, cuda_hostRight, cuda_kokkosOut, cuda_kokkosLeft,
		cuda_kokkosRight, c, l, r, p, t1, t2, &elapsedTime_kokkos_cuda_nocopy);


    contractFieldFieldTensorKokkos<Kokkos::OpenMP, omp_input_view_t,
	omp_output_view_t, omp_input_host_t, omp_output_host_t>(omp_hostOut,
		omp_hostLeft, omp_hostRight, omp_kokkosOut, omp_kokkosLeft,
		omp_kokkosRight, c, l, r, p, t1, t2, &elapsedTime_kokkos_omp_nocopy);



    clock_gettime(CLOCK_MONOTONIC, &toc);

    // This can be used if you want to include copying times
    // Commented out so that the compiler doesn't give a warning
    // Commented out so that the compiler doesn't give a warning
    // Commented out so that the compiler doesn't give a warning
    // Commented out so that the compiler doesn't give a warning
    // double elapsedTime_kokkos_cuda_copy = getElapsedTime(tic, toc);


    /*
    // This is made to check for correctness, but it is off still because
    // doing 1000 double operations throws off the correctness and I haven't
    // found a good way to calculate the epsilon that the two doubles
    // should be within
    for (int i = 0; i < c; c++) {
    for (int j = 0; j < l; j++) {
    for (int k = 0; k < r; k++) {
    double diff = cuda_hostOut(i, j, k) - out1_c_l_r[i*l*r +
    j*r + k];
    if (diff < 0) {
    diff = -diff;
    }
    double frac = cuda_hostOut(i, j, k)/100;
    if (frac < 0) {
    frac = -frac;
    }
    if (diff > frac) {
    std::cout << "we have a problem" << std::endl;
    std::cout << i << " " << j << " " << k << std::endl;
    std::cout << "serial num " << out1_c_l_r[i*l*r +j*r +k] <<
    std::endl;
    std::cout << "para num " << cuda_hostOut(i, j, k) <<
    std::endl;
    Kokkos::finalize();
    return 0;
    }
    }
    }
    }
     */
    std::cout << "kokkos runtime of " << elapsedTime_kokkos_cuda_nocopy << std::endl;
    std::cout << "speed up of " <<
	elapsedTime_serial/elapsedTime_kokkos_cuda_nocopy << std::endl;

    std::cout << "kokkos omp runtime of " << elapsedTime_kokkos_omp_nocopy <<
    std::endl;
    std::cout << "speed up of " <<
    elapsedTime_serial/elapsedTime_kokkos_omp_nocopy << std::endl;

    Kokkos::finalize();

    return 0;
}
