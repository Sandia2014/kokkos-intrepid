/* This file is the attempt to get teams to be working for 
 * ContractFieldFeildScalar. 
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>

typedef Kokkos::DefaultExecutionSpace       Device;
typedef Kokkos::HostSpace::execution_space  Host;

typedef Kokkos::TeamPolicy< Device >      team_policy;
typedef team_policy::member_type team_member;

static const int TEAM_SIZE = 16;


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
  return (double) rand()/RAND_MAX;
}

class CFFS_TeamFunctor {
    unsigned int numCells;
    unsigned int numLeftFields;
    unsigned int numRightFields;
    unsigned int numPoints;
    Kokkos::View<float ***> leftView;
    Kokkos::View<float ***> rightView;
    Kokkos::View<float ***> outputView;

   
    typedef Device::scratch_memory_space SharedSpace;
    typedef Kokkos::View<float ***, SharedSpace, Kokkos::MemoryUnmanaged> 
	sharedLeft;
    
    typedef Kokkos::View<float ***, SharedSpace, Kokkos::MemoryUnmanaged> 
	sharedRight;


    CFFS_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
		    unsigned int _numRightFields, unsigned int _numPoints,
		    Kokkos::View<float ***> _leftView, Kokkos::View<float***> 
		    _rightView, Kokkos::View<float ***> _outputView) :
		    numCells(_numCells), numLeftFields(_numLeftFields), 
		    numRightFields(_numRightFields), numPoints(_numPoints),
		    leftView(_leftView), rightView(_rightView), 
		    outputView(_outputView) {
	// Nothing to do
    }

    void operator() (const int elementIndex) const {
	int myID =  elementIndex;
	int myMatrix = myID / (numLeftFields * numRightFields);
	int matrixIndex = myID % (numLeftFields * numRightFields);

	int matrixRow = matrixIndex / numRightFields;
	int matrixCol = matrixIndex % numRightFields;

	double temp = 0;
	for (int qp = 0; qp < numPoints; qp++) {
	    temp += leftView(myMatrix, matrixRow, qp) * rightView(myMatrix, qp, matrixCol);
	}
	outputView(myMatrix, matrixRow, matrixCol) = temp;
    }


};



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
		    tmpVal += leftFields[cl * numLeftFields * numPoints + 
			    lbf * numPoints + qp] *
			    rightFields[cl * numPoints * numRightFields + 
			    rbf * numPoints + qp];
		} // P-loop
		outputFields[cl * numLeftFields * numRightFields + 
		    lbf * numRightFields + rbf] = tmpVal;
	    } // R-loop
	} // L-loop
    } // C-loop
}




int main(int argc, char* args[]) {
    Kokkos::initialize(argc, args);
    
    int c = 10000, l = 100, r = 100, p = 10;

    srand(time(NULL));

    typedef Kokkos::View<float ***, Kokkos::LayoutRight, Kokkos::Cuda> 
	cuda_input_view;
    typedef Kokkos::View<float ***, Kokkos::LayoutRight, Kokkos::Cuda> 
	cuda_output_view;
    typedef typename cuda_input_view::HostMirror cuda_input_host;
    typedef typename cuda_output_view::HostMirror cuda_output_host;

    cuda_input_view leftCuda("left_input", c, l, p);
    cuda_input_view rightCuda("right_input", c, p, r);
    cuda_output_view outCuda("output", c, l, r);

    cuda_input_host cuda_hostLeft = Kokkos::create_mirror_view(leftCuda);
    cuda_input_host cuda_hostRight = Kokkos::create_mirror_view(rightCuda);
    cuda_output_host cuda_hostOut = Kokkos::create_mirror_view(outCuda);

    double * leftField = new double[c*l*p];
    double * rightField = new double[c*r*p];
    double * outField = new double[c*l*r];

    double n;
    for (int cl = 0; cl < c; ++cl) {
	for (int qp = 0; qp < p; ++qp) {
	    for(int rbf = 0; rbf < r; ++rbf) {
		n = random_double();
		rightCuda(cl, qp, rbf) = n;
		rightField[cl * p * r + rbf * p + qp] = n;
	    }
	    for(int lbf = 0; lbf < l; ++lbf) {
		n = random_double();
		leftCuda(cl, lbf, qp) = n;
		leftField[cl * p * l  + lbf * p + qp] = n;
	    }
	}
    }

    contractFieldFieldScalarSerial(outField, leftField, rightField, c, l, r, p);


    // Need to make the functor and run it
    

    
    // Need to verify that the solution is correct
    
    Kokkos::finalize();
}

