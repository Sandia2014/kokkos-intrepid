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

//Hacky random float... Right now it's just ints casted that way
float random_float() {
  return (float) rand()/RAND_MAX;
}

template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Reduction_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Reduction_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
      unsigned int _numRightFields, unsigned int _numPoints,
      LeftInputViewType _leftView, 
      RightInputViewType _rightView, 
      OutputViewType _outputView) :
    numCells(_numCells), 
    numLeftFields(_numLeftFields), 
    numRightFields(_numRightFields), 
    numPoints(_numPoints),
    leftView(_leftView), 
    rightView(_rightView), 
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    int myID =  thread.league_rank();
    int myMatrix = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);

    int matrixRow = matrixIndex / numRightFields;
    int matrixCol = matrixIndex % numRightFields;

    float sum = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, numPoints), 
        [&] (const unsigned int& i, float& sum) {
          sum += leftView(myMatrix, matrixRow, i) 
                 * rightView(myMatrix, i, matrixCol);
        }, 
        sum);
    outputView(myMatrix, matrixRow, matrixCol) = sum;
  }
};


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Slicing_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Slicing_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
      unsigned int _numRightFields, unsigned int _numPoints,
      LeftInputViewType _leftView, 
      RightInputViewType _rightView, 
      OutputViewType _outputView) :
    numCells(_numCells), 
    numLeftFields(_numLeftFields), 
    numRightFields(_numRightFields), 
    numPoints(_numPoints),
    leftView(_leftView), 
    rightView(_rightView), 
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    int l = thread.team_rank();
    int r = thread.league_rank() % numRightFields;
    int c = thread.league_rank() / numRightFields;

    Kokkos::View<float*, Kokkos::MemoryUnmanaged> shared_slice(thread.team_shmem(), numPoints);
    for (int p = thread.team_rank(); p < numPoints; p += thread.team_size()) {
      shared_slice(p) = leftView(c, l, p);
    }
    thread.team_barrier();

    float sum = 0;
    for (int p = 0; p < numPoints; ++p) {
      sum += shared_slice(p) * rightView(c, p, r);
    }
    outputView(c, l, r) = sum;

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints;
  }
};


void contractFieldFieldScalarSerial(float * outputFields, // c, l, r
    float *             leftFields,  // c, l ,p
    float *             rightFields,  // c, r, p
    int                  numCells,
    int                  numLeftFields,
    int                  numRightFields,
    int                  numPoints) {

  float tmpVal;
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

  int c = 1000, l = 16, r = 10, p = 10;

  srand(time(NULL));

  typedef Kokkos::View<float ***, Kokkos::LayoutRight, Kokkos::Cuda> 
    cuda_input_view;
  typedef Kokkos::View<float ***, Kokkos::LayoutRight, Kokkos::Cuda> 
    cuda_output_view;
  typedef typename cuda_input_view::HostMirror cuda_input_host;
  typedef typename cuda_output_view::HostMirror cuda_output_host;

  cuda_input_view leftCuda("left_input", c, l, p);
  cuda_input_view rightCuda("right_input", c, p, r);
  cuda_output_view outReductionCuda("output (reduction)", c, l, r);
  cuda_output_view outSlicingCuda("output (slicing)", c, l, r);

  cuda_input_host cuda_hostLeft = Kokkos::create_mirror_view(leftCuda);
  cuda_input_host cuda_hostRight = Kokkos::create_mirror_view(rightCuda);
  cuda_output_host cuda_reduction_hostOut = Kokkos::create_mirror_view(outReductionCuda);
  cuda_output_host cuda_slicing_hostOut = Kokkos::create_mirror_view(outSlicingCuda);

  float * leftField = new float[c*l*p];
  float * rightField = new float[c*r*p];
  float * outField = new float[c*l*r];

  float n;
  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < p; ++qp) {
      for(int rbf = 0; rbf < r; ++rbf) {
        n = random_float();
        cuda_hostRight(cl, qp, rbf) = n;
        rightField[cl * p * r + rbf * p + qp] = n;
      }
      for(int lbf = 0; lbf < l; ++lbf) {
        n = random_float();
        cuda_hostLeft(cl, lbf, qp) = n;
        leftField[cl * p * l  + lbf * p + qp] = n;
      }
    }
  }

  Kokkos::deep_copy(rightCuda, cuda_hostRight);
  Kokkos::deep_copy(leftCuda, cuda_hostLeft);

  contractFieldFieldScalarSerial(outField, leftField, rightField, c, l, r, p);


  // ---------------------------------------------------------------- //
  //                         REDUCTION                                //
  // ---------------------------------------------------------------- //
  const team_policy reduction_policy( c*l*r , p );
  
  CFFS_Reduction_TeamFunctor<cuda_input_view, cuda_input_view, cuda_output_view>
    kokkosReductionFunctor(c, l, r, p, leftCuda, rightCuda, outReductionCuda);

  Kokkos::parallel_for( reduction_policy, kokkosReductionFunctor );
  Kokkos::fence();

  Kokkos::deep_copy(cuda_reduction_hostOut, outReductionCuda);


  // ---------------------------------------------------------------- //
  //                         SLICING                                  //
  // ---------------------------------------------------------------- //
  const team_policy slicing_policy(c*r, l);

  CFFS_Slicing_TeamFunctor<cuda_input_view, cuda_input_view, cuda_output_view>
    kokkosSlicingFunctor(c, l, r, p, leftCuda, rightCuda, outSlicingCuda);

  Kokkos::parallel_for( slicing_policy, kokkosSlicingFunctor );
  Kokkos::fence();

  Kokkos::deep_copy(cuda_slicing_hostOut, outReductionCuda);



  // Need to verify that the solution is correct
  for (int cl = 0; cl < c; ++cl) {
    for (int lbf = 0; lbf < l; ++lbf) {
      for (int rbf = 0; rbf < r; ++rbf) {
        if (std::abs(outField[cl*l*r + lbf*r + rbf] - cuda_reduction_hostOut(cl, lbf, rbf))
             / std::abs(outField[cl*l*r + lbf*r + rbf]) >= 1e-4 )
        {
          Kokkos::finalize();
          fprintf(stderr, "Calculation error in reduction.");
          exit(1);
        }
        if (std::abs(outField[cl*l*r + lbf*r + rbf] - cuda_slicing_hostOut(cl, lbf, rbf))
             / std::abs(outField[cl*l*r + lbf*r + rbf]) >= 1e-4 )
        {
          Kokkos::finalize();
          fprintf(stderr, "Calculation error in reduction.");
          exit(1);
        }

      }
    }
  }

  printf("We win");
  Kokkos::finalize();
}

