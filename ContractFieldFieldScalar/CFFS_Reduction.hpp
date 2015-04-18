
#ifndef KOKKOS_INCLUDES
#define KOKKOS_INCLUDES
#include <Kokkos_Core.hpp>
typedef Kokkos::DefaultExecutionSpace Device;
typedef Kokkos::HostSpace::execution_space Host;
typedef Kokkos::TeamPolicy<Device> team_policy;
typedef team_policy::member_type team_member;
#endif


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Reduction_TeamFunctor {
	unsigned int _numCells;
	unsigned int _numLeftFields;
	unsigned int _numRightFields;
	unsigned int _numPoints;
	LeftInputViewType _leftView;
	RightInputViewType _rightView;
	OutputViewType _outputView;

	// It is expected that _leftView is in the order (c, l, p) and _rightView
	// is in the order of (c, p, r)
	CFFS_Reduction_TeamFunctor(unsigned int numCells, unsigned int numLeftFields,
			unsigned int numRightFields, unsigned int numPoints,
			LeftInputViewType leftView, 
			RightInputViewType rightView, 
			OutputViewType outputView) :
		_numCells(numCells), 
		_numLeftFields(numLeftFields), 
		_numRightFields(numRightFields), 
		_numPoints(numPoints),
		_leftView(leftView), 
		_rightView(rightView), 
		_outputView(outputView) {
			// Nothing to do
		}

	KOKKOS_INLINE_FUNCTION
		void operator() (const team_member & thread) const {

			float sum = 0;
			
			const unsigned int threadsPerTeam = thread.team_size();

			// This is the reduction special case which has the purpose
			// of teams doing more than one output element when there
			// are more threads than multiplies taking place.
			// The only time this is used is when _numPoints (which is
			// the number of multiplies required to calculate an output
			// is less than or equal to 16).
			//
			// NOTE: This requires the outputView to be all 0s beforehand
			// because of the atomic_fetch_add. This could be changed
			// so that all the threads atomic_fetch_add to a shared
			// memory location then by setting the output location
			// to the shared memory location.
			if (_numPoints <= threadsPerTeam/2) {	
				int myID = thread.league_rank()*(threadsPerTeam/_numPoints)+thread.team_rank()/_numPoints;
				
				// myMatrix tells the thread which cell its output location is
				// in
				int myMatrix = myID / (_numLeftFields * _numRightFields);

				// matrixIndex is the thread's in the output matrix. This is
				// also calculated as:
				// matrixIndex = myID % (_numLeftFields * _numRightFields)
				// But this is a more optimized way of calculating it
				int matrixIndex = myID - (myMatrix * (_numLeftFields * _numRightFields));

				// The thread's row in the output matrix
				int matrixRow = matrixIndex / _numRightFields;

				// The thread's column in the ouput matrix. Also calculated by:
				// matrixCol = matrixIndex % _numRightFields
				// but this way is more optimized.
				int matrixCol = matrixIndex - (matrixRow * _numRightFields);

				int pointIndex = thread.team_rank() % _numPoints;

				// Do my one multiplication
				float mult = _leftView(myMatrix, matrixRow, pointIndex) 
					* _rightView(myMatrix, pointIndex, matrixCol);

				// Update the correct reduction location
				Kokkos::atomic_fetch_add(&_outputView(myMatrix, matrixRow, matrixCol), mult);
			}

			else {
				// The ID of the output location I am helping calculate
				int myID =  thread.league_rank();

				// The cell of the output location I am helping to calculate
				int myMatrix = myID / (_numLeftFields * _numRightFields);

				// The threads index in the output matrix. Also calculated as:
				// matrixIndex = myID % (_numLeftFields * _numRightFields)
				// but this is more optimized.
				int matrixIndex = myID - (myMatrix * (_numLeftFields * _numRightFields));

				// The row index of the output location
				int matrixRow = matrixIndex / _numRightFields;

				// The column index of the ouput location. Also calculated as:
				// matrixCol = matrixIndex % _numRightFields
				int matrixCol = matrixIndex - (matrixRow * _numRightFields);

				// Doing the parallel reduce over all _numPoints and putting the
				// solution in the variable sum
				Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, _numPoints), 
						[&] (const unsigned int& i, float& localSum) {
						localSum += _leftView(myMatrix, matrixRow, i) 
						* _rightView(myMatrix, i, matrixCol);
						}, 
						sum);
				
				// Have the 0th thread put the solution into the output
				if (thread.team_rank() == 0) {
					_outputView(myMatrix, matrixRow, matrixCol) = sum;
				}
			}
		}
};


