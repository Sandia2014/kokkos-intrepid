// Creation of FRED team Reduciton for CFFT

#include <Kokkos_Core.hpp>

typedef Kokkos::DefaultExecutionSpace Device;
typedef Kokkos::HostSpace::execution_space Host;
typedef Kokkos::TeamPolicy<Device> team_policy;
typedef team_policy::member_type team_member;

template<class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFT_Fred_Reduction_TeamFunctor {
	/* This function does a team reduction by making a team for every output
	 * element. The number of threads per team is then the smaller of TEAM_SIZE
	 * or _numPoints * _dim1Tens * _dim2Tens. Then each thread does an equal
	 * amount of work and reduces to its correct output. This is the "FRED" type
	 * of reduction, as in the number of threads per output is not equal to
	 * _dim2Tens or _dim1Tens * _dim2Tens, this means a thread is not
	 * responsible for doing numPoints multiplies then finishing, it can do more
	 * or less than that.
	 */
	
	unsigned int _numCells;
	unsigned int _numLeftFields;
	unsigned int _numRightFields;
	unsigned int _numPoints;
	unsigned int _dim1Tens;
	unsigned int _dim2Tens;
	LeftInputViewType _leftView;
	RightInputViewType _rightView;
	OutputViewType _outputView;


	CFFT_Fred_Reduction_TeamFunctor(unsigned int numCells,
									unsigned int numLeftFields,
									unsigned int numRightFields,
									unsigned int numPoints,
									unsigned int dim1Tens,
									unsigned int dim2Tens,
									LeftInputViewType leftView,
									RightInputViewType rightView,
									OutputViewType outputView) :
									_numCells(numCells),
									_numLeftFields(numLeftFields),
									_numRightFields(numRightFields),
									_numPoints(numPoints),
									_dim1Tens(dim1Tens),
									_dim2Tens(dim2Tens),
									_leftView(leftView),
									_rightView(rightView),
									_outputView(outputView) {
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION 
	void operator() (const team_member& thread) const {
		float sum = 0;
	//	float threadSum = 0;
		// Getting information about thread and saving it in a variable
		const unsigned int teamSize = thread.team_size();
		const unsigned int threadRank = thread.team_rank();
		const unsigned int teamID = thread.league_rank();

		// Calculating this thread's output index
		const unsigned int myMatrix = teamID / (_numLeftFields * _numRightFields);
		const unsigned int matrixIndex = teamID % 
											(_numLeftFields * _numRightFields);
		const unsigned int matrixRow = matrixIndex / _numRightFields;
		const unsigned int matrixCol = matrixIndex % _numRightFields;
		
		const unsigned int reductionSize = _numPoints * _dim1Tens * _dim2Tens;
		
		/*
		for (int index = teamID; index < reductionSize; index += teamSize) {
			const unsigned int qp = index / (_dim1Tens * _dim2Tens);
			const unsigned int dim1 = (index % (_dim1Tens * _dim2Tens)) / _dim2Tens;
			const unsigned int dim2 = index % _dim2Tens;

			threadSum += _leftView(myMatrix, matrixRow, qp, dim1, dim2) *
						 _rightView(myMatrix, matrixCol, qp, dim1, dim2);
		}
		*/
		
		const unsigned int dimSize = _dim1Tens * _dim2Tens;
		Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, reductionSize),
			[&] (const unsigned int& i, float& localSum) {
			int qp = i / dimSize;
			int dim1 = (i % dimSize) / _dim2Tens;
			int dim2 = i % _dim2Tens;
			localSum += _leftView(myMatrix, matrixRow, qp, dim1, dim2) *
				_rightView(myMatrix, matrixCol, qp, dim1, dim2);
			}, sum);
		/*
		Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, teamSize),
			[&] (const unsigned int& i, float& localSum) {
			sum += threadSum;
			}, sum);
		*/
		_outputView(myMatrix, matrixRow, matrixCol) = sum;
	}
	private:
		CFFT_Fred_Reduction_TeamFunctor();
};

