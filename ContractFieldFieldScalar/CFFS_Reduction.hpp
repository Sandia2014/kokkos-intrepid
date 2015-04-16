


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Reduction_TeamFunctor {
	unsigned int _numCells;
	unsigned int _numLeftFields;
	unsigned int _numRightFields;
	unsigned int _numPoints;
	LeftInputViewType _leftView;
	RightInputViewType _rightView;
	OutputViewType _outputView;


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

			/* This requires the outputView to be all 0s beforehand */
			if (_numPoints <= threadsPerTeam/2) {	
				int myID = thread.league_rank()*(threadsPerTeam/_numPoints)+thread.team_rank()/_numPoints;
				int myMatrix = myID / (_numLeftFields * _numRightFields);
				int matrixIndex = myID - (myMatrix * (_numLeftFields * _numRightFields));
				int matrixRow = matrixIndex / _numRightFields;
				int matrixCol = matrixIndex - (matrixRow * _numRightFields);

				int pointIndex = thread.team_rank() % _numPoints;

				float mult = _leftView(myMatrix, matrixRow, pointIndex) 
					* _rightView(myMatrix, pointIndex, matrixCol);

				Kokkos::atomic_fetch_add(&_outputView(myMatrix, matrixRow, matrixCol), mult);
			}

			else {
				int myID =  thread.league_rank();
				int myMatrix = myID / (_numLeftFields * _numRightFields);
				int matrixIndex = myID - (myMatrix * (_numLeftFields * _numRightFields));

				int matrixRow = matrixIndex / _numRightFields;
				int matrixCol = matrixIndex - (matrixRow * _numRightFields);


				Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, thread.team_size()), 
						[&] (const unsigned int& i, float& sum) {
						int j = i;
						while (j < _numPoints) {
						sum += _leftView(myMatrix, matrixRow, j) 
						* _rightView(myMatrix, j, matrixCol);
						j += thread.team_size();
						}
						}, 
						sum);
				_outputView(myMatrix, matrixRow, matrixCol) = sum;
			}
		}
};


