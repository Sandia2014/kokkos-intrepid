/* Created by: Alex Gruver and Tyler Marklyn
 *
 * This implements the simple flat parallel scheme in Kokkos
 */

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarKokkosCudaFunctor {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numCells;
	int _numPoints;
	int _numLeftFields;
	int _numRightFields;

	contractFieldFieldScalarKokkosCudaFunctor(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numCells,
			int numLeftFields,
			int numRightFields,
			int numPoints) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numCells(numCells),
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

		float temp = 0;
		for (int qp = 0; qp < _numPoints; qp++) {
		    temp += _leftFields(myMatrix, qp, matrixRow) * _rightFields(myMatrix, qp, matrixCol);
		}
		_outputFields(myMatrix, matrixRow, matrixCol) = temp;
	    }
};


