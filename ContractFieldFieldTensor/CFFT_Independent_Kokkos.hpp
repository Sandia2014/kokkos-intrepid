/*
 * Created by: Alex Gruver
 *
 * This function implements the simple flat parallel scheme for CFFT using Kokkos
 */

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
        // Calculating the index in the output array for this thread
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

