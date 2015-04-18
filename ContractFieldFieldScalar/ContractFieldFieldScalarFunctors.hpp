#include <Kokkos_Core.hpp>

template <class DeviceType, class KokkosJunkVector>
struct KokkosFunctor_ClearCache {

  typedef size_t     value_type;
  typedef DeviceType device_type;

  KokkosJunkVector _junkDataToClearTheCache;

  KokkosFunctor_ClearCache(KokkosJunkVector dev_junkDataToClearTheCache) :
    _junkDataToClearTheCache(dev_junkDataToClearTheCache) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int index,
                  value_type & junkDataCounter) const {
    junkDataCounter += _junkDataToClearTheCache(index);
  }

private:
  KokkosFunctor_ClearCache();

};

template <class DeviceType, class KokkosContractionData,
          class KokkosContractionResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _contractionSize;
  KokkosContractionData _data_A;
  KokkosContractionData _data_B;
  KokkosContractionResults _results;

  KokkosFunctor_Independent(const unsigned int contractionSize,
                            KokkosContractionData data_A,
                            KokkosContractionData data_B,
                            KokkosContractionResults results) :
    _contractionSize(contractionSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int contractionIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _contractionSize;
         ++entryIndex) {
      sum +=
        _data_A(contractionIndex, entryIndex) *
        _data_B(contractionIndex, entryIndex);
    }
    _results(contractionIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};



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
