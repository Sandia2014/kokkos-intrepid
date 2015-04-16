template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_AdaptiveSlicing_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  unsigned int tens1;
  unsigned int tens2;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_AdaptiveSlicing_TeamFunctor(unsigned int _numCells,
      unsigned int _numLeftFields,
      unsigned int _numRightFields,
      unsigned int _numPoints,
      unsigned int _tens1,
      unsigned int _tens2,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    tens1(_tens1),
    tens2(_tens2),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    const unsigned int blockSize = thread.team_size();
    const unsigned int contractionSize = numPoints * tens1 * tens2;
    const unsigned int threadRow = thread.team_rank() / numLeftFields;
    const unsigned int col = thread.team_rank() - (threadRow * numLeftFields);

    unsigned int currentBlock = thread.league_rank();
    const unsigned int numBlocks = thread.league_size();

    Kokkos::View<float*, Kokkos::MemoryUnmanaged> sliceStorage(thread.team_shmem(), numPoints*2);

      thread.team_barrier();
      const unsigned int cell = (currentBlock*2) / numLeftFields;
      const unsigned int row = (currentBlock*2) - cell * numLeftFields;

      if((cell < numberOfTensors) && ((row+threadRow) < numLeftFields)) {
        for (unsigned int p = col; p < contractionSize; p += (blockDim.x/2)) {
          const unsigned int pointIndex = p / (tens1*tens2);
          const unsigned int pointMod = p - (pointIndex * (tens1*tens2));
          const unsigned int tens1Index = pointMod / tens2;
          const unsigned int tens2Index = pointMod - (tens2 * tens1Index);
          sliceStorage(p + (threadRow*contractionSize)) =
                dev_tensorData_Left(cell, row+threadRow, pointIndex, tens1Index,tens2Index);
          }
        //dev_contractionResults[cell*numRightFields*numLeftFields + row*numRightFields + col] = -1;
        thread.team_barrier();
        float sum = 0;
        for (int p = 0; p < contractionSize; ++p) {
          const unsigned int pointIndex = p / (tens1*tens2);
          const unsigned int pointMod = p - (pointIndex * (tens1*tens2));
          const unsigned int tens1Index = pointMod / tens2;
          const unsigned int tens2Index = pointMod - (tens2 * tens1Index);

          sum += sliceStorage(p + (threadRow*contractionSize)) *
            dev_tensorData_Right(cell,pointIndex,tens1Index,tens2Index,col);
          }

          dev_tensorResults(cell,row+threadRow,col) = sum;
      }

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints * tens1 * tens2 * 2;
  }
};
