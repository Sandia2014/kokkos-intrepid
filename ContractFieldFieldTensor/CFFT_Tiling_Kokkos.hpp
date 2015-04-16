// An implementation of CFFT that uses a modified matrix tiling algorithm.
//
// It effectively treats all of the contraction dimensions as a single dimension
// for the purposes of tiling. (Compare to CFFS tiling)
//

template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Tiling_TeamFunctor_1D {

  const unsigned int _numCells;
  const unsigned int _numLeftFields;
  const unsigned int _numRightFields;
  const unsigned int _numPoints;
  const unsigned int _dimTens1;
  const unsigned int _dimTens2;

  LeftInputViewType _leftView;
  RightInputViewType _rightView;
  OutputViewType _outputView;

  const unsigned int _tile_size;

  CFFS_Tiling_TeamFunctor_1D(const unsigned int numCells,
      const unsigned int numLeftFields,
      const unsigned int numRightFields,
      const unsigned int numPoints,
      const unsigned int dimTens1,
      const unsigned int dimTens2,
      LeftInputViewType leftView,
      RightInputViewType rightView,
      OutputViewType outputView,
      const unsigned int tile_size) :
    _numCells(numCells),
    _numLeftFields(numLeftFields),
    _numRightFields(numRightFields),
    _numPoints(numPoints),
    _dimTens1(dimTens1),
    _dimTens2(dimTens2),
    _leftView(leftView),
    _rightView(rightView),
    _outputView(outputView),
    _tile_size(tile_size)
  {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    //NOTE: THIS WHOLE THING WORKS ASSUMING NUMLEFTFIELDS==NUMRIGHTFIELDS
    const unsigned int numBasis = _numLeftFields;

    // Here we pretend that all three contraction dimensions are a single dimension: contractionSize
    const unsigned int contractionSize = _dimTens1 * _dimTens2 * _numPoints;

    const unsigned int numberOfPointTiles = ((contractionSize-1) / _tile_size) + 1;
    const unsigned int numberOfBasisTiles = ((numBasis-1) / _tile_size) + 1;

    const unsigned int numberOfTiles = _numCells * numberOfBasisTiles * numberOfBasisTiles;
    const unsigned int subRow = thread.team_rank() / _tile_size;
    const unsigned int subCol = thread.team_rank() - subRow * _tile_size;

    unsigned int resultTileIndex = thread.league_rank();
    Kokkos::View<float*, Kokkos::MemoryUnmanaged> tileStorage(thread.team_shmem(), 2 * _tile_size * _tile_size);

    while (resultTileIndex < numberOfTiles) {

      const unsigned int resultMatrix = resultTileIndex / (numberOfBasisTiles * numberOfBasisTiles);
      const unsigned int resultSubmatrixIndex = resultTileIndex - (resultMatrix * numberOfBasisTiles * numberOfBasisTiles); // (mod)

      // calculate result tile indices
      const unsigned int resultTileRow = resultSubmatrixIndex / numberOfBasisTiles;
      const unsigned int resultTileCol = resultSubmatrixIndex - resultTileRow * numberOfBasisTiles; // (mod)

      // calculate this threads actual output index
      const unsigned int row = resultTileRow * _tile_size + subRow;
      const unsigned int col = resultTileCol * _tile_size + subCol;
      float sum = 0;

      // for tileNumber in 0...numberOfTilesPerSide
      for (unsigned int tileNumber = 0;
          tileNumber < numberOfPointTiles; ++tileNumber) {

        // these are base indices into the shared memory
        const unsigned int leftBaseIndex = subRow * _tile_size;
        const unsigned int rightBaseIndex = _tile_size*_tile_size + subCol;

        // Here we break it back down so that we can use it
        const unsigned int leftContractionIndex = tileNumber*_tile_size + subCol;
        const unsigned int left_qp = leftContractionIndex / (_dimTens1*_dimTens2);
        const unsigned int left_combinedTens = leftContractionIndex - left_qp * (_dimTens1 * _dimTens2); // (mod)
        const unsigned int left_iTens1 = left_combinedTens / _dimTens2;
        const unsigned int left_iTens2 = left_combinedTens - left_iTens1 * _dimTens2; // (mod)

        const unsigned int rightContractionIndex = tileNumber * _tile_size + subRow;
        const unsigned int right_qp = rightContractionIndex / (_dimTens1*_dimTens2);
        const unsigned int right_combinedTens = rightContractionIndex - right_qp * (_dimTens1 * _dimTens2); // (mod)
        const unsigned int right_iTens1 = right_combinedTens / _dimTens2;
        const unsigned int right_iTens2 = right_combinedTens - right_iTens1 * _dimTens2; // (mod)

        // load the left and right tiles into shared memory
        if (resultMatrix < _numCells && row < numBasis && leftContractionIndex < contractionSize)
          tileStorage(thread.team_rank()) = _leftView(resultMatrix, row, left_qp, left_iTens1, left_iTens2);
        else
          tileStorage(thread.team_rank()) = 0.0;
        if (resultMatrix < _numCells && rightContractionIndex < contractionSize && col < numBasis)
          tileStorage(thread.team_rank() + (_tile_size * _tile_size)) =
            _rightView(resultMatrix, right_qp, right_iTens1, right_iTens2, col);
        else
          tileStorage(thread.team_rank() + (_tile_size * _tile_size)) = 0.0;

        // make sure everyone's finished loading their pieces of the tiles
        thread.team_barrier();

        for (unsigned int dummy = 0; dummy < _tile_size; ++dummy) {
          sum +=
            tileStorage(leftBaseIndex + dummy) *
            tileStorage(rightBaseIndex + dummy * _tile_size);
        }
        thread.team_barrier();
      }

      if (resultMatrix < _numCells && row < numBasis && col < numBasis)
        _outputView(resultMatrix, row, col) = sum;

      resultTileIndex += thread.league_size();
    }
  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * team_size * 2;
  }

};
