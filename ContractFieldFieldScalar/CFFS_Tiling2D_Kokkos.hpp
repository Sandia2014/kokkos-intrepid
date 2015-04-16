template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Tiling_TeamFunctor {
  const unsigned int numCells;
  const unsigned int numLeftFields;
  const unsigned int numRightFields;
  const unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;
  const unsigned int tile_size;


  CFFS_Tiling_TeamFunctor(const unsigned int _numCells,
      const unsigned int _numLeftFields,
      const unsigned int _numRightFields,
      const unsigned int _numPoints,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView,
      const unsigned int _tile_size) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView),
    tile_size(_tile_size)
    {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    //NOTE: THIS WHOLE THING WORKS ASSUMING NUMLEFTFIELDS==NUMRIGHTFIELDS
    const unsigned int numBasis = numLeftFields;

    //NOTE: This relies on contractionSize being a multiple of tileSize (16)
    const unsigned int numberOfPointTiles = numPoints / tile_size;
    //NOTE: This relies on numBasis being a multiple of tileSize(16)
    const unsigned int numberOfBasisTiles = numBasis / tile_size;

    const unsigned int numberOfTiles = numCells*numberOfBasisTiles*numberOfBasisTiles;

    const unsigned int subRow = thread.team_rank() / tile_size;
    const unsigned int subCol = thread.team_rank() - subRow * tile_size;

    unsigned int resultTileIndex = thread.league_rank();

    Kokkos::View<float**, Kokkos::MemoryUnmanaged> left_tile(thread.team_shmem(), tile_size, tile_size);
    Kokkos::View<float**, Kokkos::MemoryUnmanaged> right_tile(thread.team_shmem(), tile_size, tile_size);

    while (resultTileIndex < numberOfTiles) {

    const unsigned int resultMatrix = resultTileIndex / (numberOfBasisTiles * numberOfBasisTiles);
    const unsigned int resultSubmatrixIndex =
      resultTileIndex - resultMatrix * numberOfBasisTiles * numberOfBasisTiles;

    const unsigned int resultTileRow = resultSubmatrixIndex / numberOfBasisTiles;
    const unsigned int resultTileCol = resultSubmatrixIndex -
      resultTileRow * numberOfBasisTiles;

    // calculate this threads actual output index
    const unsigned int row = resultTileRow * tile_size + subRow;
    const unsigned int col = resultTileCol * tile_size + subCol;

    float sum = 0;
    // for tileNumber in 0...numberOfTilesPerSide
    for (unsigned int tileNumber = 0;
        tileNumber < numberOfPointTiles; ++tileNumber) {

      // load the left and right tiles into shared memory
      if (resultMatrix < numCells && row < numBasis && tileNumber*tile_size + subCol < numPoints){
        left_tile(subRow, subCol) = leftView(resultMatrix, row, tileNumber*tile_size + subCol);
      }
      else {
        left_tile(subRow,subCol) = 0.0;
      }
      if (resultMatrix < numCells && tileNumber * tile_size + subRow < numPoints && col < numBasis) {
        right_tile(subRow, subCol) = rightView(resultMatrix, tileNumber*tile_size + subRow, col);
      }
      else {
        right_tile(subRow,subCol) = 0.0;
      }

      // make sure everyone's finished loading their pieces of the tiles
      thread.team_barrier();

      for (unsigned int dummy = 0; dummy < tile_size; ++dummy) {
        sum += left_tile(subRow, dummy) * right_tile(dummy, subCol);
      }
      thread.team_barrier();
    }
    if (resultMatrix < numCells && row < numBasis && col < numBasis)
      outputView(resultMatrix, row, col) = sum;
    resultTileIndex += thread.league_size();
  }

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * team_size * 2;
  }

};
