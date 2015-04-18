__global__
void
doCudaContractions_Tiling_kernel(const unsigned int numCells,
                                 const unsigned int contractionSize,
                                 const unsigned int tileSize,
                                 const unsigned int numBasis,
                                 const float * const __restrict__ dev_contractionData_Right,
                                 const float * const __restrict__ dev_contractionData_Left,
                                 float * dev_contractionResults) {

  extern __shared__ float tileStorage[];

  const unsigned int numbersPerTile = tileSize * tileSize;
  //NOTE: This relies on contractionSize being a multiple of tileSize (16)
  const unsigned int numberOfHorizontalTiles = contractionSize / tileSize;
  //NOTE: This relies on numBasis being a multiple of tileSize(16)
  const unsigned int numberOfVerticalTiles = numBasis / tileSize;

  const unsigned int numberOfTiles = numCells * numberOfVerticalTiles * numberOfVerticalTiles;

  const unsigned int subRow = threadIdx.x / tileSize;
  const unsigned int subCol = threadIdx.x  - subRow * tileSize;

  unsigned int resultTileIndex = blockIdx.x;

  while (resultTileIndex < numberOfTiles) {

    unsigned int resultSubmatrixIndex = resultTileIndex % (numberOfVerticalTiles * numberOfVerticalTiles);
    unsigned int resultMatrix = resultTileIndex / (numberOfVerticalTiles * numberOfVerticalTiles);

    // for tileNumber in 0...numberOfTilesPerSide
    for (unsigned int tileNumber = 0;
        tileNumber < numberOfHorizontalTiles; ++tileNumber) {
      // calculate result tile indices

      const unsigned int resultTileRow = resultSubmatrixIndex / numberOfHorizontalTiles;
      const unsigned int resultTileCol = resultSubmatrixIndex  -
        resultTileRow * numberOfHorizontalTiles;

      // calculate this threads actual output index
      const unsigned int row = resultTileRow * tileSize + subRow;
      const unsigned int col = resultTileCol * tileSize + subCol;

      // these are base indices into the shared memory
      const unsigned int leftBaseIndex = subRow * tileSize;
      const unsigned int rightBaseIndex = numbersPerTile + subCol;

      const unsigned int resultIndex = row * numBasis + col;

      // load the left and right tiles into shared memory
      syncthreads();

      if (resultMatrix < numCells && row < numBasis && tileNumber*tileSize + subCol < contractionSize)
        tileStorage[threadIdx.x] = dev_contractionData_Left[resultMatrix * numBasis * contractionSize
                                   + row * contractionSize + tileNumber * tileSize + subCol];
      else
        tileStorage[threadIdx.x] = 0.0;


      if (resultMatrix < numCells && tileNumber * tileSize + subRow < contractionSize && col < numBasis)
        tileStorage[threadIdx.x + blockDim.x] = dev_contractionData_Right[resultMatrix * numBasis * contractionSize
                                                 + (tileNumber * tileSize + subRow) * numBasis + col];
      else
        tileStorage[threadIdx.x + blockDim.x] = 0.0;
      // make sure everyone's finished loading their pieces of the tiles
      syncthreads();

      double sum = 0;
      for (unsigned int dummy = 0; dummy < tileSize; ++dummy) {
        sum +=
          tileStorage[leftBaseIndex + dummy] *
          tileStorage[rightBaseIndex + dummy * tileSize];
      }
      if (resultMatrix < numCells && row < numBasis && col < numBasis)
        dev_contractionResults[resultIndex] += sum;
    }
    resultTileIndex += gridDim.x;
  }

}
