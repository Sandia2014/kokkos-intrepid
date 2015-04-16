__global__
void
doCudaContractions_Slicing_kernel(const unsigned int numCells,
                                     const unsigned int contractionSize,
                                     const unsigned int numBasis,
                                     const float * const __restrict__ dev_contractionData_Right,
                                     const float * const __restrict__ dev_contractionData_Left,
                                     float * dev_contractionResults) {

  extern __shared__ float sliceStorage[];

  const unsigned int col = threadIdx.x;

  unsigned int currentBlock = blockIdx.x;
  unsigned int numBlocks = numBasis*numCells;

  while (currentBlock < numBlocks) {
    syncthreads();
    const unsigned int cell = currentBlock / numBasis;
    const unsigned int row = currentBlock - cell * numBasis;

    dev_contractionResults[cell*numBasis*numBasis + row*numBasis + col] = -1;

    for (unsigned int p = threadIdx.x; p < contractionSize; p += blockDim.x) {
      sliceStorage[p] = dev_contractionData_Left[cell*numBasis*contractionSize +
        row*contractionSize + p];
    }
    syncthreads();

    float sum = 0;
    for (int p = 0; p < contractionSize; ++p) {
      sum += sliceStorage[p] * dev_contractionData_Right[cell*numBasis*contractionSize +
        p*numBasis + col];
    }

    dev_contractionResults[cell*numBasis*numBasis + row*numBasis + col] = sum;

    currentBlock += gridDim.x;
  }
}
