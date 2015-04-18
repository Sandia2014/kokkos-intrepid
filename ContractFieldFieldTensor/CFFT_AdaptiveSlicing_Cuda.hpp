/*
 * Created by: Alex Gruver
 *
 * This algorithm is identical to the one used in CFFT_AdaptiveSlicing_Kokkos, so for 
 * information on how exactly the algorithm works please refer to that file, as the two
 * approaches are the same except for minor syntax differences
 */


__global__
void
doCudaContractions_AdaptiveSlicing_kernel(const unsigned int numberOfTensors,
                                 const unsigned int numLeftFields,
                                 const unsigned int numRightFields,
                                 const unsigned int numPoints,
                                 const unsigned int tens1,
                                 const unsigned int tens2,
                                 const float * const __restrict__ dev_tensorData_Left,
                                 const float * const __restrict__ dev_tensorData_Right,
                                 float * dev_tensorResults) {

  extern __shared__ float sliceStorage[];

  const unsigned int contractionSize = numPoints * tens1 * tens2;
  const unsigned int threadRow = threadIdx.x / numLeftFields;
  const unsigned int col = threadIdx.x - (threadRow * numLeftFields);

  unsigned int currentBlock = blockIdx.x;
  const unsigned int numBlocks = gridDim.x;

  while (currentBlock < numBlocks) {
    syncthreads();
    const unsigned int cell = (currentBlock*2) / numLeftFields;
    const unsigned int row = (currentBlock*2) - cell * numLeftFields;

    if((cell < numberOfTensors) && ((row+threadRow) < numLeftFields)) {
      for (unsigned int p = col; p < contractionSize; p += (blockDim.x/2)) {
        sliceStorage[p + (threadRow*contractionSize)] = dev_tensorData_Left[cell*numLeftFields*contractionSize +
          (row+threadRow)*contractionSize + p];
        }
      syncthreads();
      float sum = 0;
      for (int p = 0; p < contractionSize; ++p) {
        sum += sliceStorage[p + (threadRow*contractionSize)] * dev_tensorData_Right[cell*numRightFields*contractionSize +
          p*numRightFields + col];
        }

        dev_tensorResults[cell*numRightFields*numLeftFields + (row+threadRow)*numRightFields + col] = sum;
    }
    currentBlock += gridDim.x;
  }
}
