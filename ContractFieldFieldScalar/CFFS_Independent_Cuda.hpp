/* Created by: Alex Gruver
 *
 * This implements the simple flat parallel algorithm in raw cuda.
 */

__global__
void
doCudaContractions_Independent_kernel(const unsigned int numberOfContractions,
                                     const unsigned int maxNumberOfContractions,
                                     const unsigned int contractionSize,
                                     const unsigned int numBasis,
                                     const float * const __restrict__ dev_contractionData_LayoutLeft_Right,
                                     const float * const __restrict__ dev_contractionData_LayoutLeft_Left,
                                     float * dev_contractionResults) {

  unsigned int contractionIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (contractionIndex < numberOfContractions) {

    int myID = contractionIndex;
    int myMatrix = myID / (numBasis * numBasis);
    int matrixIndex = myID % (numBasis * numBasis);

    int matrixRow = matrixIndex / numBasis;
    int matrixCol = matrixIndex % numBasis;

    float temp = 0;
    for (int qp = 0; qp < contractionSize; qp++) {
      temp += dev_contractionData_LayoutLeft_Left[myMatrix * numBasis * contractionSize + qp * numBasis + matrixRow]
      * dev_contractionData_LayoutLeft_Right[myMatrix * numBasis * contractionSize + qp*numBasis + matrixCol];
    }

    dev_contractionResults[myMatrix * numBasis * numBasis + matrixRow * numBasis + matrixCol] = temp;
    contractionIndex += blockDim.x * gridDim.x;
  }
}
