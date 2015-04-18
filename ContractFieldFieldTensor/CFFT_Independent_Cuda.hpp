/* 
 * Created by: Alex Gruver
 *
 * This is the simple flat parallel scheme for CFFT implemented in raw cuda
 */

__global__
void
doCudaTensors_Independent_kernel(const unsigned int numberOfTensors,
                                 const unsigned int numLeftFields,
                                 const unsigned int numRightFields,
                                 const unsigned int numPoints,
                                 const unsigned int tens1,
                                 const unsigned int tens2,
                                 const float * const __restrict__ dev_tensorData_Left,
                                 const float * const __restrict__ dev_tensorData_Right,
                                 float * dev_tensorResults) {

  unsigned int myID = blockIdx.x * blockDim.x + threadIdx.x;
  while (myID < (numberOfTensors * numLeftFields * numRightFields)) {
    float sum = 0;
    // Calculate indices
    int myCell = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);
    int lbf = matrixIndex / numRightFields;
    int rbf = matrixIndex % numRightFields;

    // For making indexing calculations easier:
    int clOff = numLeftFields*numPoints*tens1*tens2;
    int crOff = numRightFields*numPoints*tens1*tens2;
    int cOut = numLeftFields*numRightFields;
    int lOff = numPoints*tens1*tens2;
    int lOut = numRightFields;
    int pOff = tens1*tens2;
    int tenOff = tens2;

    for (int qp = 0; qp < numPoints; qp++) {
      for (int iTens1 = 0; iTens1 < tens1; iTens1++) {
        for (int iTens2 = 0; iTens2 < tens2; iTens2++) {
          sum += dev_tensorData_Left[myCell*clOff+lbf*lOff+qp*pOff+iTens1*tenOff+iTens2] *
          dev_tensorData_Right[myCell*crOff+qp*numRightFields*tens1*tens2+
                              iTens1*numRightFields*tens2+iTens2*numRightFields
                              +rbf];
        }
      }
    }

    dev_tensorResults[myCell*cOut+lbf*lOut+rbf] = sum;

    myID += blockDim.x * gridDim.x;
  }
}
