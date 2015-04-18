/*
 * Created by: Alex Gruver
 *
 * This kernel uses a modified slicing approach to compute the output array.
 *
 * Instead of loading one row of the left input matrix into shared memory, it loads
 * in two per thread block. This allows the algorithm to better saturate the GPU, since
 * more threads are running simultaneously.  
 *
 */

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

    //The size of a block is the size of each thread's team (in this case hard coded to 2 times the number of right fields)
    const unsigned int blockSize = thread.team_size();

    //Calculate the size of each contraction
    const unsigned int contractionSize = numPoints * tens1 * tens2;

    //Calculate the "threadRow" of each thread, which is a binary value denoting whether this thread falls within
    //the first or second row of the block
    const unsigned int threadRow = thread.team_rank() / numLeftFields;

    //Calculate the column that each thread will iterate down in the right matrix
    const unsigned int col = thread.team_rank() - (threadRow * numLeftFields);

    //Find the block to which this thread belongs, and the total number of blocks working on the problem.
    unsigned int currentBlock = thread.league_rank();
    const unsigned int numBlocks = thread.league_size();

    //Create the shared memory that our threads will use
    Kokkos::View<float*, Kokkos::MemoryUnmanaged> sliceStorage(thread.team_shmem(), contractionSize*2);
    
    //Synchronize the threads
    thread.team_barrier();

    //Calculate the cell and the row of the left input matrix that this thread will be helping to load
    //into shared memory
    const unsigned int cell = (currentBlock*2) / numLeftFields;
    const unsigned int row = (currentBlock*2) - cell * numLeftFields;

    //Make sure that our thread is working on elements that are actually valid
    if((cell < numCells) && ((row+threadRow) < numLeftFields)) {
      //Each thread will help to iterate over a contraction in the left input matrix to load it into shared
      //memory. The threads in the block are broken up into two groups (by their different threadRow values)
      //and load the two contractions into shared memory simultaneously. The arthmetic is to compute the 
      //point, tens1, and tens2 values associated with each element in the contraction size.
      for (unsigned int p = col; p < contractionSize; p += (blockDim.x/2)) {
        const unsigned int pointIndex = p / (tens1*tens2);
        const unsigned int pointMod = p - (pointIndex * (tens1*tens2));
        const unsigned int tens1Index = pointMod / tens2;
        const unsigned int tens2Index = pointMod - (tens2 * tens1Index);
        sliceStorage(p + (threadRow*contractionSize)) = leftView(cell, row+threadRow, pointIndex, tens1Index,tens2Index);
      }

      //Once everything is loaded into shared memory, synchronize the threads again.
      thread.team_barrier();
      float sum = 0;

      //Now, each thread iterates over their contraction in the right matrix, doing a long dot product 
      //using the elements in shared memory. Again, the additional arithmentic converts to the dimensions
      //we used to index into right and left view
      for (int p = 0; p < contractionSize; ++p) {
        const unsigned int pointIndex = p / (tens1*tens2);
        const unsigned int pointMod = p - (pointIndex * (tens1*tens2));
        const unsigned int tens1Index = pointMod / tens2;
        const unsigned int tens2Index = pointMod - (tens2 * tens1Index);

        sum += sliceStorage(p + (threadRow*contractionSize)) *
          rightView(cell,pointIndex,tens1Index,tens2Index,col);
      }
      //Write to outputView and we're done!
      outputView(cell,row+threadRow,col) = sum;
    }

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints * tens1 * tens2 * 2;
  }
};
