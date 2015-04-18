/*
 * Created by: Alex Gruver
 *
 * This is a generalization of our Slicing algorithm from ContractFieldFieldScalar. 
 * It works by pretending all three of the contraction dimensions are a single dimension,
 * which is an okay generalization because they are adjacent in memory.
 */

template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Slicing_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  unsigned int tens1;
  unsigned int tens2;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Slicing_TeamFunctor(unsigned int _numCells,
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
    // These calculations are taken directly from the CFFS version
    int r = thread.team_rank();
    int c = thread.league_rank() / numLeftFields;
    int l = thread.league_rank() - c * numLeftFields; // (mod)

    // Allocate our slice in shared memory
    Kokkos::View<float*, Kokkos::MemoryUnmanaged> shared_slice(thread.team_shmem(), numPoints);
    for (int p = thread.team_rank(); p < numPoints*tens1*tens2; p += thread.team_size()) {
      // Here, we break "p" down into the actual 3 indices for the contraction
      const unsigned int pointIndex = p / (tens1*tens2);
      const unsigned int pointMod = p - (pointIndex * (tens1*tens2)); // (mod)
      const unsigned int tens1Index = pointMod / tens2;
      const unsigned int tens2Index = pointMod - (tens2 * tens1Index); // (mod)

      // And load the appropriate slice into shared memory.
      shared_slice(p) = leftView(c, l, pointIndex,tens1Index,tens2Index );
    }
    thread.team_barrier();

    float sum = 0;
    for (int p = 0; p < numPoints*tens1*tens2; ++p) {
      // Again, we must break "p" down into its compoenents to index our input
      const unsigned int pointIndex = p / (tens1*tens2);
      const unsigned int pointMod = p - (pointIndex * (tens1*tens2)); // (mod)
      const unsigned int tens1Index = pointMod / tens2;
      const unsigned int tens2Index = pointMod - (tens2 * tens1Index); // (mod)

      // And use our slice to calculate the appropriate output
      sum += shared_slice(p) * rightView(c, pointIndex, tens1Index, tens2Index, r);
    }

    // No special indexing is needed here, as we only collapsed the contraction indicies,
    // which do not appear in the output tensor
    outputView(c, l, r) = sum;

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints * tens1 * tens2;
  }
};
