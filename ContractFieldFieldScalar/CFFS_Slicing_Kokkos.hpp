template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Slicing_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Slicing_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
      unsigned int _numRightFields, unsigned int _numPoints,
      LeftInputViewType _leftView,
      RightInputViewType _rightView,
      OutputViewType _outputView) :
    numCells(_numCells),
    numLeftFields(_numLeftFields),
    numRightFields(_numRightFields),
    numPoints(_numPoints),
    leftView(_leftView),
    rightView(_rightView),
    outputView(_outputView) {
      // Nothing to do
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member & thread) const {
    int r = thread.team_rank();
    int c = thread.league_rank() / numLeftFields;
    int l = thread.league_rank() - c * numLeftFields;

    Kokkos::View<float*, Kokkos::MemoryUnmanaged> shared_slice(thread.team_shmem(), numPoints);
    for (int p = thread.team_rank(); p < numPoints; p += thread.team_size()) {
      shared_slice(p) = leftView(c, l, p);
    }
    thread.team_barrier();

    float sum = 0;
    for (int p = 0; p < numPoints; ++p) {
      sum += shared_slice(p) * rightView(c, p, r);
    }
    outputView(c, l, r) = sum;

  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * numPoints;
  }
};
