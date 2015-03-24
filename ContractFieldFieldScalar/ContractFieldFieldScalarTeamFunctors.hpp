#include <Kokkos_Core.hpp>

typedef Kokkos::DefaultExecutionSpace       Device;
typedef Kokkos::HostSpace::execution_space  Host;

typedef Kokkos::TeamPolicy< Device >      team_policy;
typedef team_policy::member_type team_member;

static const int TILESIZE = 8;

template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Reduction_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Reduction_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
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
    int myID =  thread.league_rank();
    int myMatrix = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);

    int matrixRow = matrixIndex / numRightFields;
    int matrixCol = matrixIndex % numRightFields;

    float sum = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadLoop(thread, numPoints), 
        [&] (const unsigned int& i, float& sum) {
          sum += leftView(myMatrix, matrixRow, i) 
                 * rightView(myMatrix, i, matrixCol);
        }, 
        sum);
    outputView(myMatrix, matrixRow, matrixCol) = sum;
  }
};


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
    int l = thread.league_rank() % numLeftFields;
    int r = thread.team_rank();
    int c = thread.league_rank() / numLeftFields;

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


template <class LeftInputViewType, class RightInputViewType, class OutputViewType>
struct CFFS_Tiling_TeamFunctor {
  unsigned int numCells;
  unsigned int numLeftFields;
  unsigned int numRightFields;
  unsigned int numPoints;
  LeftInputViewType leftView;
  RightInputViewType rightView;
  OutputViewType outputView;


  CFFS_Tiling_TeamFunctor(unsigned int _numCells, unsigned int _numLeftFields,
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
    // Num teams is (numLeftField * numRightField)/TILESIZE^2 * numCells
    int numTiles = thread.league_size() / numCells;
    int c = thread.league_rank() / numTiles;
    int tilePosition = thread.league_rank() % numTiles;
    int lTile = tilePosition / ((numRightFields-1) / TILESIZE + 1);
    int rTile = tilePosition % ((numRightFields-1) / TILESIZE + 1);

    int tileCol = thread.team_rank() % TILESIZE;
    int tileRow = thread.team_rank() / TILESIZE;
    
    int l = lTile*TILESIZE + tileRow;
    int r = rTile*TILESIZE + tileCol;

    Kokkos::View<float**, Kokkos::MemoryUnmanaged> left_tile(thread.team_shmem(), TILESIZE, TILESIZE);
    Kokkos::View<float**, Kokkos::MemoryUnmanaged> right_tile(thread.team_shmem(), TILESIZE, TILESIZE);

    float totalSum = 0;
    for (int tileIndex = 0; tileIndex < ((numPoints-1)/ TILESIZE) + 1; ++tileIndex) {
	if (tileIndex*TILESIZE + tileCol < numPoints && l < numLeftFields) {
	    left_tile(tileRow, tileCol) = leftView(c, l, tileIndex*TILESIZE + tileCol);
	}
	else {
	    left_tile(tileRow, tileCol) = 0.0;
	}
	if (tileIndex*TILESIZE + tileRow < numPoints && r < numRightFields) {
	    right_tile(tileRow, tileCol) = rightView(c, tileIndex*TILESIZE + tileRow, r);
	}
	else {
	    right_tile(tileRow, tileCol) = 0.0;
	}
	thread.team_barrier();

	float sum = 0;
	for (int i = 0; i < TILESIZE; ++i) {
	    sum += left_tile(tileRow, i) * right_tile(i, tileCol);
	}
	totalSum += sum;
	
	thread.team_barrier();
    }

    if (l < numLeftFields && r < numRightFields) {
	outputView(c, l, r) = totalSum;
    }
  }

  size_t team_shmem_size( int team_size ) const {
    return sizeof(float) * team_size * 2;
  }

};


