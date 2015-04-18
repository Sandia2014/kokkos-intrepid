template <class DeviceType, class KokkosTensorData,
          class KokkosTensorResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _tensorSize;
  KokkosTensorData _data_A;
  KokkosTensorData _data_B;
  KokkosTensorResults _results;

  KokkosFunctor_Independent(const unsigned int tensorSize,
                            KokkosTensorData data_A,
                            KokkosTensorData data_B,
                            KokkosTensorResults results) :
    _tensorSize(tensorSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int tensorIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _tensorSize;
         ++entryIndex) {
      sum +=
        _data_A(tensorIndex, entryIndex) *
        _data_B(tensorIndex, entryIndex);
    }
    _results(tensorIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};
