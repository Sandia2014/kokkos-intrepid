template <class DeviceType, class KokkosContractionData,
          class KokkosContractionResults>
struct KokkosFunctor_Independent {

  typedef DeviceType device_type;

  const unsigned int _contractionSize;
  KokkosContractionData _data_A;
  KokkosContractionData _data_B;
  KokkosContractionResults _results;

  KokkosFunctor_Independent(const unsigned int contractionSize,
                            KokkosContractionData data_A,
                            KokkosContractionData data_B,
                            KokkosContractionResults results) :
    _contractionSize(contractionSize), _data_A(data_A), _data_B(data_B),
    _results(results) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int contractionIndex) const {
    double sum = 0;
    for (unsigned int entryIndex = 0; entryIndex < _contractionSize;
         ++entryIndex) {
      sum +=
        _data_A(contractionIndex, entryIndex) *
        _data_B(contractionIndex, entryIndex);
    }
    _results(contractionIndex) = sum;
  }

private:
  KokkosFunctor_Independent();

};
