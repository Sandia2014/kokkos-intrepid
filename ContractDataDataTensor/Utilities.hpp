// Utilities.hpp
// Various useful helper functions
// Ellen Hui (mostly stolen from Jeff Amelang), 2015

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>

// c++ junk
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <array>
#include <fstream>
using std::string;
using std::vector;
using std::array;


enum CudaStyle {CudaStyle_Independent,
                CudaStyle_Reduction};


string
convertCudaStyleToString(const CudaStyle cudaStyle) {
  switch (cudaStyle) {
  case CudaStyle_Independent:
    return string("CudaStyle_Independent");
  case CudaStyle_Reduction:
    return string("CudaStyle_Reduction");
  default:
    fprintf(stderr, "invalid cuda style\n");
    exit(1);
  };
}

// stolen from
// "http://stackoverflow.com/questions/14038589/
//  what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api"
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline
void
gpuAssert(const cudaError_t code, const char *file, const int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort == true) {
      exit(code);
    }
  }
}

timespec
getTimePoint() {
  timespec timepoint;
  clock_gettime(CLOCK_MONOTONIC, &timepoint);
  return timepoint;
}

// yay for having to use pre-c++11 timing because of nvcc
double
getElapsedTime(const timespec & start, const timespec & end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}


void
writeTimesMatrixToFile(const vector<vector<float> > & times,
                       const string filename) {

  const unsigned int numberOfContractionSizes = times.size();
  // yeah, yeah, kinda unsafe
  const unsigned int numberOfMemorySizes = times[0].size();
  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%s.csv", filename.c_str());
  FILE* file = fopen(sprintfBuffer, "w");
  for (unsigned int contractionSizeIndex = 0;
       contractionSizeIndex < numberOfContractionSizes;
       ++contractionSizeIndex) {
    for (unsigned int memorySizeIndex = 0;
         memorySizeIndex < numberOfMemorySizes;
         ++memorySizeIndex) {
      if (memorySizeIndex > 0) {
        fprintf(file, ", ");
      }
      fprintf(file, "%10.4e", times[contractionSizeIndex][memorySizeIndex]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

void
checkAnswer(const vector<float> & correctResults,
            const vector<float> & calcResults,
            const unsigned int contractionSize,
            const unsigned int memorySize,
            const string flavorName) {
  for (unsigned int dotProductIndex = 0;
       dotProductIndex < correctResults.size();
       ++dotProductIndex) {
    if (std::abs(correctResults[dotProductIndex] -
                 calcResults[dotProductIndex]) /
        std::abs(correctResults[dotProductIndex]) > 1e-4) {
      fprintf(stderr, "invalid answer for dot product index %u for "
              "flavor %s, "
              "should be %e but we have %e, "
              "contractionSize = %u, memorySize = %8.2e\n",
              dotProductIndex, flavorName.c_str(),
              correctResults[dotProductIndex],
              calcResults[dotProductIndex],
              contractionSize, float(memorySize));
      exit(1);
    }
  }
}
