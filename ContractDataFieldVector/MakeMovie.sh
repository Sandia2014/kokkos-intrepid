#!/bin/bash

#suffix=shadowfax
suffix=clearCache_shadowfax

flavors=()
flavors+=('NormalizedTime_serial')
flavors+=('NormalizedTime_omp')
flavors+=('NormalizedTime_cudaIndependent')
flavors+=('NormalizedTime_cudaReduction')
flavors+=('NormalizedTime_cudaSwitching')
flavors+=('NormalizedTime_kokkosOmp')
flavors+=('NormalizedTime_kokkosCudaIndependent')

flavors+=('VersusSerial_omp')
flavors+=('VersusSerial_cudaIndependent')
flavors+=('VersusSerial_cudaReduction')
flavors+=('VersusSerial_cudaSwitching')
flavors+=('VersusSerial_kokkosOmp')
flavors+=('VersusSerial_kokkosCudaIndependent')

flavors+=('VersusOmp_cudaIndependent')
flavors+=('VersusOmp_cudaReduction')
flavors+=('VersusOmp_cudaSwitching')
flavors+=('VersusOmp_kokkosOmp')
flavors+=('VersusOmp_kokkosCudaIndependent')

flavors+=('VersusCudaIndependent_cudaReduction')
flavors+=('VersusCudaIndependent_cudaSwitching')
flavors+=('VersusCudaIndependent_kokkosCudaIndependent')

flavors+=('VersusOmp_kokkosOmp_linear')
flavors+=('VersusCudaIndependent_kokkosCudaIndependent_linear')

for flavor in "${flavors[@]}"
do
  mogrify -format jpg -density 300 figures/orbitFrames/${flavor}_${suffix}*.pdf
  avconv -r 25 -f image2 -i figures/orbitFrames/${flavor}_${suffix}_%02d.jpg -qscale 0 -vcodec mpeg4 movies/${flavor}_${suffix}.mp4
  #ffmpeg -r 25 -f image2 -i figures/orbitFrames/${flavor}_${suffix}_%02d.jpg -qscale 0 -vcodec mpeg4 movies/${flavor}_${suffix}.mp4
done
