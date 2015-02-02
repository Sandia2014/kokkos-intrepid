import math
import os
import sys
import numpy
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import csv
from mpl_toolkits.mplot3d import Axes3D
from numpy import log10

prefix = 'data/ContractDataDataTensor_'
suffix = '_clearCache_shadowfax'
outputPrefix = 'figures/'

# read in all of the data.  
# TODO: you'll need to disable everything that's not relevant here or it'll be angry about missing files
dotProductSize = numpy.loadtxt(open(prefix + 'contractionSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
memorySize = numpy.loadtxt(open(prefix + 'memorySize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
numberOfDotProducts = numpy.loadtxt(open(prefix + 'numberOfDotProducts' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
serialTimes = numpy.loadtxt(open(prefix + 'serialTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
ompTimes = numpy.loadtxt(open(prefix + 'ompTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
#cudaIndependentTimes = numpy.loadtxt(open(prefix + 'cudaIndependentTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
#cudaReductionTimes = numpy.loadtxt(open(prefix + 'cudaReductionTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
#cudaSwitchingTimes = numpy.loadtxt(open(prefix + 'cudaSwitchingTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosOmpTimes = numpy.loadtxt(open(prefix + 'kokkosOmpTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaIndependentTimes = numpy.loadtxt(open(prefix + 'kokkosCudaIndependentTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

# set up a list of the times and names, for easy iteration later
# TODO: make this consistent with the files that you read in and/or care about
allTimes = []
allNames = []
# NOTE: if you are doing comparisons against serial time, it's assumed that the first entry in allTimes is serial
allTimes.append(serialTimes)
allNames.append('serial')
# NOTE: if you are doing comparisons against omp time, it's assumed that the second entry in allTimes is openmp.  if you aren't doing those comparisons, you should go disable that portion of this script.
allTimes.append(ompTimes)
allNames.append('omp')
# NOTE: if you are doing comparisons against cuda time, it's assumed that the third entry in allTimes is cuda.  if you aren't doing those comparisons, you should go disable that portion of this script.
#allTimes.append(cudaIndependentTimes)
#allNames.append('cudaIndependent')
# there are no assumptions about the rest of the ordering
#allTimes.append(cudaReductionTimes)
#allNames.append('cudaReduction')
#allTimes.append(cudaSwitchingTimes)
#allNames.append('cudaSwitching')
allTimes.append(kokkosOmpTimes)
allNames.append('kokkosOmp')
allTimes.append(kokkosCudaIndependentTimes)
allNames.append('kokkosCudaIndependent')

# these are toggles for whether to make image files and whether to make orbit files for making movies
makeImageFiles = True
#makeImageFiles = False
#makeOrbitFilesForMovies = True
makeOrbitFilesForMovies = False
numberOfOrbitFrames = 100


#markerPool = ['-', '--', ':']
markerPool = ['-', '--']
colors = cm.gist_ncar(numpy.linspace(1, 0, len(allTimes)))
markers = []
for i in range(len(allTimes)):
  markers.append(markerPool[i % len(markerPool)])

fig3d = plt.figure(0)
fig2d = plt.figure(1, figsize=(14, 6))
ax2d = plt.subplot(111)
box2d = ax2d.get_position()
ax2d.set_position([box2d.x0, box2d.y0, box2d.width * 0.60, box2d.height])
bbox_to_anchor2d = (1.87, 0.5)

# make an image of just the number of dot products
# TODO: you might want to make an image of the number of cells, so you'd adjust this.
fig3d = plt.figure(0)
ax = fig3d.gca(projection='3d')
ax.view_init(elev=0, azim=-111)
surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(numberOfDotProducts), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(numPoints)')
plt.ylabel('log10(memorySize)')
ax.set_zlabel('log10(numCells)')
plt.title('number of cells')
if (makeImageFiles == True):
  ax.view_init(elev=2, azim=-23)
  filename = outputPrefix + 'numCells' + suffix
  plt.savefig(filename + '.pdf')
  print 'saved file to %s' % filename
else:
  plt.show()

# goal: make images showing just the raw times
# find the min and max values across all flavors so that the color scale is the same for each graph
maxValue = -10
minValue = 10
for timesIndex in numpy.arange(0, len(allTimes)):
  maxValue = numpy.max([maxValue, numpy.max(log10(allTimes[timesIndex]))])
  minValue = numpy.min([minValue, numpy.min(log10(allTimes[timesIndex]))])
# make the color scale
colorNormalizer = matplotlib.colors.Normalize(vmin=minValue, vmax=maxValue)
# for each time
for timesIndex in range(len(allTimes)):
  # make a 3d plot
  fig3d = plt.figure(0)
  plt.clf()
  times = allTimes[timesIndex]
  name = allNames[timesIndex]
  ax = fig3d.gca(projection='3d')
  ax.view_init(elev=0, azim=-111)
  surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(times), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
  surf.set_norm(colorNormalizer)
  plt.xlabel('log10(numPoints)')
  plt.ylabel('log10(memorySize)')
  ax.set_zlabel('log10(raw time) [seconds]')
  ax.set_zlim([minValue, maxValue])
  plt.title(name + ' raw time')
  if (makeImageFiles == True):
    ax.view_init(elev=2, azim=-23)
    filename = outputPrefix + 'RawTimes_' + name + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
  else:
    plt.show()
# make a 2D plot of all flavors, for the smallest and largest sizes of memory
fig2d = plt.figure(1)
for memorySizeIndex in [-1, 0]:
  legendNames = []
  plt.cla()
  for timesIndex in range(len(allTimes)):
    times = allTimes[timesIndex]
    name = allNames[timesIndex]
    plt.plot(dotProductSize[:, memorySizeIndex], times[:, memorySizeIndex], markers[timesIndex], color=colors[timesIndex], hold='on', linewidth=2)
    legendNames.append(name)
  plt.xscale('log')
  plt.yscale('log')
  plt.title('raw times for memory size %.2e' % memorySize[0, memorySizeIndex], fontsize=16)
  plt.xlabel('number of points', fontsize=16)
  plt.ylabel('raw time [seconds]', fontsize=16)
  plt.xlim([dotProductSize[0, 0], dotProductSize[-1, 0]])
  ax2d.legend(legendNames, loc='center right', bbox_to_anchor=bbox_to_anchor2d)
  if (makeImageFiles == True):
    sizeDescription = 'largestSize' if (memorySizeIndex == -1) else 'smallestSize'
    filename = outputPrefix + 'RawTimes_2d_' + sizeDescription + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
  else:
    plt.show()


# now make plots that are normalized by memory size
maxValue = -10
minValue = 10
for timesIndex in numpy.arange(0, len(allTimes)):
  maxValue = numpy.max([maxValue, numpy.max(log10(allTimes[timesIndex] / memorySize))])
  minValue = numpy.min([minValue, numpy.min(log10(allTimes[timesIndex] / memorySize))])
colorNormalizer = matplotlib.colors.Normalize(vmin=minValue, vmax=maxValue)
for timesIndex in range(len(allTimes)):
  fig3d = plt.figure(0)
  plt.clf()
  times = allTimes[timesIndex]
  name = allNames[timesIndex]
  ax = fig3d.gca(projection='3d')
  ax.view_init(elev=0, azim=-111)
  surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(times / memorySize), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
  surf.set_norm(colorNormalizer)
  plt.xlabel('log10(numPoints)')
  plt.ylabel('log10(memorySize)')
  ax.set_zlabel('log10(normalized time [seconds / memorySize])')
  ax.set_zlim([minValue, maxValue])
  plt.title(name + ' normalized time')
  if (makeImageFiles == True):
    ax.view_init(elev=2, azim=-23)
    filename = outputPrefix + 'NormalizedTime_' + name + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
# possibly make orbit plots for movies
    if (makeOrbitFilesForMovies == True):
      for frameIndex in range(numberOfOrbitFrames):
        ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
        filename = outputPrefix + 'orbitFrames/NormalizedTime_' + name + suffix + '_%02d.pdf' % frameIndex
        plt.savefig(filename)
        print 'saved file to %s' % filename
  else:
    plt.show()


# now make relative speedups over serial
maxSpeedup = -10
minSpeedup = 10
for timesIndex in numpy.arange(1, len(allTimes)):
  maxSpeedup = numpy.max([maxSpeedup, numpy.max(log10(allTimes[0] / allTimes[timesIndex]))])
  minSpeedup = numpy.min([minSpeedup, numpy.min(log10(allTimes[0] / allTimes[timesIndex]))])
colorNormalizer = matplotlib.colors.Normalize(vmin=minSpeedup, vmax=maxSpeedup)
# intentionally start at 1 so that i don't compare serial to serial
for timesIndex in numpy.arange(1, len(allTimes)):
  fig3d = plt.figure(0)
  plt.clf()
  times = allTimes[timesIndex]
  name = allNames[timesIndex]
  ax = fig3d.gca(projection='3d')
  ax.view_init(elev=0, azim=-111)
  surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(allTimes[0] / times), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
  surf.set_norm(colorNormalizer)
  plt.xlabel('log10(numPoints)')
  plt.ylabel('log10(memorySize)')
  ax.set_zlabel('log10(speedup) [unitless]')
  ax.set_zlim([minSpeedup, maxSpeedup])
  plt.title(name + ' speedup over serial')
  if (makeImageFiles == True):
    ax.view_init(elev=2, azim=-23)
    filename = outputPrefix + 'VersusSerial_' + name + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
    if (makeOrbitFilesForMovies == True and timesIndex > 0):
      for frameIndex in range(numberOfOrbitFrames):
        ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
        filename = outputPrefix + 'orbitFrames/VersusSerial_' + name + suffix + '_%02d.pdf' % frameIndex
        plt.savefig(filename)
        print 'saved file to %s' % filename
  else:
    plt.show()
fig2d = plt.figure(1)
for memorySizeIndex in [-1, 0]:
  legendNames = []
  plt.cla()
  for timesIndex in range(len(allTimes)):
    times = allTimes[timesIndex]
    name = allNames[timesIndex]
    plt.plot(dotProductSize[:, memorySizeIndex], allTimes[0][:, memorySizeIndex] / times[:, memorySizeIndex], markers[timesIndex], color=colors[timesIndex], hold='on', linewidth=2)
    legendNames.append(name)
  plt.xscale('log')
  plt.yscale('log')
  plt.title('speedup over serial for memory size %.2e' % memorySize[0, memorySizeIndex], fontsize=16)
  plt.xlabel('number of points', fontsize=16)
  plt.ylabel('speedup [unitless]', fontsize=16)
  #plt.ylim([0, 6])
  plt.xlim([dotProductSize[0, 0], dotProductSize[-1, 0]])
  ax2d.legend(legendNames, loc='center right', bbox_to_anchor=bbox_to_anchor2d)
  if (makeImageFiles == True):
    sizeDescription = 'largestSize' if (memorySizeIndex == -1) else 'smallestSize'
    filename = outputPrefix + 'VersusSerial_2d_' + sizeDescription + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
  else:
    plt.show()

"""
# now make relative speedup over openmp
# TODO: you might disable this part
maxSpeedup = -10
minSpeedup = 10
for timesIndex in numpy.arange(2, len(allTimes)):
  maxSpeedup = numpy.max([maxSpeedup, numpy.max(log10(allTimes[1] / allTimes[timesIndex]))])
  minSpeedup = numpy.min([minSpeedup, numpy.min(log10(allTimes[1] / allTimes[timesIndex]))])
colorNormalizer = matplotlib.colors.Normalize(vmin=minSpeedup, vmax=maxSpeedup)
# intentionally start at 2 so that i don't compare serial or omp to omp
for timesIndex in numpy.arange(2, len(allTimes)):
  fig3d = plt.figure(0)
  plt.clf()
  times = allTimes[timesIndex]
  name = allNames[timesIndex]
  ax = fig3d.gca(projection='3d')
  ax.view_init(elev=0, azim=-111)
  surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(allTimes[1] / times), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
  surf.set_norm(colorNormalizer)
  plt.xlabel('log10(dotProductSize)')
  plt.ylabel('log10(memorySize)')
  ax.set_zlabel('log10(speedup) [unitless]')
  ax.set_zlim([minSpeedup, maxSpeedup])
  plt.title(name + ' speedup over omp')
  if (makeImageFiles == True):
    ax.view_init(elev=2, azim=-23)
    filename = outputPrefix + 'VersusOmp_' + name + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
    if (makeOrbitFilesForMovies == True and timesIndex > 1):
      for frameIndex in range(numberOfOrbitFrames):
        ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
        filename = outputPrefix + 'orbitFrames/VersusOmp_' + name + suffix + '_%02d.pdf' % frameIndex
        plt.savefig(filename)
        print 'saved file to %s' % filename
  else:
    plt.show()
fig2d = plt.figure(1)
for memorySizeIndex in [-1, 0]:
  legendNames = []
  plt.cla()
  for timesIndex in range(len(allTimes)):
    times = allTimes[timesIndex]
    name = allNames[timesIndex]
    plt.plot(dotProductSize[:, memorySizeIndex], allTimes[1][:, memorySizeIndex] / times[:, memorySizeIndex], markers[timesIndex], color=colors[timesIndex], hold='on', linewidth=2)
    legendNames.append(name)
  plt.xscale('log')
  plt.yscale('log')
  plt.title('speedup over openmp for memory size %.2e' % memorySize[0, memorySizeIndex], fontsize=16)
  plt.xlabel('dot product size', fontsize=16)
  plt.ylabel('speedup [unitless]', fontsize=16)
  plt.xlim([dotProductSize[0, 0], dotProductSize[-1, 0]])
  ax2d.legend(legendNames, loc='center right', bbox_to_anchor=bbox_to_anchor2d)
  if (makeImageFiles == True):
    sizeDescription = 'largestSize' if (memorySizeIndex == -1) else 'smallestSize'
    filename = outputPrefix + 'VersusOmp_2d_' + sizeDescription + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
  else:
    plt.show()
    """

# relative speedup over cudaIndependent
# TODO: you might disable this part
""" disabled: no raw cuda
maxSpeedup = -10
minSpeedup = 10
for timesIndex in numpy.arange(3, len(allTimes)):
  maxSpeedup = numpy.max([maxSpeedup, numpy.max(log10(allTimes[2] / allTimes[timesIndex]))])
  minSpeedup = numpy.min([minSpeedup, numpy.min(log10(allTimes[2] / allTimes[timesIndex]))])
colorNormalizer = matplotlib.colors.Normalize(vmin=minSpeedup, vmax=maxSpeedup)
# intentionally start at 3 so that i don't compare cuda or serial or omp to cuda
for timesIndex in numpy.arange(3, len(allTimes)):
  fig3d = plt.figure(0)
  plt.clf()
  times = allTimes[timesIndex]
  name = allNames[timesIndex]
  ax = fig3d.gca(projection='3d')
  ax.view_init(elev=0, azim=-111)
  surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), log10(allTimes[2] / times), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
  surf.set_norm(colorNormalizer)
  plt.xlabel('log10(dotProductSize)')
  plt.ylabel('log10(memorySize)')
  ax.set_zlabel('log10(speedup) [unitless]')
  ax.set_zlim([minSpeedup, maxSpeedup])
  plt.title(name + ' speedup over cudaIndependent')
  if (makeImageFiles == True):
    ax.view_init(elev=2, azim=-23)
    filename = outputPrefix + 'VersusCudaIndependent_' + name + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
    if (makeOrbitFilesForMovies == True and timesIndex > 2):
      for frameIndex in range(numberOfOrbitFrames):
        ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
        filename = outputPrefix + 'orbitFrames/VersusCudaIndependent_' + name + suffix + '_%02d.pdf' % frameIndex
        plt.savefig(filename)
        print 'saved file to %s' % filename
  else:
    plt.show()
fig2d = plt.figure(1)
for memorySizeIndex in [-1, 0]:
  legendNames = []
  plt.cla()
  for timesIndex in range(len(allTimes)):
    times = allTimes[timesIndex]
    name = allNames[timesIndex]
    plt.plot(dotProductSize[:, memorySizeIndex], allTimes[2][:, memorySizeIndex] / times[:, memorySizeIndex], markers[timesIndex], color=colors[timesIndex], hold='on', linewidth=2)
    legendNames.append(name)
  plt.xscale('log')
  plt.yscale('log')
  plt.title('speedup over cuda independent for memory size %.2e' % memorySize[0, memorySizeIndex], fontsize=16)
  plt.xlabel('dot product size', fontsize=16)
  plt.ylabel('speedup [unitless]', fontsize=16)
  plt.xlim([dotProductSize[0, 0], dotProductSize[-1, 0]])
  ax2d.legend(legendNames, loc='center right', bbox_to_anchor=bbox_to_anchor2d)
  if (makeImageFiles == True):
    sizeDescription = 'largestSize' if (memorySizeIndex == -1) else 'smallestSize'
    filename = outputPrefix + 'VersusCudaIndependent_2d_' + sizeDescription + suffix
    plt.savefig(filename + '.pdf')
    print 'saved file to %s' % filename
  else:
    plt.show()
"""

# these graphs are essentially duplicates of ones made already, but with a linear scale instead of logarithmic (by request of carter).
# these graphs just compare kokkos omp versus openmp and kokkos cuda versus cuda
"""
# omp
fig3d = plt.figure(0)
plt.clf()
ax = fig3d.gca(projection='3d')
ax.view_init(elev=0, azim=-111)
surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), (allTimes[1] / allTimes[allNames.index("kokkosOmp")]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(dotProductSize)')
plt.ylabel('log10(memorySize)')
ax.set_zlabel('speedup [unitless]')
plt.title('kokkos omp speedup over omp')
if (makeImageFiles == True):
  ax.view_init(elev=2, azim=-23)
  filename = outputPrefix + 'VersusOmp_kokkosOmp_linear' + suffix
  plt.savefig(filename + '.pdf')
  print 'saved file to %s' % filename
  if (makeOrbitFilesForMovies == True):
    for frameIndex in range(numberOfOrbitFrames):
      ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
      filename = outputPrefix + 'orbitFrames/VersusOmp_kokkosOmp_linear' + suffix + '_%02d.pdf' % frameIndex
      plt.savefig(filename)
      print 'saved file to %s' % filename
else:
  plt.show()
"""
# cuda
""" Disabled while no raw cuda
fig3d = plt.figure(0)
plt.clf()
ax = fig3d.gca(projection='3d')
ax.view_init(elev=0, azim=-111)
surf = ax.plot_surface(log10(dotProductSize), log10(memorySize), (allTimes[2] / allTimes[6]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
plt.xlabel('log10(dotProductSize)')
plt.ylabel('log10(memorySize)')
ax.set_zlabel('speedup [unitless]')
plt.title('kokkos cuda speedup over cuda')
if (makeImageFiles == True):
  ax.view_init(elev=2, azim=-23)
  filename = outputPrefix + 'VersusCudaIndependent_kokkosCudaIndependent_linear' + suffix
  plt.savefig(filename + '.pdf')
  print 'saved file to %s' % filename
  if (makeOrbitFilesForMovies == True):
    for frameIndex in range(numberOfOrbitFrames):
      ax.view_init(elev=2, azim=360 * frameIndex / (numberOfOrbitFrames - 1))
      filename = outputPrefix + 'orbitFrames/VersusCudaIndependent_kokkosCudaIndependent_linear' + suffix + '_%02d.pdf' % frameIndex
      plt.savefig(filename)
      print 'saved file to %s' % filename
else:
  plt.show()
"""

#EOF

