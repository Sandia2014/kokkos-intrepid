import numpy as np
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

prefix = 'rl125p216t16/ContractFieldFieldScalar_'
outputPrefix = 'figures/'
suffix = '_clearCache_shadowfax'

numberOfCells = np.loadtxt(open(prefix + 'numberOfContractions' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
dotProductSize = np.loadtxt(open(prefix + 'contractionSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

serialTimes = np.loadtxt(open(prefix + 'serialTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosOmpTimes = np.loadtxt(open(prefix + 'kokkosOmpTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaIndependentTimes = np.loadtxt(open(prefix + 'kokkosCudaIndependentTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTeamReductionTimes = np.loadtxt(open(prefix + 'kokkosTeamReductionTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTiling = np.loadtxt(open(prefix + 'kokkosTilingTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)



allTimes = [serialTimes, kokkosOmpTimes, kokkosCudaIndependentTimes,
            kokkosCudaTeamReductionTimes, kokkosCudaTiling]

labels = ["Serial", "Kokkos Omp", "Kokkos Cuda Flat Parallel", 
          "Kokkos Cuda Team Reduction", "Kokkos Cuda Tiling"]

colors = cm.gist_ncar(np.linspace(1, 0, len(allTimes)))

fig3d = plt.figure(0)
fig2d = plt.figure(1, figsize=(14, 6))
ax2d = plt.subplot(111)
box2d = ax2d.get_position()
ax2d.set_position([box2d.x0, box2d.y0, box2d.width * 0.60, box2d.height])
bbox_to_anchor2d = (1.87, 0.5)

#log plotsfor useCase in xrange(len(numberOfCells)):
plt.cla()
plt.plot(numberOfCells, serialTimes,
             numberOfCells, kokkosOmpTimes,
             numberOfCells, kokkosCudaIndependentTimes,
             numberOfCells, kokkosCudaTeamReductionTimes,
             numberOfCells, kokkosCudaTiling)


plt.xscale('log')
plt.yscale('log')
plt.title('raw times for use case number %d' % 3, fontsize=16)
plt.xlabel('number of cells', fontsize=16)
plt.ylabel('raw time [seconds]', fontsize=16)
plt.xlim(numberOfCells[0], numberOfCells[-1])

ax2d.legend(labels, loc='center right', bbox_to_anchor=bbox_to_anchor2d)

filename = outputPrefix + "FixedSize_RawTimes_2d_UseCase" + str(3) + suffix
plt.savefig(filename + ".pdf")

print "saved file %s" % filename


#linear plotsfor useCase in xrange(len(numberOfCells)):
plt.cla()
plt.plot(numberOfCells, serialTimes,
             numberOfCells, kokkosOmpTimes,
             numberOfCells, kokkosCudaIndependentTimes,
             numberOfCells, kokkosCudaTeamReductionTimes,
             numberOfCells, kokkosCudaTiling)


plt.xscale('linear')
plt.yscale('linear')
plt.title('raw times for use case number %d' % 3, fontsize=16)
plt.xlabel('number of cells', fontsize=16)
plt.ylabel('raw time [seconds]', fontsize=16)
plt.xlim(numberOfCells[0], numberOfCells[-1])

ax2d.legend(labels, loc='center right', bbox_to_anchor=bbox_to_anchor2d)

filename = outputPrefix + "FixedSize_RawTimes_Linear2d_UseCase" + str(3) + suffix
plt.savefig(filename + ".pdf")

print "saved file %s" % filename
'''
summArray = numberOfCells
for timesArray in allTimes:
    summArray = np.vstack((summArray, timesArray))

    allLabels = np.asarray(["Number of Cells"] + labels)
    summArray = np.column_stack([allLabels, summArray])

    filename = "data/FixedSize_UseCase" + str(3) + "_summary.csv"
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(summArray)

print "saved file %s" % filename
'''


# logplot, limited usecase 3
plt.cla()
plt.plot(numberOfCells, serialTimes,
             numberOfCells, kokkosCudaIndependentTimes,
             numberOfCells, kokkosCudaTiling)


plt.xscale('log')
plt.yscale('log')
#plt.title('Raw Times Use Case #%d' % 3, fontsize=16)
plt.xlabel('Number of Contractions', fontsize=24)
plt.ylabel('Time [seconds] (Log10 scale)', fontsize=24)
plt.xlim(numberOfCells[0], numberOfCells[-1])

ax= plt.axes()

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)

plt.tight_layout() 
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(14,10)


localLabels = ["Serial", "Flat Parallel", "Shared Memory"]
ax2d.legend(localLabels, loc=4)
#plt.gcf().subplots_adjust(bottom=0.15)

filename = outputPrefix + "LimitedFixedSize_RawTimes_2d_UseCase" + str(3) + suffix
plt.savefig(filename + ".pdf")

print "saved file %s" % filename


