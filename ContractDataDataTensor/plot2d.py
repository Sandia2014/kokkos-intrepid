
import numpy as np
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv



prefix = 'data/ContractDataDataTensor_'
outputPrefix = 'figures/'
suffix = '_clearCache_shadowfax'

numberOfCells = np.loadtxt(open(prefix + 'numberOfDotProducts' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
dotProductSize = np.loadtxt(open(prefix + 'contractionSize' + suffix + '.csv','rb'),delimiter=',',skiprows=0)

serialTimes = np.loadtxt(open(prefix + 'serialTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosOmpTimes = np.loadtxt(open(prefix + 'kokkosOmpTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosOmpTeamSize2Times = np.loadtxt(open(prefix + 'kokkosOmpTeamSize2Times' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosOmpTeamSize12Times = np.loadtxt(open(prefix + 'kokkosOmpTeamSize12Times' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaIndependentTimes = np.loadtxt(open(prefix + 'kokkosCudaIndependentTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTeamDepth1Times = np.loadtxt(open(prefix + 'kokkosCudaTeamDepth1Times' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTeamDepth2Times = np.loadtxt(open(prefix + 'kokkosCudaTeamDepth2Times' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTeamDepth3Times = np.loadtxt(open(prefix + 'kokkosCudaTeamDepth3Times' + suffix + '.csv','rb'),delimiter=',',skiprows=0)
kokkosCudaTeamStrideTimes = np.loadtxt(open(prefix + 'kokkosCudaTeamStrideTimes' + suffix + '.csv','rb'),delimiter=',',skiprows=0)


allTimes = [serialTimes, kokkosOmpTimes, kokkosCudaIndependentTimes,
            kokkosCudaTeamDepth1Times, kokkosCudaTeamDepth2Times,
            kokkosCudaTeamDepth3Times, kokkosCudaTeamStrideTimes,
            kokkosOmpTeamSize2Times, kokkosOmpTeamSize12Times]

labels = ["Serial", "Kokkos Omp", "Kokkos Cuda Flat Parallel",
          "Kokkos Cuda Team Depth 1", "Kokkos Cuda Team Depth 2",
          "Kokkos Cuda Team Depth 3", "Kokkos Cuda Team Stride",
          "Kokkos Omp Team Size 2", "Kokkos Omp Team Size 12"]

colors = cm.gist_ncar(np.linspace(1, 0, len(allTimes)))

fig3d = plt.figure(0)
fig2d = plt.figure(1, figsize=(14, 6))
ax2d = plt.subplot(111)
box2d = ax2d.get_position()
ax2d.set_position([box2d.x0, box2d.y0, box2d.width * 0.60, box2d.height])
bbox_to_anchor2d = (1.87, 0.5)

#log plots
for useCase in xrange(len(numberOfCells)):
    plt.cla()
    plt.plot(numberOfCells[useCase], serialTimes[useCase],
             numberOfCells[useCase], kokkosOmpTimes[useCase],
             numberOfCells[useCase], kokkosCudaIndependentTimes[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth1Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth2Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth3Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamStrideTimes[useCase],
             numberOfCells[useCase], kokkosOmpTeamSize2Times[useCase],
             numberOfCells[useCase], kokkosOmpTeamSize12Times[useCase])


    plt.xscale('log')
    plt.yscale('log')
    plt.title('raw times for use case number %d' % (useCase + 1), fontsize=16)
    plt.xlabel('number of cells', fontsize=16)
    plt.ylabel('raw time [seconds]', fontsize=16)
    plt.xlim(numberOfCells[useCase][0], numberOfCells[useCase][-1])

    ax2d.legend(labels, loc='center right', bbox_to_anchor=bbox_to_anchor2d)

    filename = outputPrefix + "FixedSize_RawTimes_2d_UseCase" + str(useCase + 1) + suffix
    plt.savefig(filename + ".pdf")

    print "saved file %s" % filename

#limited log plots
for useCase in xrange(len(numberOfCells)):
    plt.cla()
    plt.plot(numberOfCells[useCase], serialTimes[useCase],
             numberOfCells[useCase], kokkosCudaIndependentTimes[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth2Times[useCase])


    plt.xscale('log')
    plt.yscale('log')
    plt.title('Raw Times, Use Case #%d' % (useCase + 1), fontsize=16)
    plt.xlabel('Number of Contractions', fontsize=16)
    plt.ylabel('Time [seconds] (Log10 scale)', fontsize=16)
    plt.xlim(numberOfCells[useCase][0], numberOfCells[useCase][-1])

    _labels = ["Serial", "Kokkos Cuda Flat Parallel", "Kokkos Cuda Team Reduce"]
    ax2d.legend(_labels, loc='center right', bbox_to_anchor=bbox_to_anchor2d)

    filename = outputPrefix + "LimitedFixedSize_RawTimes_2d_UseCase" + str(useCase + 1) + suffix
    plt.savefig(filename + ".pdf")

    print "saved file %s" % filename

#linear plots
for useCase in xrange(len(numberOfCells)):
    plt.cla()
    plt.plot(numberOfCells[useCase], serialTimes[useCase],
             numberOfCells[useCase], kokkosOmpTimes[useCase],
             numberOfCells[useCase], kokkosCudaIndependentTimes[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth1Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth2Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamDepth3Times[useCase],
             numberOfCells[useCase], kokkosCudaTeamStrideTimes[useCase],
             numberOfCells[useCase], kokkosOmpTeamSize2Times[useCase],
             numberOfCells[useCase], kokkosOmpTeamSize12Times[useCase])


    plt.xscale('linear')
    plt.yscale('linear')
    plt.title('raw times for use case number %d' % (useCase + 1), fontsize=16)
    plt.xlabel('number of cells', fontsize=16)
    plt.ylabel('raw time [seconds]', fontsize=16)
    plt.xlim(numberOfCells[useCase][0], numberOfCells[useCase][-1])

    ax2d.legend(labels, loc='center right', bbox_to_anchor=bbox_to_anchor2d)

    filename = outputPrefix + "FixedSize_RawTimes_Linear2d_UseCase" + str(useCase + 1) + suffix
    plt.savefig(filename + ".pdf")

    print "saved file %s" % filename



for useCase in xrange(len(numberOfCells)):
    summArray = numberOfCells[useCase]
    for timesArray in allTimes:
        summArray = np.vstack((summArray, timesArray[useCase]))

    allLabels = np.asarray(["Number of Cells"] + labels)
    summArray = np.column_stack([allLabels, summArray])

    filename = "data/FixedSize_UseCase" + str(useCase + 1) + "_summary.csv"
    with open(filename, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(summArray)

    print "saved file %s" % filename
