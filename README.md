# Kokkos applied to Intrepid's ArrayTools

## Background
This repository contains the work of the Harvey Mudd College Sandia 2014 clinic team. In this repository are nine folders corresponding to the nine Contraction operations provided in Intrepid's ArrayTools. For more information on what each contraction does, see the Sandia 2014 clinic team's final report.

## Folder Structure
Each folder corresponds to a single contraction function. Within the folder are a number of files:
 - **A file with the same names as the folder and a .cu or .cpp ending.** </br>
  This file contains the testing code that was used by the Sandia 2014 clinic team. Unless you are very interested in the testing methodology, it should be ignored.
 - **1+ .hpp files** </br>
 These files contain variations on Kokkos and Cuda kernels for the function. Each kernel should be well commented. For additional information, see the Sandia 2014 final report.
 - **generatePlots.py** </br>
 This is Python code that uses matplotlib to generate plots using the data collected by our testing code. For more information, see the 'Building and Running' section below.
 - **A Makefile** </br>
 This compiles the C++ code. For more information, see the 'Building and Running' section below.
 - **A data folder** </br>
 This folder contains the raw .csv data generated by the .cpp/.cu file.
 - **A figures folder** </br>
 This folder contains the .png files generated by generatePlots.py
 
## Building and Running Tests
The testing file in any given folder can be build by running the command 
```
make
```
from the command line in the appropriate folder.

In order to then run the tests, the command
```
./<folder-name>
```
should suffice. Note that if the `data` folder does not exist, the tests will fail. Some of the data folders were not added properly to the repo, so they need to be created before tests can be run.

Finally, in order to make figures using the data created, you can run the command
```
python generatePlots.py
```
in order to generate figures, which can then be viewed using your favorite .png viewer.
