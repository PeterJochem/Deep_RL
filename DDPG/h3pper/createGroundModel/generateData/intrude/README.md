# Description
This directory has the code required for creating a dataset of the foot interacting with the granular material as it is being driven down into the material.

# How to Compile
Copy the folder over to Kronos, and run ```mkdir build```, ```mkdir logs```, ```cd build```, and then ```cmake ..```


# Files in Repo  
```createDataSet.sh```: Highest level file. Runs intrude.cpp many times with diffrent command line args in order to generate data over the entire input space we care about. <br />
```intrude.cpp```: DEM simulation of the plate intruding into the granular material <br />
```setup.json```: Defines parameters of the DEM simulation <br /> 
```CMakeLists.txt```: For build system <br />
```README.md```

