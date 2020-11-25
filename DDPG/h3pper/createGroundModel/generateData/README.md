# Description
This directory has everything we need to generate the dataset with the plate interacting with the granular material across the entire input space we care about. ```Intrude``` has the setup for the plate intruding. ```extract``` has the setup for the other movements of the plate. <br />
The code here allows you to specify the range and discretization level of each part of the input space we care about and then run one program to generate the dataset. There are two sub-directories. One for the plate intruding and one for the extracting of the plate. There is a bash file to run for each directory that will generate a dataset over the specified input space. To combine the two datasets, run ```createOneDataset.py```    

# Files in Repo
```foot.obj```: The Blender file describing the foot <br />   
```createOneDataSet.py```: Combines the csv files from the intrude and extract folders

# Sub-Directories:
```intrude```: Generates dataset over the speceified input space of the plate intruding <br /> 
```extract```: Generates dataset over the speceified input space of the plate being extracted 

