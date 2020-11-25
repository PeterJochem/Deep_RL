# Description
This directory has everything required for comparing the neural network ground reaction model to the DEM data and the RFT data.  

# DEM Dataset For DEM/RFT/Neural Net Comparison
The dataset Juntao and Dan used over the summer should go in the dataset folder. There are 4-5 simulations worth of DEM simulations with each one having the plate hit the ground at diffrent speeds. The plate is not driven or forced into the granular material. The dataset files are too big for me to commit because of Github's size limits. I zipped the folder and put it in my google drive. Here is the link to the folder, https://drive.google.com/file/d/1bKOJLMaauMwVOnheWvTxcKUSMF-ULI9f/view?usp=sharing  

# DEM Dataset for Training the Neural Net 
I created a dataset to train the neural network on. The DEM simulations drive the plate through the granular material at speeds ranging from 1.0 cm/s to 100.0 cm/s. This dataset is too large for me to put on Github because of its large file size limits. The link to the dataset is [here](https://drive.google.com/file/d/1GkRHntBAKGFLWmFBmqRF3KvPMmSSSJs4/view?usp=sharing)  

# How to Recreate My Graphs
## Train the Models
Running ```python3 processData.py``` will read in the dataset, create a neural network, and learn from it. The neural network is saved as ```model.h5```. To use the neural network, open the groundReaction.m file and put in the new path to the model.h5 file.      

## Compare the Models
In Matlab, run <br /> 
```init_env.m```, ```init_params.m``` then run ```eval_foot_gpm("path_to_simulation_data")``` <br />

The path_to_simulation_data is the global path to one of the directories in the dataset/unforced_gpm/ directories. For example, on my computer, in Matlab, I would run ```eval_foot_gpm("/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/validateModel/DEM_RFT_Comparisons/dataset/unforced_gpm/data_set/v_minus_5")```. This will do the comparisons of both models and the ground truth simulation data for a single simulation of the experiment. Each experiment uses a diffrent velocity and so you need to run ```eval_foot_gpm("path_to_simulation_data") ``` on each directory in dataset/unforced_gpm/data_set/ to view how the model performs across a wide range of velocities. 

# Results
The dataset in this directory has a few sub-directories. Each one has the foot intrude into the granular material at a different initial speed. The foot is not forced into the material as it collides. For each trajectory, we plot the ground reaction force-x, ground reaction force-z, and moment as the DEM data says it is (yellow), RFT calculates it (blue), and how the neural network computes it (orange). The RFT and neural network calculations are not used to influence the foot's trajectory, we are simply recomputing what the ground reactions forces and moments are at each time step given the state of the foot in the DEM simulation. <br />     

The neural network is trained on a dataset 

Below are the results of eval_foot_gpm on each trajectory from the DEM data.  

![Initial Velocity = -1 cm/s](media/velocity_minus_1_results.png) <br />
An initial velocity of -1 cm/s <br />

![Initial Velocity = -2 cm/s](media/velocity_minus_2_results.png) <br />
An initial velocity of -2 cm/s <br />

![Initial Velocity = -5 cm/s](media/velocity_minus_5_results.png) <br />
An initial velocity of -5 cm/s <br />

![Initial Velocity = -10 cm/s](media/velocity_minus_10_results.png) <br />
An initial velocity of -10 cm/s <br />

![Initial Velocity = -20 cm/s](media/velocity_minus_20_results.png) <br />
An initial velocity of -20 cm/s <br />

![Initial Velocity = -30 cm/s](media/velocity_minus_30_results.png) <br />
An initial velocity of -30 cm/s <br />

![Initial Velocity = -40 cm/s](media/velocity_minus_40_results.png) <br />
An initial velocity of -40 cm/s <br />

![Initial Velocity = -50 cm/s](media/velocity_minus_50_results.png) <br />
An initial velocity of -50 cm/s <br />


# Files in Repo 
```eval_foot_gpm.m```: Highest level Matlab code for comparing the DEM data, RFT models, and neural network models of the ground. <br />   

```groundReactionModel1.m```: Matlab class for a neural network mapping [gamma, beta, depth] -> [ground reaction force x, ground reaction force z, torque] <br />

```groundReactionModel2.m```: Matlab class for a neural network mapping [gamma, beta, depth, velocity_x, velocity_z] -> [ground reaction force x, ground reaction force z, torque] <br /> 

```init_env.m```: Matlab file that must be run before running eval_foot_gpm.m <br />

```init_params.m```: Matlab file that must be run before running eval_foot_gpm.m <br />

```README.md```

     
