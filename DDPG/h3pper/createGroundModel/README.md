# Description
This directory contains everything needed to generate and validate a model of a robot's foot interacting with soft ground. It includes directions for generating datasets of robotic feet intruding into granular material, processing the data into a neural network, and validating the model. Our goal is to take the DEM simulation data and use it to learn what ground reaction forces and torques a robot's foot experiences as it intrudes through the granular material. This allows us to have a computationally tractable model of how the ground affects the foot which is then used in motion planners.    

# Model Validation
Over the summer, Dan and Juntao created a suite of tests for comparing the DEM simulations to the resistive force theory (RFT) calculations. I used this setup for comparing the neural network representation of the ground to the DEM data, and RFT model. The code and graphs for validating the model can be found in the ```validateModel``` directory. I included the code that Dan and Juntao wrote but with the modifications I made to it. The reason for this is two fold. I am on a slighly newer version of Matlab and needed a few small changes to duplicate their earlier work. I also added sections to the code for comparing the neural network to the existing models. <br /> 
I also wrote code to numerically solve the ODE's governing the foot's dynamics. From this, we can recover the foot's path over time. More details can be found in the ```validateModel``` directory.             

# Generating Data
Juntao developed DEM simulations of a rigid plate intruding into granular material. I modified his code and used it for generating a dataset that spans the input space evenly. I added bash scripts that allow you to specify the ranges and discretization levels of each input parameter. The code for generating a new dataset is in the ```generateData``` directory. To change the input space parameters and discretization levels requires only changing a few charachters in a single bash file. More details can be found in the ```generateData``` directory.     


# How to Setup Environment
For Matlab to be able to read the .h5 file and re-create the neural network, you need to have the Deep Learning Toolbox installed. In Matlab, run ```importKerasNetwork``` and if the Deep Learning Toolbox is not installed, you will get a link to download it. You may need to run Matlab as root in order to download this (https://www.mathworks.com/matlabcentral/answers/99067-why-do-i-receive-license-manager-error-9). I used Matlab2020B for the development of the code. Dan has used the ground models on 2018b though too. <br />
I also highly recommend working from a virtual environment in Python. The instructions for setting up the virtual environment are at the bottom of this README.md. Keras is very sensitive to the environment and the versions of other libraries on your computer.


# Documentation for Matlab's Keras Network Import
Matlab has a really useful tool for importing neural networks. [Keras](https://keras.io/) is a Python machine learning library that I used to train neural networks. It allows you to save a network's architecture and weights as a .h5 file. You can then use Keras in Python to rebuild the network later. Matlab's Keras Network Importing library allows you to convert a .h5 file into Matlab's neural network ecosystem. This setup resulted in much faster code than trying to have Matlab call Python. More details about Matlab's interface to Keras can be found [here](https://www.mathworks.com/help/deeplearning/ref/importkerasnetwork.html)



# How to Build on Top of My Code

# Ground Reaction Models
I made two independent Matlab classes representing the ground. The first maps [gamma, beta, depth] -> [ground reaction force x, ground reaction force z]. This class uses groundReactionForceModel1.m and model1.h5. The second maps [gamma, beta, depth, velocity_x, velocity_z, theta_dt] -> [ground reaction force x, ground reaction force z, torque about y-axis] and uses groundReactionForceModel2.m and model2.h5. <br />

### Input Space Description
gamma (rads): The angle at which the foot is intruding into the ground. <br /> 
beta (rads) - The foot's orientation in the x-z plane <br />
depth (m) - The foot's depth into the granular material <br />
theta_dt (rads/s) - The rate of change of the foot-leg joint angle <br /> 
velocity_x (m/s) - The foot's velocity in the x-direction <br />
velocity_z (m/s) - The foot's velocity in the z-direction <br /> 
<br />
### Output Space Description
ground reaction force x (N) - The force in the x-direction that the ground exerts on the foot <br />
ground reaction force z (N) - The force in the z-direction that the ground exerts on the foot <br />

## Units
I used standard SI units, and radians for angles. Sidenote: the Chrono simulations default to CGS

# Notes on Setting Up the Virtual Environment
Running these commands will create a Python virtual environment for you <br />
```python3 -m venv env_name``` <br /> 
```source env_name/bin/activate``` <br />
```pip3 install --upgrade pip``` <br />
```python3 -m pip install --upgrade setuptools``` <br />
```pip install keras``` <br />
```pip install pybullet```
```pip install numpy```

# Activating the Virtual Environment 
```source env_name/bin/activate``` <br />

 


