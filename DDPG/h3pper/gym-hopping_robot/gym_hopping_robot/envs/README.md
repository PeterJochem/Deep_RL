# Description
This is where we actually implement the OpenAI gym environment in PyBullet. There is a description of each file below. To get OpenAI gym integrated with your environment, the file structure has to be this way. 

# hopping_robot_env.py
This implements the class that represents the gym. It implements the step, reset, and render methods that are the interface to the OpenAI gym. It also handles the tracking of the robot's state. 

# hopping_robot.py
This is an example of how to interact and use the PyBullet simulation.

# hopping_robot
This directory has the files for the robot's URDF and meshes 

# plane.mtl, plane.urdf, plane.obj
These are required to implement the plane (ground) in PyBullet  
