# Description
I made two independent Matlab classes (located in ```use_ground_models```) representing the ground. The first maps [gamma, beta, depth] -> [ground reaction force x, ground reaction force z]. This class uses groundReactionForceModel1.m and model1.h5. The second maps [gamma, beta, depth, velocity_x, velocity_z, theta_dt] -> [ground reaction force x, ground reaction force z, torque about y-axis] and uses groundReactionForceModel2.m and model2.h5. More details are in the ```use_ground_models``` folder. <br />

