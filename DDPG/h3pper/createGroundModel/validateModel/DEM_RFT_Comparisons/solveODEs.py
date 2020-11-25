from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl
import numpy as np
import random
import keras
import math

#absolute_path_to_model = "/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/visualizeData/validateChrono/validation_model.h5" 
#absolute_path_to_model = "/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/visualizeData/validateChrono/mlab_data/model.h5"
absolute_path_to_model = "/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/visualizeData/validateChrono/mlab_data/model2.h5"
neural_net = keras.models.load_model(absolute_path_to_model)

def flat_plate_RHS(state, t, m, I):

    x         = state[0]
    dx_dt     = state[1]
    z         = state[2]
    dz_dt     = state[3]
    theta     = state[4]
    dtheta_dt = state[5] 
    
    #depth = gran_max_z - z
    depth = -0.0125 + (z) # abs(z) Matlab starts right when plate is on the top of granular
    #depth = 0.1145 + (-1 * z)
    #depth = z
    #print(depth) - dynamics time stepping can make this misleading
    gamma = math.atan2(dz_dt, dx_dt) # Sign?
    beta = theta #theta % (np.pi)
    
    #inVector = [gamma, beta, depth]
    inVector = [gamma, beta, depth, dx_dt, dz_dt, dtheta_dt]

    F_x, F_z, Torque = neural_net.predict(([inVector]))[0]
    Torque = Torque

    # the next three lines are almost a truism, but they're standard when expressing
    # any mechanical system (based on F=ma, so a system of "n" 2nd-order ODEs)
    # into a system of 2*n 1st-order ODEs"
    #dx_dt     = dx
    #dy_dt     = dy
    #dtheta_dt = dtheta
    
    d2x_dt2     = F_x/m 
    d2z_dt2     = F_z/m - (9.81)  
    d2theta_dt2 = Torque/I 

    return [dx_dt, d2x_dt2, dz_dt, d2z_dt2, dtheta_dt, d2theta_dt2]

# Pick initial conditions and pack them into a list:
x_0 = 0.0
velocity_x_0 = 0.0050 

z_0 = 0.0125 
velocity_z_0 = -0.0185 # -1.0 cm/s

theta_0 = 0.523599  
angular_velocity_0 = 0.0

state0 = [x_0, velocity_x_0, z_0, velocity_z_0, theta_0, angular_velocity_0]

# Define a time window for simulation:
t0 = 0
tF = 1.0
Npts = 1000
tList = np.linspace(t0, tF, Npts)

I = 0.0000520833  
m = 0.25 

sol = odeint(flat_plate_RHS, state0, tList, args=(m, I))

print("The solution is\n", sol)
plt.rcParams['figure.figsize'] = [16,10]

plt.plot(tList,sol[:,0] * 100,'r',label='$x(t)$')
plt.plot(tList,sol[:,1] * 100,'b',label='$xdt(t)$')
plt.plot(tList,sol[:,2] * 100,'g',label='$z(t)$')
plt.plot(tList,sol[:,3] * 100,'orange',label='$zdt(t)$')
plt.plot(tList,sol[:,4],'purple',label='$theta(t)$')
plt.plot(tList,sol[:,5],'magenta',label='$thetadt(t)$')

plt.legend(loc='best')
plt.xlabel('$t$')
plt.grid()
plt.show()

mpl.rcParams['legend.fontsize'] = 10

"""
fig = plt.figure()
ax0 = fig.gca(projection='3d')
labelStr = ('beta = {:.2f}, rho = {:.2f}, sigma = {:.2f}'.format(beta,rho,sigma))
ax0.plot(sol[:,0],sol[:,1],sol[:,2],label=labelStr)
ax0.legend()
plt.show()
"""

