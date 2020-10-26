import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import pybullet_data
import keras
import numpy as np
from scipy.spatial.transform import Rotation as R

absolute_path_urdf = "/home/peter/Desktop/Deep_RL/DDPG/h3pper/gym-hopping_robot/gym_hopping_robot/envs/hopping_robot/urdf/hopping_robot.urdf"
absolute_path_neural_net = "/home/peter/Desktop/Deep_RL/DDPG/h3pper/gym-hopping_robot/gym_hopping_robot/envs/hopping_robot/neural_networks/model2.h5"  

class HoppingRobotEnv(gym.Env):       
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        # Two environments? One for graphics, one w/o graphics?
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        
        self.visualizeTrajectory = False 
        self.jointIds=[]
        self.paramIds=[]
    
        self.neural_net = keras.models.load_model(absolute_path_neural_net)

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0, 0, -10)

        p.loadURDF("plane.urdf")
        self.hopper = p.loadURDF(absolute_path_urdf, [0.0, 0.0, 1.5], useFixedBase = False)
        
        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
        self.homePositionAngles = [0.0, 0.0, 0.0]

        # Setup the debugParam sliders
        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10) 
        self.homePositionAngles = [0.0, 0.0, 0.0]
        
        self.foot_points = []
        self.body_points = []
            
        activeJoint = 0
        for j in range (p.getNumJoints(self.hopper)):
            
            # Why set the damping factors to 0?
            p.changeDynamics(self.hopper, j, linearDamping = 0, angularDamping = 0)
            info = p.getJointInfo(self.hopper, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                
                self.jointIds.append(j)
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, self.homePositionAngles[activeJoint]))
                #p.resetJointState(self.hopper, j, self.homePositionAngles[activeJoint])
                activeJoint = activeJoint + 1

    
        p.setRealTimeSimulation(0) # Must do this to apply forces/torques with PyBullet method
        self.stateId = p.saveState() # Stores state in memory rather than on disk
    

    """Reset the robot to the home position"""
    def defineHomePosition(self):

        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10)

        # Why -10, 10, -10
        self.homePositionAngles = [0, 0, 0]

        activeJoint = 0
        for j in range (p.getNumJoints(self.hopper)):

            # Why set the damping factors to 0?
            p.changeDynamics(self.hopper, j, linearDamping = 0, angularDamping = 0)
            info = p.getJointInfo(self.hopper, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):

                self.jointIds.append(j)
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, self.homePositionAngles[activeJoint]))
                p.resetJointState(self.hopper, j, self.homePositionAngles[activeJoint])
                activeJoint = activeJoint + 1
     
    """Update robot's PID controller's control signals. controlSignal is a list of desired joint angles (rads).
    PyBullet does support direct torque control...iterate in this direction eventually?"""
    def controller(self, controlSignal):

        for i in range(len(self.paramIds)):
            nextJointId = self.paramIds[i]
            #targetPos = p.readUserDebugParameter(nextJointId) # This reads from the sliders. Useful for debugging        
            targetPos = controlSignal[i] # This uses the control signal parameter
            p.setJointMotorControl2(self.hopper, self.jointIds[i], p.POSITION_CONTROL, targetPos, force = 50.0) # 100.0
 
    """Return the robot to its initial state"""
    def reset(self):

        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        self.jointIds=[]
        self.paramIds=[]
        p.removeAllUserParameters() # Must remove and replace
        p.restoreState(self.stateId) 
        self.defineHomePosition()
        
        for i in range(len(self.foot_points)):
            p.removeUserDebugItem(self.foot_points[i])
            p.removeUserDebugItem(self.body_points[i])

        return self.computeObservation()

    def computeGRF(self, gamma, beta, depth, dx, dy, dz, ankle_angular_velocity):
        
        inVector = [gamma, beta, depth, dy, dz, ankle_angular_velocity]
        
        # Be careful with how you define the frames
        grf_y, grf_z, torque = (self.neural_net.predict([inVector]))[0]
        
        return grf_y, grf_z, torque
        
    """ After stepping the simulation, force the robot to be in the y-z plane
    A bit too hacky? Well, planar physics is well, hacky anyways""" 
    def planarConstraint(self):
        
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        x, y, z = robot_position
        [dx, dy, dz], [wx, wy, wz] = p.getBaseVelocity(self.hopper) 

        """ Force the base to be in the y-z plane. resetBAsePosandOrient sets the
        velocities to 0, so we must overwrite them with the original velocities. We can translate
        in the y-z plane and rotate about the x-axis"""
        p.resetBasePositionAndOrientation(self.hopper, [0.0, y, z], robot_orientation)
        p.resetBaseVelocity(self.hopper, [0.0, dy, dz], [wx, 0.0, 0.0])

    """Map the bezier point to the world frame and then do IK. Return desired joint angles"""
    def bezierToJointAngles(self, bezier_point):
                    
        bezier_world = self.hip_to_world(bezier_point)
            
        foot_index = 3     
        # Inlude a desired orientation?
        #p.calculateInverseKinematics(self.hopper, foot_index, bezier_position, target_orientation) 
        
        
        return p.calculateInverseKinematics(self.hopper, foot_index, bezier_world)

       
    """Time must be in the interval of [0, 1] """
    def bezierCurve(self, time, points):
        
        t1 = ((1 - time)**3) * points[0]
        
        t2 = (3 * ((1 - time)**2) * time * points[1]) 

        t3 = (3 * ((1 - time)) * time**2 * points[2])
        
        t4 = (time**3) + points[3] 
        
        return t1 + t2 + t3 + t4

    """ Convert a point from the hip frame to the world frame """
    def hip_to_world(self, Point_in_hip_frame):
        
        hip_position, hip_orientation = p.getBasePositionAndOrientation(self.hopper)

        hip_x, hip_y, hip_z = Point_in_hip_frame
        P_Hip = np.zeros((4, 1), dtype="float32")
        P_Hip[0][0] = hip_x 
        P_Hip[0][0] = hip_y
        P_Hip[0][0] = hip_z
        P_Hip[0][0] = 1.0
        
        """Use scipy to convert the quaternion to a rotation matrix"""
        r = R.from_quat(hip_orientation) # scipy uses x, y, z, w
        
        T_world_hip = np.zeros((4, 4), dtype='float32')
        for i in range 3
            for j in range 3:
                T_world_hip[i][j] = r[i][j]
        
        T_world_hip[0][3] = hip_x
        T_world_hip[1][3] = hip_y
        T_world_hip[2][3] = hip_z
        T_world_hip[3][3] = 1.0 

        P_world = np.matmul(T_world_hip, P_hip)  

        return P_world


    def plotPosition(self):
        world_pos, orientation, localInertialFramePosition, localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation, worldLinkLinearVelocity, worldLinkAngularVelocity = p.getLinkState(self.hopper, 0, 1)
        body_x, body_y, body_z = world_pos
        self.foot_points.append(p.addUserDebugLine([foot_x - 0.025, foot_y - 0.025, foot_z - 0.025], [foot_x, foot_y, foot_z], [1, 0, 0]))
        self.body_points.append(p.addUserDebugLine([body_x - 0.025, body_y - 0.025, body_z + 0.2], [body_x, body_y, body_z + 0.2 + 0.025], [0, 0, 1]))


    def step(self, action):
        
        # Forward prop neural network to get GRF, use that to change the gravity
        # Shoud really compute after every p.stepSimulation
        ankle_position, ankle_angular_velocity, appliedTorque, foot_x, foot_y, foot_z, foot_dx, foot_dy, foot_dz, foot_roll, foot_pitch, foot_yaw = self.getFootState()           
        #depth = plate_bottom_z - bed_z
        gamma = np.sqrt(foot_dy**2 + foot_dz**2) 
        beta = foot_roll
        
        # else if (in bed) manually set the grf
        # else if (on bottom) let grf be PyBullet defined
        
        if (self.visualizeTrajectory):
            self.plotPosition()

        customGRF = False
        if (foot_z < 0.3 and foot_z > 0.0001):
            customGRF = True

        grf_y, grf_z, torque = self.computeGRF(gamma, beta, foot_z, foot_dx, foot_dy, foot_dz, ankle_angular_velocity)
          
        # Step forward some finite number of seconds or milliseconds
        self.controller(action[0])
        for i in range (3):
            foot_index = 3     
            # Must call this each time we stepSimulation
            
            if (customGRF):
                p.applyExternalForce(self.hopper, foot_index, [0, grf_y, grf_z], [0.0, 0.0, 0.0], p.LINK_FRAME) 
                p.applyExternalTorque(self.hopper, foot_index, [torque, 0, 0], p.LINK_FRAME)
            
            p.stepSimulation()
            self.planarConstraint()
     
        isOver = self.checkForEnd()
        return self.computeObservation(), self.computeReward(isOver), isOver, None
     
    def getFootState(self):
        
        """Server keeps two lists. One of links and one of joints. These are the indexes into those lists"""
        ankle_joint_index = 3 # Known by printing world frame position of links with p.getLinkState(self.hopper, <index#>)
        foot_link_index = 3

        ankle_position, ankle_angular_velocity, ankle_joint_reaction_forces, appliedTorque = p.getJointStates(self.hopper, [ankle_joint_index])[0]
             
         
        world_pos, orientation, localInertialFramePosition, localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation, worldLinkLinearVelocity, worldLinkAngularVelocity = p.getLinkState(self.hopper, foot_link_index, 1)
        
        foot_roll, foot_pitch, foot_yaw = p.getEulerFromQuaternion(self.robot_orientation)

        foot_x, foot_y, foot_z = world_pos
        foot_dx, foot_dy, foot_dz = worldLinkLinearVelocity

        return ankle_position, ankle_angular_velocity, appliedTorque, foot_x, foot_y, foot_z, foot_dx, foot_dy, foot_dz, foot_roll, foot_pitch, foot_yaw

    def computeObservation(self):

        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        base_roll, base_pitch, base_yaw = p.getEulerFromQuaternion(self.robot_orientation)
        base_x, base_y, base_z = self.robot_position
        
        ankle_position, ankle_angular_velocity, appliedTorque, foot_x, foot_y, foot_z, foot_dx, foot_dy, foot_dz, foot_roll, foot_pitch, foot_yaw = self.getFootState() 

        return [base_roll, base_pitch, base_yaw, base_x, base_y, base_z, ankle_position, ankle_angular_velocity, appliedTorque, foot_x, foot_y, foot_z, foot_dx, foot_dy, foot_dz, foot_roll, foot_pitch, foot_yaw]  

    def checkForEnd(self):

        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)

        x, y, z = self.robot_position

        # could also check the z coordinate of the robot?
        """
        if (abs(roll) > (1.0) or abs(pitch) > (1.0)):
            self.isOver = True
            return True
        """
        if (z < 0.5):
            return True

        return False

    """Required for the OpenAI Gym API""" 
    def render(self, mode='human', close = False):
        pass
    
    """Read the state of the simulation to compute and return the 
    reward scalar for the agent. A great video on reward shaping
    from the legendary control theory youtuber Brian Douglas -> 
    https://www.mathworks.com/videos/reinforcement-learning-part-4-the-walking-robot-problem-1557482052319.html
    Remember, you get what you incentivize, not what you want"""
    def computeReward(self, isOver):

        stillAliveBonus = 0.0625
        
        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        x, y, z = self.robot_position 
        [dx, dy, dz], [wx, wy, wz] = p.getBaseVelocity(self.hopper)
        
        # Remember the actions are the joint angles, not the joint torques
        reward = dy + y  
        if (isOver == False):
            reward = reward + stillAliveBonus 

        return reward

