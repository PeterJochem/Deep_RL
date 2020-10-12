import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import pybullet_data
import keras

absolute_path_urdf = "/home/peter/Desktop/HoppingRobot_Fall/RL/gym-hopping_robot/gym_hopping_robot/envs/hopping_robot/urdf/hopping_robot.urdf"

class HoppingRobotEnv(gym.Env):       
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        # Two environments? One for graphics, one w/o graphics?
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        
        self.jointIds=[]
        self.paramIds=[]

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0, 0, -10)

        p.loadURDF("plane.urdf")
        self.hopper = p.loadURDF(absolute_path_urdf, [0.0, 0.0, 1.5], useFixedBase = False)

        print("The number of joints on the hopping robot is " + str(p.getNumJoints(self.hopper)))
        
        # Setup the debugParam sliders
        # Why -10, 10, -10
        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10) 
        self.homePositionAngles = [0.0, 0.0, 0.0]
        
        
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
        

        # What exactly does this do?
        p.setRealTimeSimulation(0)
        
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

    def goToHomePosition(self):

        activeJoint = 0
        for j in range (p.getNumJoints(self.hopper)):

            # Why set the damping factors to 0?
            p.changeDynamics(self.hopper, j, linearDamping = 0, angularDamping = 0)
            info = p.getJointInfo(self.hopper, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.hopper, j, self.homePositionAngles[activeJoint])
                activeJoint = activeJoint + 1

    """Check for controller signals. Allows user to use the GUI sliders
    Do I need this to programatically control robot????"""
    # controlSignal is the list of 1 joint angle 
    def controller(self, controlSignal):

        for i in range(len(self.paramIds)):
            nextJointId = self.paramIds[i]
            #targetPos = p.readUserDebugParameter(nextJointId) # This reads from the sliders
            targetPos = controlSignal[i] # This uses the control signal parameter
            p.setJointMotorControl2(self.hopper, self.jointIds[i], p.POSITION_CONTROL, targetPos, force = 100.0)
    
    """Return the robot to its initial state"""
    def reset(self):

        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        self.jointIds=[]
        self.paramIds=[]
        p.removeAllUserParameters() # Must remove and replace
        self.defineHomePosition()
        p.restoreState(self.stateId)
        #self.goToHomePosition()
    
        return self.computeObservation()

    def step(self, action):
        
        # Forward prop neural network to get GRF, use that to change the gravity
        # FIX ME
        p.getCameraImage(320, 200)
        p.setGravity(0, 0, p.readUserDebugParameter(self.gravId))

        # Step forward some finite number of seconds or milliseconds
        self.controller(action[0])
        for i in range (10):
                p.stepSimulation()

                #time.sleep(1.0/240.0)
                # self.cubePos, self.cubeOrn = p.getBasePositionAndOrientation(self.hopper)
                #time.sleep(0.001)

    
        # observation = list of joint angles

        return self.computeObservation(), self.computeReward(), self.checkForEnd(), None
        # what is info?
        #return observation, reward, done, info
    
    def getFootState(self):
            
        foot_joint_index = 1 # True for old robot but not the new one
        foot_position, foot_velocity, foot_reaction_forces, appliedTorque = p.getJointStates(self.hopper, [foot_joint_index])[0]

        #print(p.getJointStates(self.hopper, [foot_joint_index]))
        
        return foot_position, foot_velocity, foot_reaction_forces, appliedTorque
        

    def computeObservation(self):

        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)
        x, y, z = self.robot_position
        
        foot_angle, foot_velocity, foot_reaction_forces, appliedTorque = self.getFootState() 

        # Should I give it the x, y, z too?
        return [foot_angle, roll, pitch, yaw]  


    def checkForEnd(self):

        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)

        # could also check the z coordinate of the robot?
        if (abs(roll) > (1.0) or abs(pitch) > (1.0)):
            #input() 
            self.isOver = True
            return True

        return False

         
    def render(self, mode='human', close = False):
        pass
    
    """Read the state of the simulation to compute 
    and return the reward scalar for the agent"""
    def computeReward(self):
        
        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.hopper)
        x, y, z = self.robot_position

        # Convert quaternion to Euler angles
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)

        # Reward is inversely proportional to the rotation about the y axis (pitch), and x angle (roll)
        # Reward is proportional to the forward motion in the +x direction
        # FIX ME - which axis should we walk on?
        if (roll == 0.0):
            roll = 0.001
        if (pitch == 0.0):
            pitch = 0.001
        return ((-1.0/roll) + (-1.0/pitch))/1000.0 

            




