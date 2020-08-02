import numpy as np
import time
import random
import sys
import gym
import warnings
from PIL import Image
import cv2
from dataInstance import *
from observation import *
import copy
import signal, os
from replayBuffer import *

warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    from keras.layers import Activation, Dense, Flatten, Conv2D
    from keras.models import Sequential
    from keras.optimizers import RMSprop, Adam
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def handler1(signum, frame):

    print("The ctrl-z (sigstp) signal handler ran!")
    print("Here we should save the neural network and show what the agent has learned")
    # Add another boolean to switch back into training mode after pausing to view progress


signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.downSizedLength = 64
        self.downSizedWidth = 64
    
        self.discount = 0.95
        self.memorySize = 1000

        self.replayBuffer = replayBuffer(self.memorySize)

        # These are the most recently seen observations 
        self.observationT1 = None
        self.observationT2 = None
        self.observationT3 = None
        self.observationT4 = None
        self.observationT5 = None
    
        # Creates enviroment from OpenAI gym
        self.env = gym.make('Breakout-v0')
        self.state = self.env.reset()

        # There are 4 actions for the Brickbreaker enviroment
        self.action_space = 4

        


        # Define the two neural networks for Double Deep Q Learning
        self.QA = Sequential()
        self.QA.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (64, 64, 4) ) )
        self.QA.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
        self.QA.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
        self.QA.add(Flatten())
        self.QA.add(Dense(self.action_space, activation = 'relu') )

        self.QA.compile(loss = "mean_squared_error",
        optimizer = RMSprop(lr = 0.00025,
        rho = 0.95,
        epsilon = 0.01),
        metrics = ["accuracy"])

        self.QB = copy.deepcopy(self.QA)

    
    # In order to make room for the next observation, move
    # back each one in the "queue"
    def reorderObservations(self):
    
        self.observationT5 = self.observationT4
        self.observationT4 = self.observationT3
        self.observationT3 = self.observationT2
        self.observationT2 = self.observationT1
 
    # Generate the first three frames so we can perform an action
    # We need 4 frames to use the nueral network
    # Input: void
    # Return: void
    def generateFirst3Frames(self):

        self.env.render()
        action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT3 = observation(state)
        
        self.env.render()
        self.action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT2 = observation(state)

        self.env.render()
        action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT1 = observation(state)

    def updateNetworks(self):
        QB = copy.deepcopy(self.QA)        
    
    
    def target_value(self, nextState, immediateReward):
        """ Use the Bellman equation to compute the optimal
        action and its predicted cumulative reward """
        
        # The network predicts the discounted value from that state
        # so predict the value for each state and then take the largest index with argmax
        action = np.argmax( self.QA.predict(nextState) )
            
        # print(self.QB.predict(nextState) )
    
        y_target = immediateReward + (self.discount * (self.QB.predict(nextState))[0][action] )
           
        return y_target
        
        

myAgent = Agent()

# Box(210, 160, 3)
# print(env.observation_space)

# Generate the first three frames so we can perform an action
# We need 4 frames to use the neural network
myAgent.generateFirst3Frames()

while(True):
    myAgent.env.render()
    action = myAgent.env.action_space.sample()
    state, reward, done, info = myAgent.env.step(action)
    
    myAgent.reorderObservations()
    myAgent.observationT1 = observation(state) 
    
    # Turn most recent 4 observations into a training instance  
    myDataInstance = dataInstance(myAgent.observationT1, myAgent.observationT2, myAgent.observationT3, myAgent.observationT4)
    # myDataInstance.viewAllImages()
    
    prediction = myAgent.QA.predict( np.array( [myDataInstance.data] ) )

    # Label the data and put into the replay buffer
    myDataInstance.value = myAgent.target_value( np.array( [ myDataInstance.data ] ), reward) 
    
    myAgent.replayBuffer.append(myDataInstance)
    

    # print( np.argmax(prediction) )








