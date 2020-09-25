import numpy as np
import time
import random
import sys
import gym
import warnings
from PIL import Image
import cv2
from experience import *
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

allReward = [] # List of rewards over time, for logging and visualization

"""Signal handler displays the agent's progress"""
def handler1(signum, frame):

    # Plot the data
    plt.plot(allReward)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()

    myAgent.epsilon_min = 0.00001

signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.discount = 0.95
        self.memorySize = 5000
        self.replay_sample_size = 2000 # How many experiences to sample from replay when we train

        self.epsilon = 1.0 # How often to explore        
        self.epsilon_decay = 0.95 # After each training, rate of decay of epsilon
        self.epsilon_min = 0.1 # Don't let the epsilon be less than this
        self.batch_size = 64 # Mini batch size for keras .fit method

        self.trainRate = 450 # After how many moves should we train? 
        self.moveNumber = 0 # Number of actions taken in the current episode 
        self.synchronize = 15 # The number of episodes before synchronizing networks
        self.save = 50 # This describes how often to save each network to disk  
        self.epochs = 1 # Epochs to train on a batch of data for neural network 

        self.replayMemory = replayBuffer(self.memorySize)
        
        self.currentGameNumber = 0 
        self.cumulativeReward = 0 # Current game's total reward

        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()

        self.action_space = self.env.action_space.n # Number of actions agent can take 

        self.Q_online = Sequential()
        self.Q_online.add(Dense(24, input_dim = 4, activation='relu'))
        self.Q_online.add(Dense(24, activation='relu'))
        self.Q_online.add(Dense(2, activation='linear'))
        self.Q_online.compile(loss = "mse",  optimizer = Adam(lr = 0.001))

        self.Q_target = copy.deepcopy(self.Q_online)

    def updateNetworks(self):
        self.Q_target = copy.deepcopy(self.Q_online)       
         
    
    # Be careful with nextState - is that really what you want?
    def target_value(self, current_state, next_state, immediateReward, done):
        """ Use the Bellman equation to compute the optimal
        action and its predicted cumulative reward """
        
        if (done):
            return -100.0

        next_action = np.argmax(self.Q_online.predict(np.array([next_state])))
    
        return immediateReward + (self.discount * self.Q_target.predict(np.array([next_state]))[0][next_action])
          
    def train(self):
        
        training_data = self.replayMemory.sample(self.replay_sample_size)
        
        inputs = [""] * len(training_data)
        labels = [""] * len(training_data)

        for i in range(len(training_data)):
            if (training_data[i] != None):
                if (training_data[i].state is not None):
                    
                    state, next_state, reward, done, action = training_data[i].unpack()    

                    predicted_values = self.Q_target.predict(np.array([state]))[0]
                    predicted_values[action] = self.target_value(state, next_state, reward, done)

                    inputs[i] = state
                    labels[i] = predicted_values

        inputs = np.array(inputs)
        labels = np.array(labels)
        history = self.Q_online.fit(inputs, labels, batch_size = self.batch_size, epochs = self.epochs, verbose = 0)
              
        
    def saveNetworks(self):
        self.Q_online.save("neural_networks/online")
        self.Q_target.save("neural_networks/target")

    def handleGameEnd(self):
        
        print("Game number " + str(self.currentGameNumber) + " ended with a total reward of " + str(self.cumulativeReward))
        self.env.reset()
        allReward.append(self.cumulativeReward)
        self.cumulativeReward = 0
         
        # Check if it is time to update/synchronize the two networks
        if ((self.currentGameNumber + self.synchronize + 1) % self.synchronize == 0):
            print(self.currentGameNumber)
            print("Synchronized networks")
            myAgent.updateNetworks()
          
        self.currentGameNumber = self.currentGameNumber + 1


myAgent = Agent()
    
# Randomnly choose your first action
current_state, reward, done, info = myAgent.env.step(0)

while(True):
    myAgent.env.render()
    
    predicted_values = myAgent.Q_online.predict(np.array([current_state]))[0]
      
    action = np.argmax(predicted_values)
    
    if (random.random() < myAgent.epsilon):
        action = random.randint(0, myAgent.action_space - 1)
       
    next_state, reward, done, info = myAgent.env.step(action)
        
    currentExperience = experience(current_state, next_state, action, reward, done)
    
    # Create the experience
    target_value = myAgent.target_value(current_state, next_state, reward, done)
    
    predicted_values = myAgent.Q_target.predict(np.array([current_state]))[0]
    currentExperience.actions_values = predicted_values
    currentExperience.actions_values[action] = target_value
    
    myAgent.replayMemory.append(currentExperience)

    current_state = next_state

    # Update counters
    myAgent.moveNumber = myAgent.moveNumber + 1
    myAgent.cumulativeReward = myAgent.cumulativeReward + reward
    
    if (done):
        myAgent.handleGameEnd()
          

    # Check if it is time to train
    if ((myAgent.moveNumber + myAgent.trainRate) % myAgent.trainRate == 0):
        print("training")
        myAgent.epsilon = myAgent.epsilon * myAgent.epsilon_decay 
        if (myAgent.epsilon < myAgent.epsilon_min):
            myAgent.epsilon = myAgent.epsilon_min

        myAgent.train()

    # Check to save the network for later use
    if ((myAgent.currentGameNumber + myAgent.save) % myAgent.save == 0):    
        pass
        # myAgent.saveNetworks()
    






