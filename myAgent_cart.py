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

allReward = []

def handler1(signum, frame):

    print("The ctrl-z (sigstp) signal handler ran!")
    print("Here we should save the neural network and show what the agent has learned")
    # Add another boolean to switch back into training mode after pausing to view progress

    # Plot the data
    plt.plot(allReward)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()


signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.discount = 0.95
        self.memorySize = 5000 # FIX ME
            
        self.epsilon = 1.0 # How often to explore        
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1 # Don't let the epsilon be less than this
        self.batchSize = 2000

        self.trainRate = 450 # After how many moves should we train? 
        self.moveNumber = 0 # Number of actions taken. Reset periodically 
        self.synchronize = 15 # The number of games to play before synchronizing networks
        self.save = 50 # This describes how often to save each network to disk

        self.replayMemory = replayBuffer(self.memorySize)
        
        self.currentGameNumber = 0

        self.cumulativeReward = 0 # Current game's total reward

        # Creates enviroment from OpenAI gym
        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()

        # There are 2 actions for the cart-pole enviroment
        # Set this programatically?
        self.action_space = 2


        # Define the two neural networks for Double Deep Q Learning
        self.QA = Sequential()
        self.QA.add(Dense(24, input_dim = 4, activation='relu'))
        self.QA.add(Dense(24, activation='relu'))
        # Try a softmax?
        self.QA.add(Dense(2, activation='linear'))
        self.QA.compile(loss = "mse",  optimizer = Adam(lr = 0.001))
        #self.QA.compile(loss = "mse",  optimizer = Adam(lr = 0.0000001))
        

        # Try the Huber loss
        #self.QA.compile(loss = "mean_squared_error",
        #optimizer = RMSprop(lr = 0.00025,
        #rho = 0.95,
        #epsilon = 0.01),
        #metrics = ["accuracy"])

        self.QB = copy.deepcopy(self.QA)

    def updateNetworks(self):
        self.QB = copy.deepcopy(self.QA)        
    
    
    # Be careful with nextState - is that really what you want?
    def target_value(self, current_state, next_state, immediateReward, done):
        """ Use the Bellman equation to compute the optimal
        action and its predicted cumulative reward """
        
        if (done):
            return -100.0

        # The network predicts the discounted value from that state
        # so predict the value for each state and then take the largest index with argmax
        next_action = np.argmax(self.QA.predict(np.array([next_state])))
    
        #print(self.QA.predict(np.array([state])))

        return immediateReward + (self.discount * self.QB.predict(np.array([next_state]))[0][next_action])
       
    # Be careful with nextState - is that really what you want?
    def target_value2(self, next_state, immediateReward, done, action):
        """ Use the Bellman equation to compute the optimal
        action and its predicted cumulative reward """

        if (done):
            return -100.0
        
        next_action = np.argmax(self.QA.predict(np.array([next_state])))
        
        return immediateReward + (self.discount * (self.QB.predict(np.array([next_state]))[0][next_action]))

    
    def train(self):
        """ Describe """
        
        # Get random sample from the ReplayMemory
        training_data = self.replayMemory.sample(self.batchSize)
        
        # Avoid training on the data garbage?
        # Add a check to make sure its not garbage!!!!!!!!

        # Split the data into input and label
        inputs = [""] * len(training_data)
        labels = [""] * len(training_data)

        for i in range(len(training_data)):
            # faster way to do this?
            if (training_data[i] != None):
                if (training_data[i].state is not None):
                    # We need to recompute the value using the new, target network 
                    #target_value(self, state, immediateReward, done)
                    action_value = self.target_value2(training_data[i].next_state, training_data[i].reward, training_data[i].done, training_data[i].action) 
                    
                    # Q_B is the values! QA is for action selection
                    predicted_values = self.QB.predict(np.array([training_data[i].state]))

                    training_data[i].action_values = predicted_values
                    training_data[i].actions_values[training_data[i].action] = action_value

                    inputs[i] = training_data[i].state
                    labels[i] = training_data[i].actions_values

    
        inputs = np.array(inputs)
        labels = np.array(labels)
     
        history = self.QA.fit(    
            inputs,
            labels,
            batch_size = 64,
            epochs = 1,
            verbose = 0
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )
             
    
        
    def saveNetworks(self):
        self.QA.save("neural_networks/A")
        self.QB.save("neural_networks/B")

    def resetGame(self):
        
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
    
    predicted_values = myAgent.QA.predict(np.array([current_state]))[0]
      
    action = np.argmax(predicted_values)
    
    if (random.random() < myAgent.epsilon):
        # Choose action randomly
        # FIX ME - depends on exact way the data is published in game
        action = random.randint(0, myAgent.action_space - 1)
       
    next_state, reward, done, info = myAgent.env.step(action)
        
    currentExperience = dataInstance(current_state, next_state, action, reward, done)
    
    # Label the data
    target_value = myAgent.target_value(current_state, next_state, reward, done)
    
    predicted_values = myAgent.QB.predict(np.array([current_state]))[0]
    currentExperience.actions_values = predicted_values
    currentExperience.actions_values[action] = target_value
    
    myAgent.replayMemory.append(currentExperience)

    current_state = next_state

    # Update counters
    myAgent.moveNumber = myAgent.moveNumber + 1
    myAgent.cumulativeReward = myAgent.cumulativeReward + reward
    
    if (done):
        myAgent.resetGame()
         
    
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
    






