import signal, os
from replayBuffer import *
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow import keras
from optimalPath import optimalPath 

allReward = [] # List of rewards over time, for logging and visualization

# FIX ME - I turned the noise off
useNoise = True

maxScore = -10000

def computeLearnedPath():
        
    all_x = []
    all_y = []

    time = 0
    current_state = myAgent.desiredPath.reset()
    myAgent.resetRandomProcess()

    while(True):
        action = []
        useNoise = False
        action = myAgent.chooseAction(current_state)[0]

        # observation, reward, done, info
        next_state, reward, done, info = myAgent.step(action)
        
        print(next_state)

        all_x.append(next_state[1])
        all_y.append(next_state[2])

        # Update counters
        myAgent.moveNumber = myAgent.moveNumber + 1
        myAgent.cumulativeReward = myAgent.cumulativeReward + reward

        if (done == True):
            myAgent.handleGameEnd()
            break

        current_state = next_state
        time = time + 1
    
    time = 0
    current_state = myAgent.desiredPath.reset()
    myAgent.resetRandomProcess()
    useNoise = True
    return all_x, all_y
    

"""Signal handler displays the agent's progress"""
def handler1(signum, frame):

    # Plot the data
    plt.plot(allReward)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()

    plt.scatter(myAgent.desiredPath.x, myAgent.desiredPath.y)
    
    # Compute the agent's path
    learned_x, learned_y = computeLearnedPath()
    plt.scatter(learned_x, learned_y)
    plt.show()

    useNoise = False

signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.discount = 0.99
        self.memorySize = 1000000 
        self.batch_size = 100 # Mini batch size for keras .fit method
        self.explorationSteps = 101
        self.trainEvery = 50

        self.moveNumber = 0 # Number of actions taken in the current episode 
        self.save = 50 # This describes how often to save each network to disk  

        self.currentGameNumber = 0 
        self.cumulativeReward = 0 # Current game's total reward
        
        # These parameters are specefic to the environment
        self.max_control_signal = 2.0
        self.lowerLimit = -2.0
        self.upperLimit = 2.0
        
        self.desiredPath = optimalPath()
        
        self.polyak_rate = 0.001
        self.action_space_size = 4 # Size of the action vector  
        # [a_x, a_y, u_x, u_y] 
        
        self.state_space_size = 5 # Size of the observation vector
        # [time, a_x, a_y, agent's_x, agent's_y]

        self.replayMemory = replayBuffer(self.memorySize, self.state_space_size, self.action_space_size)

        self.actor = self.defineActor()
        self.actor_target = self.defineActor()
        
        self.critic = self.defineCritic()
        self.critic_target = self.defineCritic()

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        
        # Actor's learning rate should be smaller
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0001

        self.critic_optim = tf.keras.optimizers.Adam(self.critic_learning_rate)
        self.actor_optim = tf.keras.optimizers.Adam(self.actor_learning_rate)

        # Random Process Hyper parameters
        std_dev = 0.0005 #0.05
        self.init_noise_process(average = np.zeros(self.action_space_size), std_dev = float(std_dev) * np.ones(self.action_space_size))

        
    def defineActor(self):

        actor_initializer = tf.random_uniform_initializer(minval = -0.003, maxval = 0.003)
        
        inputs = layers.Input(shape = (self.state_space_size, ))
        
        nextLayer = layers.BatchNormalization()(inputs) # Normalize this?
        nextLayer = layers.Dense(300)(nextLayer)
        nextLayer = layers.Activation("relu")(nextLayer)
        
        nextLayer = layers.Dense(200)(nextLayer)
        nextLayer = layers.Activation("relu")(nextLayer)

        # tanh maps into the interval of [-1, 1]
        outputs = layers.Dense(self.action_space_size, activation = "linear", kernel_initializer = actor_initializer)(nextLayer)
        
        # max_control signal is 2.0 for axis of the toy PMTG problem
        #outputs = outputs * self.max_control_signal
        return tf.keras.Model(inputs, outputs)
    
    def defineCritic(self):
       
        critic_initializer = tf.random_uniform_initializer(minval = -0.003, maxval = 0.003)

        state_inputs = layers.Input(shape=(self.state_space_size))
        
        state_stream = layers.Dense(200)(state_inputs)
        state_stream = layers.Activation("relu")(state_stream)

        action_inputs = layers.Input(shape = (self.action_space_size) )
        
        # Merge the two seperate information streams
        merged_stream = layers.Concatenate()([state_stream, action_inputs])
        merged_stream = layers.Dense(200)(merged_stream) 
        merged_stream = layers.Activation("relu")(merged_stream)

        outputs = layers.Dense(1, kernel_initializer = critic_initializer)(merged_stream)

        return tf.keras.Model([state_inputs, action_inputs], outputs)
    
    """Initialize an Ornstein–Uhlenbeck Process to generate correlated noise"""
    def init_noise_process(self, average, std_dev, theta = 0.15, dt = 0.01, x_start = None):
    
        self.theta = theta
        self.average = average
        self.std_dev = std_dev
        self.dt = dt
        self.x_start = x_start
        self.x_prior = np.zeros_like(self.average)
    
    """Generate another instance of the Ornstein–Uhlenbeck process"""
    def noise(self):
        noise = (self.x_prior + self.theta * (self.average - self.x_prior) * self.dt + 
                self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.average.shape) )
        
        self.x_prior = noise
        return noise
    
    """End Ornstein–Uhlenbeck process and start a new one"""
    def resetRandomProcess(self):
        std_dev = 0.2 
        self.init_noise_process(average = np.zeros(self.action_space_size), std_dev = float(std_dev) * np.ones(self.action_space_size))


    def chooseAction(self, state):

        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        
        action = self.actor(state)      
        
        """I turned the noise off for PMTG"""
        noise = self.noise()
        if (useNoise == True):
            action = action.numpy().squeeze() + noise
        else:
            action = action.numpy().squeeze()
    
        # Make sure action is withing legal range
        action = np.clip(action, self.lowerLimit, self.upperLimit)

        return [np.squeeze(action)]

    @tf.function
    def update(self, states, actions, rewards, next_states, isTerminals): 
    
        with tf.GradientTape() as tape:
            
            target_actions = self.actor_target(next_states, training = True)
            newVector = tf.cast(1 - isTerminals, dtype = tf.float32)
            predicted_values = rewards + self.discount * newVector * self.critic_target([next_states, target_actions], training = True)

            critic_value = self.critic([states, actions], training = True)
            critic_loss = tf.math.reduce_mean(tf.math.square(predicted_values - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            
            actions2 = self.actor(states, training = True)
            critic_value = self.critic([states, actions2], training = True)
            actor_loss = tf.math.reduce_mean(-1 * critic_value) # Remember to negate the loss!
                
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
       
    def train(self):
        
        states, actions, rewards, next_states, isTerminals = self.replayMemory.sample(self.batch_size) 
        
        # Convert to Tensorflow data types
        states  = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        isTerminals = tf.convert_to_tensor(isTerminals)

        self.update(states, actions, rewards, next_states, isTerminals)
    

    def handleGameEnd(self):

        print("Game number " + str(self.currentGameNumber) + " ended with a total reward of " + str(self.cumulativeReward + 10))
        allReward.append(self.cumulativeReward)
        self.cumulativeReward = 0
        self.currentGameNumber = self.currentGameNumber + 1

    def randomAction(self):
        # Assumes the action space is symmetric, for each entry in action vector
        return (np.random.rand(1, self.action_space_size) - self.upperLimit/2.0) * self.upperLimit
    
    def saveNetworks(self):
        self.actor.save_weights("networks/actor")
        self.critic.save_weights("networks/critic")
    
    def step(self, action):
        
        x_a_tg, y_a_tg, nn_modulation_x, nn_modulation_y = action

        # new_a_x, new_a_y, nn_x, nn_y 
        # self.currentIndex, (x_tg + nn_x), (y_tg + nn_y), new_a_x, new_a_y
        nextState = self.desiredPath.step(x_a_tg, y_a_tg, nn_modulation_x, nn_modulation_y)
    
        _, nextX, nextY, _, _ = nextState
        reward = self.desiredPath.reward(nextX, nextY)   
        isDone = self.desiredPath.isDone()
        info = None

        return nextState, reward, isDone, info 


"""Polyak averaging from online network to the target network"""
@tf.function
def update_target(target_weights, online_weights, polyak_rate):

    for (target, online) in zip(target_weights, online_weights):
        target.assign(online * polyak_rate + target * (1 - polyak_rate))


#tf.compat.v1.enable_eager_execution()
myAgent = Agent()    

totalStep = 0

while (True): 
    
    time = 0
    current_state = myAgent.desiredPath.reset()
    myAgent.resetRandomProcess()

    while(True):
        totalStep = totalStep + 1
        
        #print(time)
        action = []
        #if (totalStep < myAgent.explorationSteps):
        #    action = myAgent.randomAction().squeeze()
        #else:
        action = myAgent.chooseAction(current_state)[0]
            
        # observation, reward, done, info
        next_state, reward, done, info = myAgent.step(action) 
         
        myAgent.replayMemory.append(current_state, action, reward, next_state, done)
        
        # Update counters
        myAgent.moveNumber = myAgent.moveNumber + 1
        myAgent.cumulativeReward = myAgent.cumulativeReward + reward
    
        if (done == True):
            myAgent.handleGameEnd()
            break
        
        if (((totalStep % myAgent.trainEvery) == 0) and (totalStep > myAgent.explorationSteps)):
            for i in range(myAgent.trainEvery):
                myAgent.train()
                update_target(myAgent.actor_target.variables, myAgent.actor.variables, myAgent.polyak_rate)
                update_target(myAgent.critic_target.variables, myAgent.critic.variables, myAgent.polyak_rate)
    
        
        current_state = next_state
        time = time + 1

            
