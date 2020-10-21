import signal, os
from replayBuffer import *
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow import keras

allReward = [] # List of rewards over time, for logging and visualization
useNoise = True

maxScore = -10000

"""Signal handler displays the agent's progress"""
def handler1(signum, frame):

    # Plot the data
    plt.plot(allReward)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()

    useNoise = False

signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.discount = 0.99
        self.memorySize = 1000000 
        self.batch_size = 100 # Mini batch size for keras .fit method
        self.explorationSteps = 1000
        self.trainEvery = 50
        self.policy_delay = 2

        self.moveNumber = 0 # Number of actions taken in the current episode 
        self.save = 50 # This describes how often to save each network to disk  

        self.currentGameNumber = 0 
        self.cumulativeReward = 0 # Current game's total reward
        
        # These parameters are Specefic to the Hopper-V2
        self.max_control_signal = 1.0
        self.lowerLimit = -1.0
        self.upperLimit = 1.0

        self.env = gym.make('Hopper-v2')
        self.state = self.env.reset()
        
        self.polyak_rate = 0.001
        self.action_space_size = 3 # Size of the action vector  
        self.state_space_size = 11 # Size of the observation vector
        
        self.replayMemory = replayBuffer(self.memorySize, self.state_space_size, self.action_space_size)

        self.actor = self.defineActor()
        self.actor_target = self.defineActor()
        
        self.critic_a = self.defineCritic()
        self.critic_target_a = self.defineCritic()

        self.critic_b = self.defineCritic()
        self.critic_target_b = self.defineCritic()

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target_a.set_weights(self.critic_a.get_weights())
        self.critic_target_b.set_weights(self.critic_b.get_weights())

        # Actor's learning rate should be smaller
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0001

        self.critic_a_optim = tf.keras.optimizers.Adam(self.critic_learning_rate)
        self.critic_b_optim = tf.keras.optimizers.Adam(self.critic_learning_rate)
        self.actor_optim = tf.keras.optimizers.Adam(self.actor_learning_rate)

        # Random Process Hyper parameters
        std_dev = 0.2
        self.init_noise_process(average = np.zeros(self.action_space_size), std_dev = float(std_dev) * np.ones(self.action_space_size))

        
    def defineActor(self):

        actor_initializer = tf.random_uniform_initializer(minval = -0.003, maxval = 0.003)
        
        inputs = layers.Input(shape = (self.state_space_size, ))
        
        nextLayer = layers.BatchNormalization()(inputs) # Normalize this?
        nextLayer = layers.Dense(400)(nextLayer)
        nextLayer = layers.BatchNormalization()(nextLayer)
        nextLayer = layers.Activation("relu")(nextLayer)
        
        nextLayer = layers.Dense(300)(nextLayer)
        nextLayer = layers.BatchNormalization()(nextLayer)
        nextLayer = layers.Activation("relu")(nextLayer)

        # tanh maps into the interval of [-1, 1]
        outputs = layers.Dense(self.action_space_size, activation = "tanh", kernel_initializer = actor_initializer)(nextLayer)
        
        # max_control signal is 1.0 for EACH joint of the Hopper robot
        outputs = outputs * self.max_control_signal
        return tf.keras.Model(inputs, outputs)
    
    def defineCritic(self):
       
        critic_initializer = tf.random_uniform_initializer(minval = -0.003, maxval = 0.003)

        state_inputs = layers.Input(shape=(self.state_space_size))
        
        state_stream = layers.BatchNormalization()(state_inputs) # Normalize this?
        state_stream = layers.Dense(400)(state_stream)
        state_stream = layers.BatchNormalization()(state_stream)
        state_stream = layers.Activation("relu")(state_stream)

        action_inputs = layers.Input(shape = (self.action_space_size) )
        
        # Merge the two seperate information streams
        merged_stream = layers.Concatenate()([state_stream, action_inputs])
        merged_stream = layers.Dense(300)(merged_stream) 
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

        noise = self.noise()
        if (useNoise == True):
            action = action.numpy() + noise
        else:
            action = action.numpy()

        # Make sure action is withing legal range
        action = np.clip(action, self.lowerLimit, self.upperLimit)

        return [np.squeeze(action)]

    @tf.function
    def update_critics(
        self, states, actions, rewards, next_states, isTerminals
    ): 
    
        with tf.GradientTape(persistent = True) as tape:
            
            target_actions = self.actor_target(next_states, training = True)
            newVector = tf.cast(1 - isTerminals, dtype = tf.float32)
            
            predicted_values_1 = rewards + self.discount * newVector * self.critic_target_a([next_states, target_actions], training = True)
            predicted_values_2 = rewards + self.discount * newVector * self.critic_target_b([next_states, target_actions], training = True)
    
            predicted_values = tf.math.minimum(predicted_values_1, predicted_values_2)
            predicted_values = tf.convert_to_tensor(predicted_values)    
        
            critic_a_value = self.critic_a([states, actions], training = True)
            critic_b_value = self.critic_b([states, actions], training = True)
        
            critic_a_loss = tf.math.reduce_mean(tf.math.square(predicted_values - critic_a_value))
            critic_b_loss = tf.math.reduce_mean(tf.math.square(predicted_values - critic_b_value))

        critic_a_grad = tape.gradient(critic_a_loss, self.critic_a.trainable_variables)
        critic_b_grad = tape.gradient(critic_b_loss, self.critic_b.trainable_variables)
        
        self.critic_a_optim.apply_gradients(zip(critic_a_grad, self.critic_a.trainable_variables))
        self.critic_b_optim.apply_gradients(zip(critic_b_grad, self.critic_b.trainable_variables))

       
    @tf.function
    def update_actor(
        self, states, actions, rewards, next_states, isTerminals
    ):

        with tf.GradientTape() as tape:

            actor_actions = self.actor(states, training = True)
            critic_a_value = self.critic_a([states, actor_actions], training = True) # Which critic do I use?
            actor_loss = tf.math.reduce_mean(-1 * critic_a_value) # Remember to negate the loss!

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def train_actor(self):
        
        states, actions, rewards, next_states, isTerminals = self.replayMemory.sample(self.batch_size) 
    
        # Convert to Tensorflow data types
        states  = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        isTerminals = tf.convert_to_tensor(isTerminals)

        self.update_actor(states, actions, rewards, next_states, isTerminals)
    
    def train_critics(self):

        states, actions, rewards, next_states, isTerminals = self.replayMemory.sample(self.batch_size)

        # Convert to Tensorflow data types
        states  = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        isTerminals = tf.convert_to_tensor(isTerminals)

        self.update_critics(states, actions, rewards, next_states, isTerminals)

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



"""Polyak averaging from online network to the target network"""
@tf.function
def update_target(target_weights, online_weights, polyak_rate):

    for (target, online) in zip(target_weights, online_weights):
        target.assign(online * polyak_rate + target * (1 - polyak_rate))


#tf.compat.v1.enable_eager_execution()
myAgent = Agent()    

totalStep = 0

while (True): 
    current_state = myAgent.env.reset()
    myAgent.resetRandomProcess()

    while(True):
        myAgent.env.render()

        totalStep = totalStep + 1
        
        action = []
        if (totalStep < myAgent.explorationSteps):
            action = myAgent.randomAction()
        else:
            action = myAgent.chooseAction(current_state)[0]
        
        # observation, reward, done, info
        next_state, reward, done, info = myAgent.env.step(action)
    
        # state, action, reward, next_state
        if (done == True):
            reward = -10.0 # Does it work without it?

        myAgent.replayMemory.append(current_state, action, reward, next_state, done)
        
        # Update counters
        myAgent.moveNumber = myAgent.moveNumber + 1
        myAgent.cumulativeReward = myAgent.cumulativeReward + reward
    
        if (done == True):
            myAgent.handleGameEnd()
            break
        
        if (((totalStep % myAgent.trainEvery) == 0) and (totalStep > myAgent.explorationSteps)):
            for i in range(myAgent.trainEvery):
       
                myAgent.train_critics()
        
                if (i % myAgent.policy_delay == 0):
                    
                    myAgent.train_actor()
                    update_target(myAgent.actor_target.variables, myAgent.actor.variables, myAgent.polyak_rate)
                    update_target(myAgent.critic_target_a.variables, myAgent.critic_a.variables, myAgent.polyak_rate)
                    update_target(myAgent.critic_target_b.variables, myAgent.critic_b.variables, myAgent.polyak_rate)
    

        
        """
        if (myAgent.cumulativeReward > maxScore):
            maxScore = myAgent.cumulativeReward
            myAgent.saveNetworks()
        """

        current_state = next_state

            
