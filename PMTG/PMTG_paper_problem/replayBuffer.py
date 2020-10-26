import random
import numpy as np

class replayBuffer():
    """ This class describes a rotating list of experiences"""
    def __init__(self, max_size, state_space_size, action_space_size):
               
        self.max_size = max_size
        self.index = 0
        self.size = 0
        
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        self.states = np.zeros((max_size, state_space_size))
        self.actions = np.zeros((max_size, action_space_size))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_space_size))
        self.dones = np.zeros((max_size, 1))

    def append(self, state, action, reward, next_state, done):
        """Rotating list"""
        
        self.states[self.index] = state     
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        
        if (done == True):
            self.dones[self.index] = 1.0
        else:
            self.dones[self.index] = 0.0

        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        max_index = min(self.size, self.max_size)
        
        random_experience_indexes = np.random.choice(max_index, batch_size)
        
        rS = self.states[random_experience_indexes]
        rA = self.actions[random_experience_indexes]        
        rR = self.rewards[random_experience_indexes]
        rNS = self.next_states[random_experience_indexes]
        rD = self.dones[random_experience_indexes]

        return rS, rA, rR, rNS, rD

