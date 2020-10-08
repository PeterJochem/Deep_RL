import random
import numpy as np

class replayBuffer():
    """ This class describes a list of experiences"""
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


    def append(self, state, action, reward, next_state):
        """Rotating list. Evict oldest items if list if full"""
        
        self.states[self.index] = state     
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state


        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        
        max_index = min(self.size, self.max_size)
        
        random_experience_indexes = np.random.choice(max_index, batch_size)
        
        """
        if (self.size <= self.max_size and self.index > batch_size):
            #experiences = random.sample(self.buffer[:self.index - 1], batch_size)
            startIndex = 0
            stopIndex = self.index - 1

        elif (self.index <= batch_size):
            #experiences = self.buffer[:self.index - 1]
            startIndex = 0
            stopIndex = self.index - 1

        else:
            # experiences = random.sample(self.buffer, batch_size)
            startIndex = 0
            stopIndex = self.max_size
        """

        """
        states = [single_experience.state for single_experience in experiences]
        actions = [single_experience.action for single_experience in experiences]
        rewards = [single_experience.reward for single_experience in experiences]
        next_states = [single_experience.next_state for single_experience in experiences]
        """ 
        
        rS = self.states[random_experience_indexes]
        rA = self.actions[random_experience_indexes]        
        rR = self.rewards[random_experience_indexes]
        rNS = self.next_states[random_experience_indexes]
        
        return rS, rA, rR, rNS

