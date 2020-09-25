import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# This class describes a data instance
class dataInstance():

    # Constructor
    def __init__(self, state, next_state, action, reward, done, action_values = [0.0]):
         
        self.state = state
         
        self.reward = reward
        self.done = done
        
        self.action = action
        self.next_state = next_state

        # This is the "label" for the data
        self.actions_values = action_values
             
