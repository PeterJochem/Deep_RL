import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

"""Stores a single experience. Has data needed to compute target values"""
class experience():

    def __init__(self, state, next_state, action, reward, done, action_values = [0.0]):
         
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.actions_values = action_values # This is the "label" for the data

    def unpack(self):
        return self.state, self.next_state, self.reward, self.done, self.action


