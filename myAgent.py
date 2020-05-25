import numpy as np
import time
import random
import sys
import os
import gym

import warnings
warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    # from keras import backend as k
    import matplotlib.pyplot as plt
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Creates enviroment from OpenAI gym
env = gym.make('Breakout-v0')

observation = env.reset()
while (True):

    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    # time.sleep(0.2)
