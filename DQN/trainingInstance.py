import numpy as np

class trainingInstance:

    def __init__(self, observationPrior, observation, reward, action, done, gameNum):

        self.observationPrior = observationPrior
        self.observation = observation
        self.reward = reward
        self.action = action
        self.gameNum = gameNum
        self.done = done


