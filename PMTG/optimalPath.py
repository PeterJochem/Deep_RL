import numpy as np
import matplotlib.pyplot as plt

"""Generate and store a set of points that define a trajectory for 
the agent to follow. Trajectory is roughly a figure 8"""
class optimalPath:

    def __init__(self):

        self.numPts = 3000
        self.totalTime = 1.0
        
        self.currentIndex = 0

        self.x = np.zeros(self.numPts)
        self.y = np.zeros(self.numPts)
        
        # Parameterize the eight shape path
        self.a_x_start = 2.0
        self.a_y_start = 4.0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start

        for i in range(self.numPts):

            nextTime = (float(i)/self.numPts) * self.totalTime
            next_x = self.a_x * np.sin(2 * np.pi * nextTime)
            next_y = (self.a_y/2.0) * (np.sin(2 * np.pi * nextTime) * np.cos(2 * np.pi * nextTime))
            
            self.x[i] = next_x
            self.y[i] = next_y

            self.degradePath()
        
    """Change the parameters that determine the path so that the 
    optimal path is not a perfect figure 8"""
    def degradePath(self):
        self.a_x = self.a_x - (2 * self.a_x /self.numPts)
        self.a_y = self.a_y - (self.a_y / (2 * self.numPts))

    def viewPlot(self):

        plt.scatter(self.x, self.y)
        plt.show()

    def reset(self):
        
        self.curretIndex = 0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start
        
        return self.x[0], self.y[0]

    def reward(self, action):
        
        print(action)
        agent_x = action[0]
        agent_y = action[1]
        current_x, current_y = self.x[self.currentIndex], self.y[self.currentIndex] 

        return np.sqrt(((agent_x - current_x)**2) + ((agent_y - current_y)**2)) 
            
    def step(self):
        
        self.currentIndex = self.currentIndex + 1
        return self.x[self.currentIndex], self.y[self.currentIndex]       


    def isDone(self):

        if (self.currentIndex >= self.numPts):
            return True
        return False
    


#myPath = optimalPath()
#myPath.viewPlot()
