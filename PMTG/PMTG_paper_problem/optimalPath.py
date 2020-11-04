import numpy as np
import matplotlib.pyplot as plt

"""Generate and store a set of points that define a trajectory for 
the agent to follow. Trajectory is roughly a figure 8"""
class optimalPath:

    def __init__(self):

        self.numPts = 200
        self.totalTime = 1.0
        
        self.currentIndex = 0

        self.x = np.zeros(self.numPts)
        self.y = np.zeros(self.numPts)
        
        # Parameterize the eight shape path
        self.a_x_start = 2.0
        self.a_y_start = 2.0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start

        self.a_x_history = np.zeros(self.numPts)
        self.a_y_history = np.zeros(self.numPts)

        for i in range(self.numPts):

            nextTime = (float(i)/self.numPts) * self.totalTime
            next_x = self.a_x * np.sin(2 * np.pi * nextTime)
            next_y = (self.a_y/2.0) * (np.sin(2 * np.pi * nextTime) * np.cos(2 * np.pi * nextTime))
            
            self.x[i] = next_x
            self.y[i] = next_y

            self.degradePath()

            self.a_x_history[i] = self.a_x 
            self.a_y_history[i] = self.a_y
    
    """Describe me"""
    def computeTG_at_Index(self, time, a_x, a_y):
        
        time = (float(time)/self.numPts) * self.totalTime
        next_x = a_x * np.sin(2 * np.pi * time)
        next_y = (a_y/2.0) * (np.sin(2 * np.pi * time) * np.cos(2 * np.pi * time))
        
        return next_x, next_y

    """Change the parameters that determine the path so that the 
    optimal path is not a perfect figure 8"""
    def degradePath(self):
        self.a_x = self.a_x - (2 * self.a_x /self.numPts)/10.0
        self.a_y = self.a_y - (self.a_y / (2 * self.numPts))/10.0

    def viewPlot(self):
        plt.scatter(self.x[0:50], self.y[0:50])
        plt.show()

    def reset(self):
        
        self.currentIndex = 0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start
           
        return self.currentIndex, self.x[self.currentIndex], self.y[self.currentIndex], self.a_x, self.a_y

    def reward(self, x, y):
         
        optimal_x, optimal_y = self.x[self.currentIndex], self.y[self.currentIndex] 

        return -np.sqrt(((x - optimal_x)**2) + ((y - optimal_y)**2)) 
            
    def step(self, new_a_x, new_a_y, nn_x, nn_y):
        
        #self.currentIndex = self.currentIndex + 1
            
        # need to recompute the TG's at index value
        x_tg, y_tg = self.computeTG_at_Index(self.currentIndex, new_a_x, new_a_y)
        
        self.currentIndex = self.currentIndex + 1
        # [time, agent's_x, agent's_y, TG's a_x, TG's a_y]
        return self.currentIndex, (x_tg + nn_x), (y_tg + nn_y), new_a_x, new_a_y      

    def isDone(self):

        #if (self.currentIndex >= self.numPts - 2):
        if (self.currentIndex >= self.numPts - 1):    
            return True
        return False
    
#myPath = optimalPath()
#myPath.viewPlot()
